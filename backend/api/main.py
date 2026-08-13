"""LoreMind HTTP API."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend import config, llm
from backend.generation.rag_pipeline import answer
from backend.logging_setup import setup_logging
from backend.ratelimit import RateLimiter

setup_logging()
log = logging.getLogger("loremind.api")

# Bounds concurrent generations. Without this, load simply queues inside the
# provider while every request holds server resources.
_slots = asyncio.Semaphore(config.MAX_CONCURRENT_QUERIES)

_limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE, config.RATE_LIMIT_BURST)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the models at startup so the first user request is not also paying
    # ~30s of model loading. Failure here is logged, not fatal: /health will
    # report it and the app can still serve the frontend.
    from backend.retrieval import get_retriever

    try:
        await anyio.to_thread.run_sync(get_retriever)
        log.info("retriever ready")
    except Exception as exc:  # noqa: BLE001 - degrade rather than refuse to boot
        log.exception("retriever failed to initialise: %s", exc)
    yield


app = FastAPI(title="LoreMind", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Pydantic's max_length rejects an over-long query, but only after the whole
# body has been read into memory. Refusing on Content-Length first means a large
# upload cannot cost us the allocation at all.
_MAX_BODY_BYTES = config.MAX_QUERY_CHARS * 8 + 2048


@app.middleware("http")
async def limit_body_size(request, call_next):
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > _MAX_BODY_BYTES:
        return JSONResponse(
            {"detail": f"request body exceeds {_MAX_BODY_BYTES} bytes"},
            status_code=413,
        )
    return await call_next(request)


class QueryRequest(BaseModel):
    # max_length is the load-bearing fix: an unbounded query both wedged the
    # CPU-bound reranker and pushed the system prompt out of the context window.
    query: str = Field(..., min_length=1, max_length=config.MAX_QUERY_CHARS)

    model_config = {"extra": "forbid"}

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("query must contain non-whitespace characters")
        return cleaned


def _sse(event: str, data: object) -> str:
    """Encode one SSE event.

    The payload is always JSON. The original protocol sent raw token text after
    `data: `, which broke whenever a token contained a newline and made the
    terminating `[DONE] {json}` frame ambiguous.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _event_stream(query: str):
    try:
        async with _slots:
            stream, sources, fallback = await answer(query)

            yield _sse("sources", sources)

            if stream is None:
                yield _sse("token", fallback)
                yield _sse("done", {"grounded": False})
                return

            try:
                async for token in stream:
                    yield _sse("token", token)
            except llm.LLMError as exc:
                log.error("generation failed: %s", exc)
                yield _sse("error", "The archives are unreachable. Please try again.")
                yield _sse("done", {"grounded": False})
                return

            yield _sse("done", {"grounded": True})

    except asyncio.CancelledError:
        # Client disconnected. Let it propagate so the generation is abandoned
        # instead of burning capacity for a response nobody will read.
        log.info("client disconnected mid-stream")
        raise
    except Exception as exc:  # noqa: BLE001 - never leak a traceback to the client
        log.exception("unhandled error answering %r: %s", query[:80], exc)
        yield _sse("error", "The Force encountered a disturbance. Please try again.")
        yield _sse("done", {"grounded": False})


def _client_id(request: Request) -> tuple[str, str]:
    """Resolve the rate-limiting identity. Returns (identity, source).

    Only data a trusted proxy vouches for may be used. X-Forwarded-For is a list
    the client controls the left of: each proxy appends the peer address it
    actually saw, so the trustworthy entries are the right-most ones. Reading
    index 0 — which this originally did — reads whatever the caller sent, letting
    anyone rotate the header and bypass the limiter entirely.
    """
    if not config.TRUST_PROXY_HEADERS:
        return (request.client.host if request.client else "unknown"), "peer"

    # Cloudflare overwrites this with the real client address and drops any
    # client-supplied copy, so it cannot be forged from outside.
    vouched = request.headers.get(config.CLIENT_IP_HEADER, "").strip()
    if vouched:
        return vouched, config.CLIENT_IP_HEADER

    forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if forwarded:
        hops = max(1, config.TRUSTED_PROXY_HOPS)
        # Count from the right; clamp so a short list cannot fall back to the
        # client-controlled left-hand entries.
        if len(forwarded) >= hops:
            return forwarded[-hops], f"xff[-{hops}]"
        return forwarded[-1], "xff[-1]"

    return (request.client.host if request.client else "unknown"), "peer"


@app.post("/query")
async def query_loremind(payload: QueryRequest, request: Request):
    client, _ = _client_id(request)
    allowed, retry_after = _limiter.check(client)
    if not allowed:
        return JSONResponse(
            {"detail": "Too many requests. The archives need a moment."},
            status_code=429,
            headers={"Retry-After": str(int(retry_after))},
        )

    return StreamingResponse(
        _event_stream(payload.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    from backend.retrieval import get_retriever

    try:
        retriever = await anyio.to_thread.run_sync(get_retriever)
        index = retriever.stats()
    except Exception as exc:  # noqa: BLE001
        index = {"ok": False, "detail": str(exc)[:200]}

    generation = await llm.health()
    ok = bool(index.get("ok")) and bool(generation.get("ok"))
    return {"ok": ok, "index": index, "generation": generation}


@app.get("/whoami")
def whoami(request: Request):
    """Which address the rate limiter attributes this request to, and why.

    Exists because getting proxy-header handling wrong is silent: the limiter
    still returns 429s, just for the wrong identity. Reveals only the caller's
    own address.
    """
    client, source = _client_id(request)
    return {
        "client": client,
        "source": source,
        "trust_proxy_headers": config.TRUST_PROXY_HEADERS,
        "trusted_proxy_hops": config.TRUSTED_PROXY_HOPS,
        "forwarded_chain_length": len(
            [p for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        ),
        "vouched_header_present": bool(request.headers.get(config.CLIENT_IP_HEADER)),
    }


@app.get("/api")
def api_info():
    return {
        "message": "LoreMind API is running.",
        "provider": config.LLM_PROVIDER,
        "max_query_chars": config.MAX_QUERY_CHARS,
    }


# Serving the frontend from the same origin removes the hardcoded
# http://127.0.0.1:8000 in the page and makes CORS irrelevant in production.
if config.SERVE_FRONTEND and config.FRONTEND_DIR.is_dir():

    @app.get("/")
    def index():
        return FileResponse(config.FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
    log.info("serving frontend from %s", config.FRONTEND_DIR)
