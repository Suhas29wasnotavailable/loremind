"""Generation backends.

Two providers behind one async streaming interface:

* ``ollama``  — local development. Nothing leaves the machine, but an 8B model
  saturates a laptop's GPU and fans.
* ``groq``    — deployment. Free tier, llama-3 family (so the prompt behaves the
  same), and the work happens off-device.

Both are exposed as async generators. The original implementation used a *sync*
generator inside a StreamingResponse, which meant CPU-bound retrieval and a
long-lived generation both occupied a threadpool worker while holding the GIL —
one oversized request could stall every other request in the process.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from backend import config


class LLMError(RuntimeError):
    """Raised when the configured provider cannot produce a response."""


# ------------------------------------------------------------------ ollama


async def _stream_ollama(prompt: str) -> AsyncIterator[str]:
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_OUTPUT_TOKENS,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "num_ctx": config.NUM_CTX,
        },
    }

    url = f"{config.OLLAMA_HOST.rstrip('/')}/api/chat"
    timeout = httpx.Timeout(config.GENERATION_TIMEOUT, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")[:300]
                    raise LLMError(f"ollama returned {response.status_code}: {body}")

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("error"):
                        raise LLMError(str(event["error"]))
                    token = (event.get("message") or {}).get("content") or ""
                    if token:
                        yield token
                    if event.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise LLMError(f"cannot reach ollama at {config.OLLAMA_HOST}: {exc}") from exc


# -------------------------------------------------------------------- groq


async def _stream_groq(prompt: str) -> AsyncIterator[str]:
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set")

    payload = {
        "model": config.GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_OUTPUT_TOKENS,
        "top_p": 0.95,
    }
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    url = f"{config.GROQ_BASE_URL}/chat/completions"
    timeout = httpx.Timeout(config.GENERATION_TIMEOUT, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")[:300]
                    raise LLMError(f"groq returned {response.status_code}: {body}")

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        if data == "[DONE]":
                            break
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    token = (choices[0].get("delta") or {}).get("content") or ""
                    if token:
                        yield token
        except httpx.HTTPError as exc:
            raise LLMError(f"cannot reach groq: {exc}") from exc


# ------------------------------------------------------------------ public

_PROVIDERS = {"ollama": _stream_ollama, "groq": _stream_groq}


def stream_tokens(prompt: str) -> AsyncIterator[str]:
    """Stream generated tokens from the configured provider."""
    try:
        provider = _PROVIDERS[config.LLM_PROVIDER]
    except KeyError:
        raise LLMError(
            f"unknown LOREMIND_LLM_PROVIDER {config.LLM_PROVIDER!r}; "
            f"expected one of {sorted(_PROVIDERS)}"
        ) from None
    return provider(prompt)


async def health() -> dict:
    """Cheap reachability probe for the configured provider."""
    if config.LLM_PROVIDER == "groq":
        if not config.GROQ_API_KEY:
            return {"provider": "groq", "ok": False, "detail": "GROQ_API_KEY not set"}
        return {"provider": "groq", "ok": True, "model": config.GROQ_MODEL}

    url = f"{config.OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            names = [m.get("name", "") for m in (response.json().get("models") or [])]
    except Exception as exc:  # noqa: BLE001 - health must never raise
        return {"provider": "ollama", "ok": False, "detail": str(exc)[:200]}

    wanted = config.OLLAMA_MODEL
    present = any(n == wanted or n.split(":")[0] == wanted.split(":")[0] for n in names)
    return {
        "provider": "ollama",
        "ok": present,
        "model": wanted,
        "detail": None if present else f"model not pulled; available: {names}",
    }
