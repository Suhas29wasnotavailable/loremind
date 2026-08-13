"""Single source of truth for LoreMind configuration.

Every module reads its settings from here. The embedding model in particular must
never be declared in more than one place: ingestion and query time used to name
different models, and because both happened to be 384-dimensional, Qdrant
accepted the mismatched vectors without complaint and returned nonsense.
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env support.

    Only handles `KEY=value` with optional quotes and `#` comments, which is all
    this project needs — not worth a dependency. Real environment variables
    always win, so a deployed container's config cannot be overridden by a
    stray .env baked into the image.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- corpus

LORE_PATH = Path(_env("LOREMIND_LORE_PATH", str(BACKEND_DIR / "data" / "starwars_lore.txt")))

# Facts per chunk and how far the window advances. Overlap keeps a fact from
# being cut off from the sentence that gives it context.
CHUNK_LINES = _env_int("LOREMIND_CHUNK_LINES", 6)
CHUNK_STRIDE = _env_int("LOREMIND_CHUNK_STRIDE", 5)

# ---------------------------------------------------------------- models

# Bumping either of these invalidates the index. INDEX_FINGERPRINT is written
# into the collection at build time and verified at query time, so a stale index
# fails loudly instead of silently degrading retrieval.
EMBED_MODEL = _env("LOREMIND_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = _env_int("LOREMIND_EMBED_DIM", 384)
# ONNX export of cross-encoder/ms-marco-MiniLM-L-6-v2. Same weights: logits agree
# with the PyTorch version to three decimals (see compare_backends.py).
RERANK_MODEL = _env("LOREMIND_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# Where fastembed keeps downloaded ONNX models. Set explicitly in the container
# so the download happens at build time rather than on the first request.
MODEL_CACHE_DIR = Path(_env("LOREMIND_MODEL_CACHE_DIR", str(PROJECT_DIR / ".model_cache")))

# bge-* models are trained with an asymmetric instruction: queries get this
# prefix, passages get none. Omitting it measurably degrades recall.
EMBED_QUERY_PREFIX = _env(
    "LOREMIND_EMBED_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)

INDEX_FINGERPRINT = (
    f"{EMBED_MODEL}|onnx|dim={EMBED_DIM}|lines={CHUNK_LINES}|stride={CHUNK_STRIDE}|v3"
)

# ---------------------------------------------------------------- vector store

COLLECTION_NAME = _env("LOREMIND_COLLECTION", "starwars_lore")

# Empty QDRANT_URL => embedded mode: qdrant-client runs in-process against a
# local file, so there is no server to run, host, or pay for. Set QDRANT_URL to
# point at a real Qdrant (e.g. the docker container) for local development.
QDRANT_URL = _env("LOREMIND_QDRANT_URL", "")
QDRANT_API_KEY = _env("LOREMIND_QDRANT_API_KEY", "")
QDRANT_PATH = Path(_env("LOREMIND_QDRANT_PATH", str(PROJECT_DIR / ".qdrant_local")))

# Rebuild the embedded index at startup if it is missing or stale. Cheap for a
# corpus this size and it makes deploys stateless.
AUTO_INDEX = _env("LOREMIND_AUTO_INDEX", "1") not in {"0", "false", "False", ""}

# ---------------------------------------------------------------- retrieval

SEMANTIC_TOP_K = _env_int("LOREMIND_SEMANTIC_TOP_K", 12)
BM25_TOP_K = _env_int("LOREMIND_BM25_TOP_K", 12)
FINAL_TOP_K = _env_int("LOREMIND_FINAL_TOP_K", 5)

# How many fused candidates reach the cross-encoder. It is ~98% of retrieval
# cost and scales linearly with this number, so it dominates latency on a
# CPU-throttled host. Measured with bench_candidates.py: reranking the top 8 by
# RRF gives an identical grounding separation to reranking all ~20 (on-corpus min
# 2.51, off-corpus max -6.72 either way) for 2.6x less work. Below 6 the margin
# starts to narrow.
RERANK_CANDIDATES = _env_int("LOREMIND_RERANK_CANDIDATES", 8)

# Reciprocal Rank Fusion damping. 60 is the value from the original RRF paper and
# is insensitive enough that tuning it is not worthwhile at this corpus size.
RRF_K = _env_int("LOREMIND_RRF_K", 60)

# Grounding gate, applied to the best cross-encoder score. Calibrated with
# calibrate.py over 22 answerable and 10 unanswerable questions: answerable
# queries scored 2.51..9.66, unanswerable ones -6.72..-11.24. Any threshold in
# that gap separates them perfectly, so this sits at the midpoint to leave the
# widest margin for phrasings the calibration set did not cover.
GROUNDING_MIN_SCORE = _env_float("LOREMIND_GROUNDING_MIN_SCORE", -2.0)

# Once we have decided to answer, this decides which passages are worth
# including. Deliberately more permissive than the gate: a supporting passage
# does not need to answer the question on its own to add useful detail.
CONTEXT_MIN_SCORE = _env_float("LOREMIND_CONTEXT_MIN_SCORE", -6.0)

NO_CONTEXT_MESSAGE = (
    "The archives contain little surviving information regarding this subject."
)

# ---------------------------------------------------------------- generation

# "ollama" for local development, "groq" for deployment. Both speak to a
# llama-3 family model so the prompt behaves consistently across the two.
LLM_PROVIDER = _env("LOREMIND_LLM_PROVIDER", "ollama").lower()

OLLAMA_MODEL = _env("LOREMIND_OLLAMA_MODEL", "llama3")
OLLAMA_HOST = _env("LOREMIND_OLLAMA_HOST", "http://localhost:11434")

GROQ_MODEL = _env("LOREMIND_GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = _env("GROQ_API_KEY", "")

# Overridable so the provider can be pointed at a mock for tests, or at any
# other OpenAI-compatible endpoint (Cerebras, OpenRouter, a local vLLM).
GROQ_BASE_URL = _env("LOREMIND_GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")

MAX_OUTPUT_TOKENS = _env_int("LOREMIND_MAX_OUTPUT_TOKENS", 500)
TEMPERATURE = _env_float("LOREMIND_TEMPERATURE", 0.7)
NUM_CTX = _env_int("LOREMIND_NUM_CTX", 8192)
GENERATION_TIMEOUT = _env_float("LOREMIND_GENERATION_TIMEOUT", 120.0)

# ---------------------------------------------------------------- api

# Hard cap on the query. Without it a large body both wedges the CPU-bound
# reranker and pushes the system prompt out of the model's context window.
MAX_QUERY_CHARS = _env_int("LOREMIND_MAX_QUERY_CHARS", 400)

# Bounds how many generations run at once. A single small LLM endpoint serialises
# anyway; queueing here keeps latency honest instead of thrashing.
MAX_CONCURRENT_QUERIES = _env_int("LOREMIND_MAX_CONCURRENT", 4)

# Per-client request budget. Generous for a human reading crawl-speed answers,
# restrictive enough that one script cannot drain a free-tier quota.
RATE_LIMIT_PER_MINUTE = _env_int("LOREMIND_RATE_LIMIT_PER_MINUTE", 12)
RATE_LIMIT_BURST = _env_int("LOREMIND_RATE_LIMIT_BURST", 5)

# Trust forwarding headers only behind a proxy that sets them (Spaces, Fly, Render
# do). Off by default: with it on, header handling must be exactly right or a
# client can forge its own identity and escape the rate limiter.
TRUST_PROXY_HEADERS = _env("LOREMIND_TRUST_PROXY_HEADERS", "0") not in {"0", "false", "False", ""}

# Number of proxies between the client and this app that append to
# X-Forwarded-For. The client controls the LEFT of that list; each trusted proxy
# appends the peer it actually saw on the RIGHT. So the real client address is
# the Nth entry counting from the right, and reading the left-most entry — as
# this originally did — reads attacker-supplied data. Verified against the live
# deployment: rotating the header gave 10/10 requests through a limiter that
# allows 5.
TRUSTED_PROXY_HOPS = _env_int("LOREMIND_TRUSTED_PROXY_HOPS", 1)

# Cloudflare sets this to the real client address and strips any client-supplied
# copy, so it is preferred over X-Forwarded-For when present. Render serves
# through Cloudflare, so it normally is.
CLIENT_IP_HEADER = _env("LOREMIND_CLIENT_IP_HEADER", "cf-connecting-ip").lower()

# Comma-separated list, or "*" to allow any origin.
CORS_ORIGINS = [o.strip() for o in _env("LOREMIND_CORS_ORIGINS", "*").split(",") if o.strip()]

FRONTEND_DIR = Path(_env("LOREMIND_FRONTEND_DIR", str(PROJECT_DIR / "frontend")))
SERVE_FRONTEND = _env("LOREMIND_SERVE_FRONTEND", "1") not in {"0", "false", "False", ""}
