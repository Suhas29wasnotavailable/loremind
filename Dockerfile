# LoreMind — single container serving both the API and the frontend.
#
# Sized for a free CPU tier: no GPU, no Qdrant server, no Ollama. Generation is
# delegated to Groq's free tier; retrieval runs locally on CPU because the
# embedding model (33M params) and reranker (22M) are small enough to be fast
# there. Torch is installed from the CPU index, which avoids ~2GB of CUDA
# libraries that would otherwise dominate the image.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the HF cache inside the image so model download happens at build
    # time, not on the first user request.
    HF_HOME=/app/.cache/huggingface \
    # BLAS thread thrash hurts more than it helps on small free-tier boxes.
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch first, so the heavier default wheel is never resolved.
RUN pip install --no-cache-dir \
        torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

# Bake the models and the vector index into the image. Without this the first
# request after every cold start pays ~30s of model download plus indexing.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from backend import config; \
SentenceTransformer(config.EMBED_MODEL); \
CrossEncoder(config.RERANK_MODEL); \
print('models cached')" \
 && python -m backend.ingest \
 && chmod -R a+rX /app/.cache /app/.qdrant_local

ENV LOREMIND_LLM_PROVIDER=groq \
    LOREMIND_SERVE_FRONTEND=1 \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api',timeout=5).status==200 else 1)"

# Single worker: each one loads its own copy of the models, and the free tier
# does not have the memory for two.
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 75"]
