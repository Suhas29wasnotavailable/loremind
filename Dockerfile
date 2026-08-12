# LoreMind — single container serving both the API and the frontend.
#
# No GPU, no Qdrant server, no Ollama. Retrieval runs locally on ONNX Runtime
# (the embedding model is 33M params, the reranker 22M), and generation is
# delegated to Groq's free tier. Fits comfortably in a 512MB free tier.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOREMIND_MODEL_CACHE_DIR=/app/.model_cache \
    # ONNX Runtime spawns a thread per core and an allocation arena per thread.
    # On a shared free-tier box (Render free is 0.1 CPU) extra threads cost
    # memory without buying throughput. Measured: 450MB peak under a 512MB cap
    # with 6 concurrent requests, versus OOM risk at the default thread count.
    OMP_NUM_THREADS=1 \
    ORT_NUM_THREADS=1

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

# Bake the ONNX models and the vector index into the image so a cold start does
# not download ~120MB or spend time indexing before serving the first request.
RUN python -m backend.ingest \
 && chmod -R a+rX /app/.model_cache /app/.qdrant_local

ENV LOREMIND_LLM_PROVIDER=groq \
    LOREMIND_SERVE_FRONTEND=1 \
    LOREMIND_TRUST_PROXY_HEADERS=1 \
    LOREMIND_MAX_CONCURRENT=2 \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('PORT','7860')+'/api',timeout=5).status==200 else 1)"

# Single worker, and not only for memory: embedded Qdrant takes an exclusive
# lock on its directory, so a second worker would fail to start.
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 75"]
