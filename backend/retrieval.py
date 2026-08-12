"""Hybrid retrieval: dense + BM25, fused by a cross-encoder, with a grounding gate.

Guarantees this module enforces, each of which was previously violated:

* the model used to embed the query is the model that built the index — checked
  against a fingerprint stored in the collection, so a stale index fails loudly;
* BM25 and the vector store search the *same* chunk list;
* if the reranker says nothing is relevant, we say so instead of handing an
  irrelevant context to the model and letting it improvise.

Both models run through fastembed (ONNX) rather than sentence-transformers
(PyTorch). Same weights, and verified numerically equivalent: query embeddings
agree to cosine 0.999998 and cross-encoder logits agree to three decimals across
the calibration set. Dropping torch takes the deployed image from 3.6GB to a size
that fits a free tier. See compare_backends.py.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from backend import config
from backend.corpus import Chunk, load_chunks, tokenize

log = logging.getLogger("loremind.retrieval")

_FINGERPRINT_ID = 0
_FINGERPRINT_KEY = "__loremind_index_fingerprint__"


class IndexMismatch(RuntimeError):
    """The collection was built with different settings than we are querying with."""


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm else array


class Retriever:
    """Owns the models, the vector store and the BM25 index."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.chunks: list[Chunk] = load_chunks()
        if not self.chunks:
            raise RuntimeError(f"no chunks parsed from {config.LORE_PATH}")

        log.info("loaded %d chunks from %s", len(self.chunks), config.LORE_PATH)

        # BM25 over the identical chunk list the vector store holds.
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

        cache = str(config.MODEL_CACHE_DIR) if config.MODEL_CACHE_DIR else None

        log.info("loading embedding model %s", config.EMBED_MODEL)
        self.embedder = TextEmbedding(config.EMBED_MODEL, cache_dir=cache)

        # Probe the real output width rather than trusting the config, so a model
        # swap cannot silently produce vectors the collection will not accept.
        actual_dim = len(self._embed_texts(["dimension probe"])[0])
        if actual_dim != config.EMBED_DIM:
            raise IndexMismatch(
                f"{config.EMBED_MODEL} produces {actual_dim}-dim vectors but "
                f"LOREMIND_EMBED_DIM is {config.EMBED_DIM}"
            )

        log.info("loading reranker %s", config.RERANK_MODEL)
        self.reranker = TextCrossEncoder(config.RERANK_MODEL, cache_dir=cache)

        self.client = self._connect()
        self._ensure_index()

    # ------------------------------------------------------------ models

    def _embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Embed passages. fastembed returns a generator of raw vectors."""
        return [_unit(v) for v in self.embedder.embed(texts)]

    def _embed_query(self, query: str) -> list[float]:
        # bge-* models expect a retrieval instruction on queries only. Note that
        # fastembed's query_embed() does NOT apply it (verified: cosine 0.96
        # against the prefixed vector), so it is applied explicitly here.
        return self._embed_texts([config.EMBED_QUERY_PREFIX + query])[0].tolist()

    # ------------------------------------------------------------ index

    def _connect(self) -> QdrantClient:
        if config.QDRANT_URL:
            log.info("connecting to qdrant at %s", config.QDRANT_URL)
            return QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY or None,
                timeout=15,
            )
        # Embedded mode: no server, no container, no hosting cost. Note this
        # takes an exclusive lock on the directory, so only one process at a
        # time may use it — hence a single uvicorn worker.
        config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        log.info("using embedded qdrant at %s", config.QDRANT_PATH)
        try:
            return QdrantClient(path=str(config.QDRANT_PATH))
        except RuntimeError as exc:
            raise RuntimeError(
                f"{config.QDRANT_PATH} is locked by another process. Embedded Qdrant "
                f"allows a single accessor: stop the running server before reindexing, "
                f"or set LOREMIND_QDRANT_URL to use a Qdrant server."
            ) from exc

    def _stored_fingerprint(self) -> str | None:
        try:
            points = self.client.retrieve(
                collection_name=config.COLLECTION_NAME,
                ids=[_FINGERPRINT_ID],
                with_payload=True,
            )
        except Exception:  # noqa: BLE001 - collection may not exist yet
            return None
        if not points:
            return None
        return (points[0].payload or {}).get(_FINGERPRINT_KEY)

    def _ensure_index(self) -> None:
        """Verify the index matches our settings; rebuild it if allowed."""
        exists = self.client.collection_exists(config.COLLECTION_NAME)
        fingerprint = self._stored_fingerprint() if exists else None

        if exists and fingerprint == config.INDEX_FINGERPRINT:
            count = self.client.count(config.COLLECTION_NAME, exact=True).count
            log.info("index up to date (%d points)", count)
            return

        reason = (
            "missing" if not exists
            else f"stale (built as {fingerprint!r}, need {config.INDEX_FINGERPRINT!r})"
        )

        if not config.AUTO_INDEX:
            raise IndexMismatch(
                f"qdrant collection {config.COLLECTION_NAME!r} is {reason}. "
                f"Run `python -m backend.ingest` or set LOREMIND_AUTO_INDEX=1."
            )

        log.warning("index %s — rebuilding", reason)
        self.build_index()

    def build_index(self) -> int:
        """(Re)build the collection from the canonical chunk list."""
        vectors = self._embed_texts([c.text for c in self.chunks])

        self.client.recreate_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )

        points = [
            # id 0 is reserved for the fingerprint sentinel; chunks start at 1.
            PointStruct(id=i + 1, vector=vec.tolist(), payload=chunk.payload)
            for i, (chunk, vec) in enumerate(zip(self.chunks, vectors))
        ]
        points.append(
            PointStruct(
                id=_FINGERPRINT_ID,
                vector=[0.0] * config.EMBED_DIM,
                payload={_FINGERPRINT_KEY: config.INDEX_FINGERPRINT},
            )
        )

        self.client.upsert(collection_name=config.COLLECTION_NAME, points=points)
        log.info("indexed %d chunks with %s", len(self.chunks), config.EMBED_MODEL)
        return len(self.chunks)

    # -------------------------------------------------------- retrieval

    def retrieve(self, query: str) -> tuple[str, list[dict], bool]:
        """Return (context, sources, grounded).

        ``grounded`` is False when the best cross-encoder score falls below the
        threshold, meaning the corpus does not answer this question.
        """
        # The ONNX sessions and BM25 are not documented as thread-safe and this
        # is CPU-bound anyway; serialise to keep memory and latency stable.
        with self._lock:
            dense_rank: dict[int, int] = {}
            bm25_rank: dict[int, int] = {}

            hits = self.client.query_points(
                collection_name=config.COLLECTION_NAME,
                query=self._embed_query(query),
                limit=config.SEMANTIC_TOP_K + 1,  # +1 in case the sentinel ranks
                with_payload=True,
            ).points
            for hit in hits:
                payload = hit.payload or {}
                if _FINGERPRINT_KEY in payload:
                    continue
                idx = payload.get("index")
                if isinstance(idx, int) and 0 <= idx < len(self.chunks):
                    dense_rank.setdefault(idx, len(dense_rank))

            bm25_scores = self._bm25.get_scores(tokenize(query))
            for idx in np.argsort(bm25_scores)[::-1][: config.BM25_TOP_K]:
                if bm25_scores[idx] > 0:
                    bm25_rank.setdefault(int(idx), len(bm25_rank))

            if not dense_rank and not bm25_rank:
                return "", [], False

            # Fuse the two rankings with RRF and rerank only the strongest few.
            # Feeding every candidate to the cross-encoder was ~98% of retrieval
            # latency for no measurable gain in grounding accuracy.
            fused: dict[int, float] = {}
            for idx in set(dense_rank) | set(bm25_rank):
                score = 0.0
                if idx in dense_rank:
                    score += 1.0 / (config.RRF_K + dense_rank[idx])
                if idx in bm25_rank:
                    score += 1.0 / (config.RRF_K + bm25_rank[idx])
                fused[idx] = score

            shortlist = sorted(fused, key=lambda i: fused[i], reverse=True)[
                : config.RERANK_CANDIDATES
            ]
            ordered = [self.chunks[i] for i in shortlist]
            scores = list(self.reranker.rerank(query, [c.text for c in ordered]))

            ranked = sorted(
                zip(ordered, (float(s) for s in scores)),
                key=lambda pair: pair[1],
                reverse=True,
            )[: config.FINAL_TOP_K]

        best = ranked[0][1]
        grounded = best >= config.GROUNDING_MIN_SCORE

        # Drop passages the reranker considers irrelevant so one good hit is not
        # diluted by five bad ones, but keep the bar lower than the grounding
        # gate: supporting detail need not answer the question by itself.
        kept = [(c, s) for c, s in ranked if s >= config.CONTEXT_MIN_SCORE] or ranked[:1]

        context = "\n\n".join(f"- {c.text}" for c, _ in kept)
        sources = [
            {"text": c.text, "section": c.section, "score": round(s, 3)}
            for c, s in kept
        ]

        log.info(
            "query=%r reranked=%d best=%.2f grounded=%s",
            query[:80], len(ordered), best, grounded,
        )
        return context, sources, grounded

    def stats(self) -> dict:
        try:
            count = self.client.count(config.COLLECTION_NAME, exact=True).count
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return {"ok": False, "detail": str(exc)[:200]}
        return {
            "ok": True,
            "points": count,
            "chunks": len(self.chunks),
            "embed_model": config.EMBED_MODEL,
            "rerank_model": config.RERANK_MODEL,
            "runtime": "onnx",
            "mode": "server" if config.QDRANT_URL else "embedded",
        }


# ------------------------------------------------------------------ singleton

_retriever: Retriever | None = None
_init_lock = threading.Lock()


def get_retriever() -> Retriever:
    """Build the retriever on first use.

    Loading the models at import time made the module impossible to import for
    tests or tooling, and turned a transient vector-store outage at boot into a
    crashed worker rather than a degraded one.
    """
    global _retriever
    if _retriever is None:
        with _init_lock:
            if _retriever is None:
                _retriever = Retriever()
    return _retriever
