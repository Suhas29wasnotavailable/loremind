"""Hybrid retrieval: dense + BM25, fused by a cross-encoder, with a grounding gate.

Guarantees this module enforces, each of which was previously violated:

* the model used to embed the query is the model that built the index — checked
  against a fingerprint stored in the collection, so a stale index fails loudly;
* BM25 and the vector store search the *same* chunk list;
* if the reranker says nothing is relevant, we say so instead of handing an
  irrelevant context to the model and letting it improvise.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend import config
from backend.corpus import Chunk, load_chunks, tokenize

log = logging.getLogger("loremind.retrieval")

_FINGERPRINT_ID = 0
_FINGERPRINT_KEY = "__loremind_index_fingerprint__"


class IndexMismatch(RuntimeError):
    """The collection was built with different settings than we are querying with."""


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

        log.info("loading embedding model %s", config.EMBED_MODEL)
        self.embedder = SentenceTransformer(config.EMBED_MODEL)

        # Renamed in sentence-transformers 5.x; support both.
        get_dim = getattr(
            self.embedder, "get_embedding_dimension", None
        ) or self.embedder.get_sentence_embedding_dimension
        actual_dim = get_dim()
        if actual_dim != config.EMBED_DIM:
            raise IndexMismatch(
                f"{config.EMBED_MODEL} produces {actual_dim}-dim vectors but "
                f"LOREMIND_EMBED_DIM is {config.EMBED_DIM}"
            )

        log.info("loading reranker %s", config.RERANK_MODEL)
        self.reranker = CrossEncoder(config.RERANK_MODEL)

        self.client = self._connect()
        self._ensure_index()

    # ------------------------------------------------------------ index

    def _connect(self) -> QdrantClient:
        if config.QDRANT_URL:
            log.info("connecting to qdrant at %s", config.QDRANT_URL)
            return QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY or None,
                timeout=15,
            )
        # Embedded mode: no server, no container, no hosting cost.
        config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        log.info("using embedded qdrant at %s", config.QDRANT_PATH)
        return QdrantClient(path=str(config.QDRANT_PATH))

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
        vectors = self.embedder.encode(
            [c.text for c in self.chunks],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

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

    def _embed_query(self, query: str) -> list[float]:
        # bge-* models expect the retrieval instruction on queries only.
        return self.embedder.encode(
            config.EMBED_QUERY_PREFIX + query,
            normalize_embeddings=True,
        ).tolist()

    def retrieve(self, query: str) -> tuple[str, list[dict], bool]:
        """Return (context, sources, grounded).

        ``grounded`` is False when the best cross-encoder score falls below the
        threshold, meaning the corpus does not answer this question.
        """
        # sentence-transformers and BM25 are not documented as thread-safe and
        # this is CPU-bound anyway; serialise to keep memory and latency stable.
        with self._lock:
            candidates: dict[int, Chunk] = {}

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
                    candidates[idx] = self.chunks[idx]

            bm25_scores = self._bm25.get_scores(tokenize(query))
            for idx in np.argsort(bm25_scores)[::-1][: config.BM25_TOP_K]:
                if bm25_scores[idx] > 0:
                    candidates[int(idx)] = self.chunks[int(idx)]

            if not candidates:
                return "", [], False

            ordered = list(candidates.values())
            scores = self.reranker.predict([[query, c.text] for c in ordered])

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
            "query=%r candidates=%d best=%.2f grounded=%s",
            query[:80], len(candidates), best, grounded,
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
            "mode": "server" if config.QDRANT_URL else "embedded",
        }


# ------------------------------------------------------------------ singleton

_retriever: Retriever | None = None
_init_lock = threading.Lock()


def get_retriever() -> Retriever:
    """Build the retriever on first use.

    Loading ~250MB of models at import time made the module impossible to import
    for tests or tooling, and turned a transient vector-store outage at boot into
    a crashed worker rather than a degraded one.
    """
    global _retriever
    if _retriever is None:
        with _init_lock:
            if _retriever is None:
                _retriever = Retriever()
    return _retriever
