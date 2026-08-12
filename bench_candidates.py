"""How few candidates can we rerank before the grounding gate degrades?

The cross-encoder is ~98% of retrieval cost and scales with candidate count, so
this is the only lever that meaningfully cuts latency on a CPU-throttled host.
Candidates are pre-ranked by Reciprocal Rank Fusion of the dense and BM25
rankings, which is cheap, so the cross-encoder only sees the most promising ones.
"""
import logging
import time

from backend.logging_setup import setup_logging

setup_logging(logging.WARNING)

import numpy as np

from backend import config
from backend.corpus import tokenize
from backend.retrieval import get_retriever

ON = [
    "Who is Darth Vader?", "What was Order 66?", "How did Yoda die?",
    "Tell me about the Death Star", "What is the Rule of Two?",
    "Who was Ahsoka Tano?", "What is a Force dyad?", "Who built C-3PO?",
    "What is Exegol?", "Who is Din Djarin?", "What was the Inquisitorius?",
    "Why did Luke go into exile?", "How did Anakin fall to the dark side?",
    "What happened at the Battle of Endor?", "Who was Jango Fett?",
    "What is the Darksaber?", "Tell me about Palpatine's rise to power",
    "How did Obi-Wan defeat Anakin?", "What are midi-chlorians?",
    "What happened to the Jedi Temple?",
]
OFF = [
    "What is the capital of France?", "Explain how to write a Python decorator",
    "Who won the 2019 Super Bowl?", "Give me a recipe for carbonara",
    "What is the population of Tokyo?", "Summarise the plot of The Great Gatsby",
    "How do I reset my email password?", "What did Thrawn collect as a hobby?",
    "Who was Grand Admiral Thrawn's flagship commander?",
    "What is the airspeed velocity of an unladen swallow?",
]

r = get_retriever()
K_RRF = 60  # standard RRF damping


def fused_candidates(query, limit):
    """Rank by RRF over dense and BM25, return the top `limit` chunk texts."""
    dense_rank, bm25_rank = {}, {}

    vec = r._embed_query(query)
    hits = r.client.query_points(
        collection_name=config.COLLECTION_NAME, query=vec,
        limit=config.SEMANTIC_TOP_K + 1, with_payload=True,
    ).points
    rank = 0
    for h in hits:
        p = h.payload or {}
        if isinstance(p.get("index"), int):
            dense_rank[p["index"]] = rank
            rank += 1

    scores = r._bm25.get_scores(tokenize(query))
    order = [int(i) for i in np.argsort(scores)[::-1] if scores[int(i)] > 0]
    for rank, idx in enumerate(order[: config.BM25_TOP_K]):
        bm25_rank[idx] = rank

    fused = {}
    for idx in set(dense_rank) | set(bm25_rank):
        score = 0.0
        if idx in dense_rank:
            score += 1.0 / (K_RRF + dense_rank[idx])
        if idx in bm25_rank:
            score += 1.0 / (K_RRF + bm25_rank[idx])
        fused[idx] = score

    best = sorted(fused, key=lambda i: fused[i], reverse=True)[:limit]
    return [r.chunks[i].text for i in best]


print(f"{'cands':>6} {'rerank ms':>10} {'on min':>8} {'off max':>8} {'gap':>7}  {'threshold':>9}  verdict")
for limit in (4, 6, 8, 10, 12, 16, 25):
    sets = {q: fused_candidates(q, limit) for q in ON + OFF}
    list(r.reranker.rerank(ON[0], sets[ON[0]]))  # warm

    tops = {}
    t0 = time.perf_counter()
    for q in ON + OFF:
        tops[q] = max(r.reranker.rerank(q, sets[q]))
    ms = (time.perf_counter() - t0) / len(tops) * 1000

    on_s = [tops[q] for q in ON]
    off_s = [tops[q] for q in OFF]
    gap = min(on_s) - max(off_s)
    mid = (min(on_s) + max(off_s)) / 2
    verdict = "clean" if gap > 2 else ("tight" if gap > 0 else "OVERLAP")
    print(f"{limit:>6} {ms:>10.1f} {min(on_s):>8.2f} {max(off_s):>8.2f} {gap:>7.2f}  {mid:>9.2f}  {verdict}")

r.client.close()
