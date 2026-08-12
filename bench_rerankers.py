"""Compare reranker candidates on speed and on grounded/ungrounded separation.

Render's free plan throttles CPU hard enough that the cross-encoder dominates
request latency (measured 98% of retrieval). A lighter reranker is only usable if
it still separates answerable questions from unanswerable ones, because the
grounding gate depends on that separation.
"""
import logging
import time

from backend.logging_setup import setup_logging

setup_logging(logging.WARNING)

from fastembed.rerank.cross_encoder import TextCrossEncoder

from backend import config
from backend.corpus import tokenize
from backend.retrieval import get_retriever

import numpy as np

ON = [
    "Who is Darth Vader?", "What was Order 66?", "How did Yoda die?",
    "Tell me about the Death Star", "What is the Rule of Two?",
    "Who was Ahsoka Tano?", "What is a Force dyad?", "Who built C-3PO?",
    "What is Exegol?", "Who is Din Djarin?", "What was the Inquisitorius?",
    "Why did Luke go into exile?",
]
OFF = [
    "What is the capital of France?", "Explain how to write a Python decorator",
    "Who won the 2019 Super Bowl?", "Give me a recipe for carbonara",
    "What is the population of Tokyo?", "Summarise the plot of The Great Gatsby",
    "How do I reset my email password?", "What did Thrawn collect as a hobby?",
]

MODELS = [
    "Xenova/ms-marco-MiniLM-L-6-v2",     # current
    "jinaai/jina-reranker-v1-tiny-en",
    "jinaai/jina-reranker-v1-turbo-en",
]

r = get_retriever()


def candidates_for(query):
    """Reproduce the real candidate set the pipeline would rerank."""
    out = {}
    vec = r._embed_query(query)
    for h in r.client.query_points(
        collection_name=config.COLLECTION_NAME, query=vec,
        limit=config.SEMANTIC_TOP_K + 1, with_payload=True,
    ).points:
        p = h.payload or {}
        if isinstance(p.get("index"), int):
            out[p["index"]] = r.chunks[p["index"]]
    bs = r._bm25.get_scores(tokenize(query))
    for i in np.argsort(bs)[::-1][: config.BM25_TOP_K]:
        if bs[i] > 0:
            out[int(i)] = r.chunks[int(i)]
    return [c.text for c in out.values()]

cand_cache = {q: candidates_for(q) for q in ON + OFF}
sizes = [len(v) for v in cand_cache.values()]
print(f"candidate sets: {min(sizes)}-{max(sizes)} passages (mean {sum(sizes)/len(sizes):.1f})\n")

for name in MODELS:
    try:
        rr = TextCrossEncoder(name, cache_dir=str(config.MODEL_CACHE_DIR))
    except Exception as exc:
        print(f"{name}: unavailable — {str(exc)[:80]}\n")
        continue

    list(rr.rerank(ON[0], cand_cache[ON[0]]))  # warm

    tops = {}
    t0 = time.perf_counter()
    for q in ON + OFF:
        tops[q] = max(rr.rerank(q, cand_cache[q]))
    elapsed = (time.perf_counter() - t0) / len(tops) * 1000

    on_s = [tops[q] for q in ON]
    off_s = [tops[q] for q in OFF]
    gap = min(on_s) - max(off_s)

    print(f"{name}")
    print(f"  mean rerank time : {elapsed:7.1f} ms/query")
    print(f"  on-corpus  range : {min(on_s):8.3f} .. {max(on_s):8.3f}")
    print(f"  off-corpus range : {min(off_s):8.3f} .. {max(off_s):8.3f}")
    print(f"  separation gap   : {gap:8.3f}  {'CLEAN' if gap > 0 else '*** OVERLAP ***'}")
    if gap > 0:
        print(f"  suggested threshold (midpoint): {(min(on_s)+max(off_s))/2:.3f}")
    print()

r.client.close()
