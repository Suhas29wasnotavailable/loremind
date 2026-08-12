"""Export (query, chunk) pairs and torch reference scores for ONNX comparison.

Run from the project root with the main venv. Writes /tmp/backend_cmp.json,
which /tmp/score_onnx.py then scores using fastembed so the two can be compared.
"""
import json
import logging

from backend.logging_setup import setup_logging

setup_logging(logging.WARNING)

from backend import config
from backend.retrieval import get_retriever

QUERIES = [
    "Who is Darth Vader?",
    "What was Order 66?",
    "How did Yoda die?",
    "Tell me about the Death Star",
    "What is the Rule of Two?",
    "What is the capital of France?",
    "Explain how to write a Python decorator",
    "Who won the 2019 Super Bowl?",
]

r = get_retriever()
out = {"embed_model": config.EMBED_MODEL, "cases": []}

for q in QUERIES:
    # Use the real pipeline so we compare on the passages actually retrieved.
    _, sources, grounded = r.retrieve(q)
    out["cases"].append({
        "query": q,
        "grounded_torch": grounded,
        "top_score_torch": sources[0]["score"] if sources else None,
        "passages": [s["text"] for s in sources],
        "scores_torch": [s["score"] for s in sources],
    })

# Also capture query embeddings so vector agreement can be measured.
out["query_vectors_torch"] = {
    q: r.embedder.encode(config.EMBED_QUERY_PREFIX + q, normalize_embeddings=True).tolist()
    for q in QUERIES
}

json.dump(out, open("/tmp/backend_cmp.json", "w"))
print(f"wrote {len(out['cases'])} cases with torch reference scores")
for c in out["cases"]:
    print(f"  {c['top_score_torch']:7.3f}  grounded={c['grounded_torch']!s:5}  {c['query']}")
r.client.close()
