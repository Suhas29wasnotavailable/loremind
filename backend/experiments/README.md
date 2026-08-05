# Experiments

Prototypes kept for reference. **None of these are used by the running app** —
they were the exploratory steps that led to `backend/retrieval.py`.

| file | superseded by | why |
|---|---|---|
| `proto_chunking.py` | `backend/corpus.py` | split on blank lines, which made every ASCII section banner its own chunk |
| `proto_semantic_search.py` | `backend/retrieval.py` | dense-only retrieval |
| `proto_hybrid_search.py` | `backend/retrieval.py` | fixed 0.7/0.3 score blend instead of a cross-encoder |
| `proto_reranked_search.py` | `backend/retrieval.py` | closest ancestor of the current pipeline |
| `test_embeddings.py` | — | cosine-similarity sanity check; still useful standalone |

Each declares its own model name and reads `data/starwars_lore.txt` by relative
path, so run them from `backend/`. Note that hardcoding the model is exactly how
the index came to be built with one embedding model and queried with another —
the shipped code reads it from `backend/config.py` for that reason.
