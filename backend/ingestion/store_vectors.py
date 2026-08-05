"""Deprecated — superseded by `python -m backend.ingest`.

This script used to declare its own embedding model and its own chunking rule,
which is how the index came to be built with BAAI/bge-small-en-v1.5 while the
query path embedded with all-MiniLM-L6-v2. Both are 384-dimensional, so Qdrant
accepted the mismatched vectors and silently returned nonsense.

It now delegates, so there is exactly one place that decides how the index is
built.
"""

from backend.ingest import main

if __name__ == "__main__":
    print(__doc__)
    main()
