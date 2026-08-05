"""Build the vector index.

    python -m backend.ingest            # build only if missing or stale
    python -m backend.ingest --force    # rebuild unconditionally

Uses the same chunker and the same embedding model the query path uses, because
both read them from backend.config. Normally unnecessary — the API rebuilds the
index automatically when the fingerprint does not match — but useful for
pre-baking an index into a container image or targeting a remote Qdrant.
"""

import argparse

from backend import config
from backend.logging_setup import setup_logging
from backend.retrieval import get_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the LoreMind vector index.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the existing index matches the current settings",
    )
    args = parser.parse_args()

    setup_logging()

    # Constructing the retriever already builds the index when it is missing or
    # stale, so only --force needs an explicit second pass.
    retriever = get_retriever()
    count = retriever.build_index() if args.force else len(retriever.chunks)

    print()
    print(f"  chunks indexed : {count}")
    print(f"  embed model    : {config.EMBED_MODEL}")
    print(f"  rerank model   : {config.RERANK_MODEL}")
    print(f"  collection     : {config.COLLECTION_NAME}")
    print(f"  store          : {config.QDRANT_URL or f'embedded {config.QDRANT_PATH}'}")
    print(f"  fingerprint    : {config.INDEX_FINGERPRINT}")
    print()

    # Embedded Qdrant holds a file lock; closing it explicitly avoids a noisy
    # ImportError from __del__ running during interpreter teardown.
    retriever.client.close()


if __name__ == "__main__":
    main()
