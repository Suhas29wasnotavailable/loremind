"""Logging configuration shared by the API and the CLI entrypoints."""

import logging

# These libraries log every HTTP request at INFO, which buries our own output
# behind dozens of huggingface.co cache probes on every start.
_NOISY = ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "sentence_transformers")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
