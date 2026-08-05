"""Corpus loading and chunking — the single implementation.

The lore file is a sequence of `# SECTION` banners followed by one self-contained
fact per line. Two things about the original chunking were wrong:

1. Splitting on blank lines made every ASCII banner its own chunk. Half the
   vector index was `# ===== # THE FORCE # =====`, which carries no information
   but embeds strongly against topical queries, so the noise outranked the facts.

2. Ingestion split on blank lines (51 chunks averaging 1.4k chars) while the
   query path split on newlines (816 one-line docs). BM25 and the vector store
   were therefore searching different corpora, and the cross-encoder was asked
   to rank single sentences against 5kB blobs.

Here the banners become the section label carried on each chunk, and both
retrieval paths consume the identical chunk list.
"""

from dataclasses import dataclass
from pathlib import Path

from backend import config


@dataclass(frozen=True)
class Chunk:
    text: str      # what gets embedded and shown as a source
    section: str   # e.g. "YODA"
    index: int

    @property
    def payload(self) -> dict:
        return {"text": self.text, "section": self.section, "index": self.index}


def _sections(raw: str) -> list[tuple[str, list[str]]]:
    """Split the file into (section_title, fact_lines) pairs, dropping banners."""
    sections: list[tuple[str, list[str]]] = []
    title = "GENERAL"
    lines: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            # A banner line. Ignore pure rules (`# =====`) and treat anything
            # with actual words as the new section title.
            label = stripped.lstrip("#").strip().strip("=").strip()
            if label:
                if lines:
                    sections.append((title, lines))
                    lines = []
                title = label
            continue

        lines.append(stripped)

    if lines:
        sections.append((title, lines))
    return sections


def load_chunks(
    path: Path | None = None,
    lines_per_chunk: int | None = None,
    stride: int | None = None,
) -> list[Chunk]:
    """Build the canonical chunk list. Deterministic: same file, same chunks."""
    path = path or config.LORE_PATH
    lines_per_chunk = lines_per_chunk or config.CHUNK_LINES
    stride = stride or config.CHUNK_STRIDE

    raw = Path(path).read_text(encoding="utf-8")

    chunks: list[Chunk] = []
    for title, facts in _sections(raw):
        # Sliding window with overlap, so a fact is never severed from the
        # sentence that gives it context.
        start = 0
        while start < len(facts):
            window = facts[start : start + lines_per_chunk]
            if not window:
                break
            body = " ".join(window)
            # Prefixing the section makes each chunk self-describing, which
            # matters because facts often omit the subject ("The apprentice was
            # expected to kill the master" only means something under THE SITH).
            chunks.append(
                Chunk(text=f"[{title}] {body}", section=title, index=len(chunks))
            )
            if start + lines_per_chunk >= len(facts):
                break
            start += stride

    return chunks


def tokenize(text: str) -> list[str]:
    """Tokenizer shared by BM25 indexing and BM25 querying."""
    return text.lower().split()
