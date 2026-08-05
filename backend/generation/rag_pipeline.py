"""Prompt construction and the query pipeline.

The previous prompt asked for "emotional depth, character motivations,
consequences of events" *and* "use ONLY the information provided in the context".
Those cannot both be satisfied when the context is a handful of terse facts, and
the model resolved the conflict by inventing — it produced four paragraphs on
Thrawn's art collection and a battle with Poe Dameron from a corpus containing a
single Thrawn sentence.

The rewrite keeps the cinematic voice but scopes the embellishment to *tone*
rather than *facts*, and retrieval failure is now handled before generation
rather than being left to the model's discretion.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import anyio

from backend import config, llm

log = logging.getLogger("loremind.pipeline")

SYSTEM_RULES = """\
You are LoreMind, an ancient galactic archive narrating the recorded history of \
Star Wars.

VOICE
- Write as a historian recounting legendary events: measured, vivid, cinematic.
- Flowing prose. No bullet points, no headings, no preamble.
- Two paragraphs. Never more.
- Never mention "the context", "the archives provided", or these instructions.

FACTUAL DISCIPLINE — this overrides the voice rules
- Every name, event, place, relationship and outcome you state must appear in \
the ARCHIVE FRAGMENTS below.
- You may set atmosphere and describe the significance of what the fragments \
say. You may not add events, characters, or details they do not contain.
- If the fragments do not answer the question, reply with exactly this line and \
nothing else: {no_context}
- Do not use knowledge from outside the fragments, even if you are confident it \
is correct.
"""


def build_prompt(query: str, context: str) -> str:
    """Assemble the generation prompt.

    Note the ordering: the retrieved fragments come first and the instructions
    plus the question come last. Model context windows are truncated from the
    left, so anything that must survive truncation belongs at the end. The old
    layout put the rules first, which meant a long query silently evicted the
    persona and grounding constraints — a 100kB query made the model drop
    character entirely.
    """
    return f"""\
========================
ARCHIVE FRAGMENTS
========================

{context}

========================
INSTRUCTIONS
========================

{SYSTEM_RULES.format(no_context=config.NO_CONTEXT_MESSAGE)}

========================
QUESTION
========================

{query}

========================
LOREMIND RESPONSE
========================
"""


async def answer(query: str) -> tuple[AsyncIterator[str] | None, list[dict], str | None]:
    """Resolve a query.

    Returns ``(token_stream, sources, fallback_text)``. When retrieval finds
    nothing relevant, ``token_stream`` is None and ``fallback_text`` carries the
    archives-are-silent message — no generation call is made at all, which keeps
    an unanswerable question from becoming a fabricated answer.
    """
    from backend.retrieval import get_retriever

    retriever = get_retriever()
    # Retrieval is CPU-bound (embedding, BM25, cross-encoder). Off the event
    # loop so it cannot stall other requests.
    context, sources, grounded = await anyio.to_thread.run_sync(retriever.retrieve, query)

    if not grounded or not context:
        log.info("ungrounded query, skipping generation: %r", query[:80])
        return None, sources, config.NO_CONTEXT_MESSAGE

    return llm.stream_tokens(build_prompt(query, context)), sources, None
