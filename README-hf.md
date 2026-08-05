---
title: LoreMind
emoji: 🌌
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: A Star Wars lore engine with grounded hybrid retrieval
---

# LoreMind

Hybrid retrieval (dense + BM25, cross-encoder reranked) over a curated Star Wars
corpus, narrated as a galactic archive and rendered as an opening crawl.

Retrieval runs on CPU in this Space. Generation is delegated to Groq's free tier —
set `GROQ_API_KEY` as a Space secret under **Settings → Variables and secrets**.

See the repository README for design notes.
