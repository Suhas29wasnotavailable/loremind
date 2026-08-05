# LoreMind

A Star Wars lore engine: hybrid retrieval (dense + BM25, cross-encoder reranked)
over a curated corpus, answered in the voice of a galactic archive and rendered
as an opening crawl.

```
frontend/            index.html + js/app.js — no build step
backend/
  config.py          all settings; the single source of truth
  corpus.py          the one chunking implementation
  retrieval.py       embed + BM25 + rerank + grounding gate
  llm.py             generation providers (ollama | groq)
  ingest.py          python -m backend.ingest
  generation/        prompt construction
  api/main.py        FastAPI; also serves the frontend
  data/              starwars_lore.txt
calibrate.py         choose the grounding threshold from data
regression_test.py   asserts the audited failures stay fixed
```

## Run it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# generation on this machine (needs `ollama pull llama3`)
uvicorn backend.api.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. The API serves the frontend, so there is no second
server and no CORS to configure.

On first start the app downloads two small models (~150MB) and builds the vector
index. Both are cached afterwards.

To run generation off-device instead — recommended on a laptop, since an 8B model
will peg the GPU — put a free [Groq](https://console.groq.com/keys) key in `.env`:

```bash
cp .env.example .env
# LOREMIND_LLM_PROVIDER=groq
# GROQ_API_KEY=gsk_...
```

Check everything is wired up:

```bash
curl -s localhost:8000/health | python -m json.tool
```

## Deploy

The container serves API + frontend on one port and needs no GPU, no Qdrant
server and no Ollama. Models and the vector index are baked into the image, so a
cold start does not download anything.

```bash
docker build -t loremind .
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... loremind
# -> http://localhost:7860
```

### Hugging Face Spaces (free CPU tier)

The free tier gives 2 vCPU and 16GB RAM, which comfortably fits the embedding
model and reranker. Generation is delegated to Groq, so no GPU is needed.

1. Get a free Groq key: <https://console.groq.com/keys>
2. Create a **Docker** Space: <https://huggingface.co/new-space>
3. In the Space, **Settings → Variables and secrets**, add secret `GROQ_API_KEY`
4. Push:

```bash
hf auth login                                   # needs a write token
git remote add space https://huggingface.co/spaces/<user>/loremind
cp README-hf.md README-space.md                 # Spaces needs the front-matter
git push space main
```

The Space's root `README.md` must carry the YAML front-matter from
`README-hf.md` (`sdk: docker`, `app_port: 7860`) or the build is ignored.

### Before exposing it publicly

| Setting | Default | Change to |
|---|---|---|
| `LOREMIND_CORS_ORIGINS` | `*` | your own origin |
| `LOREMIND_TRUST_PROXY_HEADERS` | `0` | `1` behind Spaces/Fly/Render, so rate limiting sees the real client IP rather than the proxy's |
| `LOREMIND_RATE_LIMIT_PER_MINUTE` | `12` | lower if the Groq free quota is a concern |

## Design notes

**One place decides the embedding model.** Ingestion and querying previously
named different models. Both were 384-dimensional, so Qdrant accepted the
mismatched vectors without error and returned effectively random neighbours —
"Tell me about the Death Star" retrieved Han Solo, Yoda and Ahsoka. Both paths
now read `config.EMBED_MODEL`, and the index stores a fingerprint of the model
and chunking settings that is verified at startup, so a stale index fails loudly
instead of degrading silently.

**Chunking keeps section context and drops the banners.** The corpus is
`# SECTION` headers over one-fact-per-line prose. Splitting on blank lines made
every ASCII banner its own chunk — half the index was `# ===== # THE FORCE
# =====`, which embeds strongly against topical queries while carrying no
information. Chunks are now overlapping 6-line windows prefixed with their
section, and BM25 and the vector store index the identical list.

**Retrieval failure is handled before generation.** The cross-encoder already
scores relevance; nothing used to read it. Calibrated over 22 answerable and 10
unanswerable questions, answerable queries score 2.5–9.7 and unanswerable ones
−6.7 to −11.2, so the gate sits at the midpoint (−2.0). Below it, no generation
call is made at all. Previously the model received irrelevant context and
invented — asked about Thrawn, whose entire corpus presence is one sentence, it
produced paragraphs on Chiss civilisation and a battle with Poe Dameron. Re-run
`calibrate.py` after changing the corpus or either model.

**The prompt separates voice from fact.** It used to ask for "emotional depth,
character motivations, consequences of events" *and* "use ONLY the information
provided". Those conflict when the context is a handful of terse facts, and the
model resolved the conflict by inventing. Embellishment is now explicitly scoped
to tone. The instructions and the question also sit *after* the fragments,
because context windows truncate from the left — with the old ordering a long
query silently evicted the persona and the grounding rules.

**Streaming does something.** Tokens were accumulated and nothing rendered until
the stream closed, so the user watched an empty starfield for the whole
generation. Tokens now drive a live transmission panel; the crawl starts when the
text is complete, because a constant-speed crawl genuinely needs the full text.
Short answers are shown statically rather than crawled for 45 seconds.

**Sources are shown.** They were computed, streamed and discarded, which made
answers impossible for a reader to check.

## Tests

```bash
python regression_test.py   # needs the server running
python calibrate.py         # threshold separation report
```
