/* ─────────────────────────────────────────────────────────────────────────
   LoreMind — client

   Changes from the original inline script:

   * The endpoint is same-origin. The backend serves this page, so there is no
     hardcoded http://127.0.0.1:8000 to break on deploy and no CORS to satisfy.
     Set window.LOREMIND_API_BASE before this script loads if you are serving
     the frontend from a different origin than the API.

   * Streaming is used for something. Previously every token was accumulated
     and nothing rendered until the stream closed, so the SSE machinery bought
     nothing and the user watched an empty starfield for the whole generation.
     Tokens now feed a live transmission panel, and the crawl begins once the
     text is complete (a crawl scrolls at constant speed, so it genuinely does
     need the full text before it starts).

   * Retrieved sources are displayed instead of being parsed and thrown away.

   * One request at a time, and a new query aborts the one in flight.
   ───────────────────────────────────────────────────────────────────────── */

const API_BASE = (window.LOREMIND_API_BASE ?? "").replace(/\/+$/, "");

const input          = document.getElementById("search-input");
const mainShell      = document.getElementById("main-shell");
const crawlSection   = document.getElementById("crawl-section");
const crawlText      = document.getElementById("crawl-text");
const newQueryBtn    = document.getElementById("new-query-btn");
const transmission   = document.getElementById("transmission");
const streamPanel    = document.getElementById("transmission-stream");
const staticAnswer   = document.getElementById("static-answer");
const sourcesToggle  = document.getElementById("sources-toggle");
const sourcesPanel   = document.getElementById("sources-panel");

// Below this length, crawling is worse than simply showing the text — a
// one-sentence refusal does not deserve a 45-second scroll.
const STATIC_ANSWER_MAX_CHARS = 220;

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let rafId       = null;
let crawlActive = false;
let inFlight    = null;   // AbortController for the current request
let busy        = false;

/* ─────────────────────────────────────────────────────────
   SSE
───────────────────────────────────────────────────────── */

/**
 * Parse an SSE byte stream into {event, data} objects.
 * Each frame is `event: <name>\ndata: <json>\n\n`; data is always JSON, so a
 * token containing newlines can no longer corrupt the framing.
 */
async function* readEvents(response, signal) {
    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = "";

    try {
        while (true) {
            if (signal.aborted) return;

            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            let split;
            while ((split = buffer.indexOf("\n\n")) !== -1) {
                const frame = buffer.slice(0, split);
                buffer = buffer.slice(split + 2);

                let name = "message";
                const dataLines = [];

                for (const line of frame.split("\n")) {
                    if (line.startsWith("event:")) {
                        name = line.slice(6).trim();
                    } else if (line.startsWith("data:")) {
                        dataLines.push(line.slice(5).trimStart());
                    }
                }

                if (!dataLines.length) continue;

                try {
                    yield { event: name, data: JSON.parse(dataLines.join("\n")) };
                } catch {
                    /* Ignore a malformed frame rather than killing the stream. */
                }
            }
        }
    } finally {
        try { reader.cancel(); } catch { /* already closed */ }
    }
}

/**
 * Run a query. Returns {answer, sources, grounded, error}.
 * onToken is called with the running text so the caller can render progress.
 */
async function queryLoreMind(query, signal, onToken) {
    const result = { answer: "", sources: [], grounded: false, error: null };

    let response;
    try {
        response = await fetch(`${API_BASE}/query`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ query }),
            signal,
        });
    } catch (err) {
        if (err.name === "AbortError") throw err;
        result.error = "The archives are unreachable. Is the server running?";
        return result;
    }

    if (!response.ok) {
        // 422 means the request itself was rejected (empty or over-long query).
        result.error = response.status === 422
            ? "That query cannot be transmitted. Try something shorter."
            : `The archives responded with an error (${response.status}).`;
        return result;
    }

    for await (const { event, data } of readEvents(response, signal)) {
        if (event === "sources") {
            result.sources = Array.isArray(data) ? data : [];
        } else if (event === "token") {
            result.answer += data;
            onToken?.(result.answer);
        } else if (event === "error") {
            result.error = String(data);
        } else if (event === "done") {
            result.grounded = Boolean(data?.grounded);
        }
    }

    return result;
}

/* ─────────────────────────────────────────────────────────
   CRAWL ENGINE  (unchanged geometry — see notes below)

   translateY in vh is distorted by the perspective matrix, so the animation
   runs in px. perspective-origin sits at 18% from the top, which means points
   near the bottom of the container are barely scaled, so px ≈ screen px there.
   Start at 2× viewport height for margin; end above the text's own height.
───────────────────────────────────────────────────────── */

function startCrawl(durationSeconds) {
    crawlActive = true;

    const H          = window.innerHeight;
    const startPx    = H * 2;
    const endPx      = -(crawlText.scrollHeight + H);
    const totalPx    = startPx - endPx;
    const pxPerFrame = totalPx / (durationSeconds * 60);

    let currentPx = startPx;
    crawlText.style.transform = `rotateX(20deg) translateY(${currentPx}px)`;

    function tick() {
        if (!crawlActive) return;

        currentPx -= pxPerFrame;
        crawlText.style.transform = `rotateX(20deg) translateY(${currentPx}px)`;

        if (currentPx > endPx) {
            rafId = requestAnimationFrame(tick);
        } else {
            rafId = null;
        }
    }

    rafId = requestAnimationFrame(tick);
}

function stopCrawl() {
    crawlActive = false;
    if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
    }
}

/* ─────────────────────────────────────────────────────────
   SOURCES
───────────────────────────────────────────────────────── */

function renderSources(sources) {
    sourcesPanel.replaceChildren();

    if (!sources.length) {
        sourcesToggle.style.display = "none";
        return;
    }

    for (const source of sources) {
        const card = document.createElement("div");
        card.className = "source-card";

        const meta = document.createElement("span");
        meta.className = "source-meta";
        meta.textContent = `${source.section ?? "archive"} · relevance ${source.score ?? "?"}`;

        const body = document.createElement("p");
        // textContent, not innerHTML: corpus text must never be parsed as markup.
        body.textContent = (source.text ?? "").replace(/^\[[^\]]*]\s*/, "");

        card.append(meta, body);
        sourcesPanel.appendChild(card);
    }

    sourcesToggle.style.display = "block";
}

sourcesToggle.addEventListener("click", () => {
    const open = sourcesPanel.style.display === "block";
    sourcesPanel.style.display = open ? "none" : "block";
    sourcesToggle.textContent  = open ? "Sources" : "Hide";
});

/* ─────────────────────────────────────────────────────────
   VIEW HELPERS
───────────────────────────────────────────────────────── */

function showTransmission() {
    streamPanel.textContent = "";
    transmission.style.display = "flex";
}

function hideTransmission() {
    transmission.style.display = "none";
    streamPanel.textContent = "";
}

function resetOutputs() {
    stopCrawl();
    hideTransmission();

    crawlSection.style.display = "none";
    crawlSection.style.opacity = "1";
    crawlText.textContent = "";

    staticAnswer.style.display = "none";
    staticAnswer.textContent = "";

    sourcesPanel.style.display = "none";
    sourcesPanel.replaceChildren();
    sourcesToggle.style.display = "none";
    sourcesToggle.textContent = "Sources";

    newQueryBtn.style.display = "none";
}

async function presentAsCrawl(text) {
    crawlText.textContent = text;

    // Park the text below the viewport before the section becomes visible.
    crawlText.style.transform = `rotateX(20deg) translateY(${window.innerHeight * 2}px)`;
    crawlSection.style.display = "block";
    crawlSection.style.opacity = "0";

    // Three frames guarantees the browser has committed and painted the
    // off-screen transform before anything becomes visible.
    await new Promise((resolve) =>
        requestAnimationFrame(() =>
            requestAnimationFrame(() => requestAnimationFrame(resolve))
        )
    );

    crawlSection.style.opacity = "1";

    // ~14 chars/second reading pace, floored so short answers are not glacial
    // and capped so long ones stay bearable.
    const duration = Math.min(Math.max(20, 10 + text.length * 0.055), 110);
    startCrawl(duration);
}

function presentAsStatic(text) {
    staticAnswer.textContent = text;
    staticAnswer.style.display = "flex";
}

/* ─────────────────────────────────────────────────────────
   SEARCH
───────────────────────────────────────────────────────── */

async function submitQuery(query) {
    if (busy) return;
    busy = true;

    // Abort anything still running from a previous query.
    inFlight?.abort();
    inFlight = new AbortController();
    const { signal } = inFlight;

    resetOutputs();
    mainShell.classList.add("shell-hidden");

    // Let the shell fade, then show the loading state immediately — the whole
    // point is that the screen is never blank while we wait.
    await wait(400);
    if (signal.aborted) { busy = false; return; }
    showTransmission();

    let result;
    try {
        result = await queryLoreMind(query, signal, (text) => {
            // Show the tail of the incoming text so progress is visible.
            streamPanel.textContent = text.slice(-260);
        });
    } catch (err) {
        if (err.name === "AbortError") { busy = false; return; }
        result = { answer: "", sources: [], grounded: false, error: String(err) };
    }

    if (signal.aborted) { busy = false; return; }

    hideTransmission();

    const text = result.error
        || result.answer.trim()
        || "The archives are silent. No transmission received.";

    renderSources(result.sources);

    if (text.length <= STATIC_ANSWER_MAX_CHARS) {
        presentAsStatic(text);
        newQueryBtn.style.display = "block";
    } else {
        await presentAsCrawl(text);
        await wait(2500);
        newQueryBtn.style.display = "block";
    }

    busy = false;
}

input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;

    const query = input.value.trim();
    if (!query) return;

    submitQuery(query);
});

/* ─────────────────────────────────────────────────────────
   RESET
───────────────────────────────────────────────────────── */

newQueryBtn.addEventListener("click", () => {
    inFlight?.abort();
    resetOutputs();

    crawlText.style.transform = `rotateX(20deg) translateY(${window.innerHeight * 2}px)`;
    mainShell.classList.remove("shell-hidden");

    input.value = "";
    setTimeout(() => input.focus(), 600);
});

// Escape returns to the search shell from anywhere.
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && mainShell.classList.contains("shell-hidden")) {
        newQueryBtn.click();
    }
});

input.focus();
