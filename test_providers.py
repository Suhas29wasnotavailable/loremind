"""Test the Groq generation path without a real API key.

Starts mock_groq.py in a background thread and exercises backend/llm.py against
it. Worth having permanently: the deployed configuration uses Groq, so a bug
here would only surface in production.

    python test_providers.py
"""

import asyncio
import os
import threading
import time
import urllib.request

PORT = 8112
os.environ.update(
    LOREMIND_LLM_PROVIDER="groq",
    LOREMIND_GROQ_BASE_URL=f"http://127.0.0.1:{PORT}/v1",
    GROQ_API_KEY="gsk_faketestkey123456",
)

import uvicorn  # noqa: E402

from mock_groq import app as mock_app  # noqa: E402

passed, failed = [], []


def check(name, condition, detail=""):
    (passed if condition else failed).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


server = uvicorn.Server(uvicorn.Config(mock_app, host="127.0.0.1", port=PORT, log_level="error"))
threading.Thread(target=server.run, daemon=True).start()

for _ in range(100):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/docs", timeout=1)
        break
    except Exception:
        time.sleep(0.1)


async def collect(**env):
    """Import llm fresh so config picks up the patched environment."""
    import importlib

    os.environ.update({k: v for k, v in env.items()})
    from backend import config, llm

    importlib.reload(config)
    importlib.reload(llm)

    tokens = []
    error = None
    try:
        async for token in llm.stream_tokens("test"):
            tokens.append(token)
    except llm.LLMError as exc:
        error = str(exc)
    return tokens, error


async def main():
    print("\n=== groq streaming (mock) ===")
    tokens, error = await collect(GROQ_API_KEY="gsk_faketestkey123456")
    text = "".join(tokens)
    check("streams without error", error is None, error or "")
    check("emits multiple tokens", len(tokens) > 10, f"{len(tokens)} tokens")
    check("skips the role-only opening chunk", all(t for t in tokens),
          "no empty tokens")
    check("preserves newlines in content", "\n" in text)
    check("reassembles cleanly", text.startswith("Darth Vader was once Anakin"),
          text[:52])

    print("\n=== groq error handling ===")
    _, error = await collect(GROQ_API_KEY="bad")
    check("rejects an invalid key", error is not None and "401" in error,
          (error or "")[:70])

    _, error = await collect(GROQ_API_KEY="")
    check("refuses when the key is missing", error is not None and "GROQ_API_KEY" in error,
          error or "")

    os.environ["LOREMIND_GROQ_BASE_URL"] = "http://127.0.0.1:9/v1"
    _, error = await collect(GROQ_API_KEY="gsk_faketestkey123456")
    check("surfaces an unreachable endpoint", error is not None and "cannot reach" in error,
          (error or "")[:70])

    print("\n" + "=" * 58)
    print(f"  {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("  failures: " + ", ".join(failed))
    print("=" * 58)
    return 1 if failed else 0


raise SystemExit(asyncio.run(main()))
