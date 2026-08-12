"""A stand-in for Groq's chat-completions endpoint.

Exercises backend/llm.py's Groq path without a real key, so the provider is
known-good before a deploy depends on it. Reproduces the parts of the wire
format that the client actually parses, including the quirks:

* an opening chunk whose delta carries only `role`, no `content`
* `finish_reason` arriving on a chunk with an empty delta
* a `data: [DONE]` sentinel rather than a closed stream
* keep-alive blank lines between frames

Run:  python mock_groq.py            # serves on 8111
"""

import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

REPLY = (
    "Darth Vader was once Anakin Skywalker, a Jedi Knight of rare promise.\n\n"
    "His fall on Mustafar left him encased in black armour, bound to the "
    "Emperor's will for a generation."
)


@app.post("/v1/chat/completions")
async def completions(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or len(auth) < 12:
        # Matches Groq's shape for a bad key.
        return JSONResponse(
            {"error": {"message": "Invalid API Key", "type": "invalid_request_error"}},
            status_code=401,
        )

    body = await request.json()
    if body.get("stream") is not True:
        return JSONResponse({"error": {"message": "test expects stream=true"}}, status_code=400)

    def frame(delta: dict, finish=None) -> str:
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("model", "mock"),
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    async def stream():
        # Opening chunk: role only, no content. The client must not emit "".
        yield frame({"role": "assistant"})
        yield ": keep-alive\n\n"          # SSE comment line, must be ignored
        for word in REPLY.split(" "):
            yield frame({"content": word + " "})
        yield frame({}, finish="stop")     # empty delta with finish_reason
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8111, log_level="warning")
