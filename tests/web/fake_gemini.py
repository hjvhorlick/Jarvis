"""A local server that speaks the real Gemini REST protocol.

Used to exercise the genuine google-genai SDK end to end without reaching
Google: the SDK serialises the request, makes a real HTTP call, and parses a
real SSE response — only the model behind it is fake.

Run standalone:  python tests/web/fake_gemini.py --port 8099
"""

from __future__ import annotations

import json
import time
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="fake-gemini")

# Every request the SDK makes, recorded so tests can assert on it. Each record
# is normalized to {model, contents, system, api_key} regardless of whether the
# caller used the camelCase or snake_case system-instruction field.
REQUESTS: list[dict] = []


def _record(model: str, body: dict, headers: dict) -> dict:
    system = (
        body.get("systemInstruction") or body.get("system_instruction") or {}
    )
    entry = {
        "model": model,
        "contents": body.get("contents", []),
        "system": "".join(p.get("text", "") for p in system.get("parts", [])),
        "api_key": headers.get("x-goog-api-key", ""),
        "body": body,
    }
    REQUESTS.append(entry)
    return entry

REPLIES = {
    "default": "Hello! I am a fake Gemini endpoint used for local end-to-end tests.",
    "greet": "Hi there. This reply came from the fake endpoint over a real HTTP call.",
}


def _reply_for(prompt: str) -> str:
    lowered = prompt.lower()
    if "greet" in lowered or "hello" in lowered or "hi" in lowered:
        return REPLIES["greet"]
    return REPLIES["default"]


def _check_auth(request: Request) -> str | None:
    """Reject the way Google does when the key is missing or wrong."""
    key = request.headers.get("x-goog-api-key", "")
    if not key:
        return "API key not specified"
    if key == "AQ.bad-key-for-tests":
        return "API key not valid. Please pass a valid API key."
    return None


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": status, "message": message, "status": code}},
        status_code=status,
    )


def _extract_prompt(body: dict) -> str:
    contents = body.get("contents") or []
    if not contents:
        return ""
    parts = (contents[-1].get("parts") or [{}])
    return "".join(p.get("text", "") for p in parts)


@app.post("/v1beta/models/{model}:generateContent")
async def generate_content(model: str, request: Request):
    """Non-streaming shape, as the SDK uses for complete()."""
    if (err := _check_auth(request)):
        return _error(400, "INVALID_ARGUMENT", err)
    body = await request.json()
    _record(model, body, dict(request.headers))
    text = _reply_for(_extract_prompt(body))
    return JSONResponse(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 8},
        }
    )


@app.post("/v1beta/models/{model}:streamGenerateContent")
async def stream_generate_content(model: str, request: Request):
    """Streaming shape with alt=sse, exactly as the SDK's *_stream calls use."""
    if (err := _check_auth(request)):
        return _error(400, "INVALID_ARGUMENT", err)

    body = await request.json()
    _record(model, body, dict(request.headers))
    text = _reply_for(_extract_prompt(body))

    def sse() -> Iterator[str]:
        # Emit in small pieces so the client's incremental parsing is exercised.
        words = text.split(" ")
        for index, word in enumerate(words):
            piece = word if index == 0 else " " + word
            chunk = {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": piece}]}}
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.005)
        final = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": ""}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": len(words)},
        }
        yield f"data: {json.dumps(final)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/__requests")
async def requests_log():
    """Expose what the endpoint received, so tests in another process can assert."""
    return JSONResponse(
        [{k: v for k, v in r.items() if k != "body"} for r in REQUESTS]
    )


@app.post("/__reset")
async def reset():
    REQUESTS.clear()
    return JSONResponse({"cleared": True})


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
