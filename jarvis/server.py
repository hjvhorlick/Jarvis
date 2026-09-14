"""FastAPI app serving the Jarvis web UI and its chat API.

Endpoints
    GET  /                 single-page chat UI
    GET  /api/health       model, provider, key status
    POST /api/chat         SSE stream of tokens for one user message
    POST /api/key          set a runtime API key (memory only, never written to disk)
    POST /api/model        switch model at runtime (e.g. to "mock")
    POST /api/reset        clear the conversation
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, load_settings
from .llm import Assistant, JarvisError, MissingAPIKey
from .memory import ASSISTANT, USER, Conversation, Message

WEB_DIR = Path(__file__).resolve().parent / "web"

# How often a streaming reply is written to disk (seconds).
FLUSH_SECONDS = 0.5


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class KeyRequest(BaseModel):
    api_key: str = ""


class ModelRequest(BaseModel):
    model: str = ""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    # autosave=False: the chat endpoint decides exactly when to touch disk, so a
    # reply can be flushed mid-stream and rolled back on failure.
    conversation = Conversation(
        path=settings.data_dir / "history.json",
        max_turns=settings.max_turns,
        autosave=False,
    )

    app = FastAPI(title="Jarvis", version=__version__)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    # --------------------------------------------------------------------- state
    def current_assistant() -> Assistant:
        return Assistant(settings)

    def public_state() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "model": settings.model,
            "provider": Assistant(settings).provider,
            "api_key_configured": settings.api_key_configured,
            "messages": [
                {"role": m.role, "text": m.text} for m in conversation.history()
            ],
        }

    # ------------------------------------------------------------------- routes
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse(public_state())

    @app.post("/api/key")
    def set_key(payload: KeyRequest) -> JSONResponse:
        settings.api_key = payload.api_key.strip()
        return JSONResponse(
            {
                "api_key_configured": settings.api_key_configured,
                "model": settings.model,
                "provider": Assistant(settings).provider,
            }
        )

    @app.post("/api/model")
    def set_model(payload: ModelRequest) -> JSONResponse:
        """Switch model at runtime, e.g. to `mock` for an offline demo."""
        model = payload.model.strip()
        if model:
            settings.model = model
        return JSONResponse(
            {
                "model": settings.model,
                "provider": Assistant(settings).provider,
                "api_key_configured": settings.api_key_configured,
            }
        )

    @app.post("/api/reset")
    def reset() -> JSONResponse:
        conversation.reset()
        conversation.save()
        return JSONResponse(public_state())

    @app.post("/api/chat")
    async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
        message = payload.message.strip()
        if not message:
            return JSONResponse({"detail": "message must not be empty"}, status_code=422)

        # Clear any placeholder left by an earlier interrupted stream.
        if conversation.drop_blank_tail():
            conversation.save()
        history = conversation.history() + [Message(role=USER, text=message)]

        def event(kind: str, **data: object) -> str:
            return f"data: {json.dumps({'type': kind, **data}, ensure_ascii=False)}\n\n"

        def generate() -> Iterator[str]:
            yield event(
                "start", model=settings.model, provider=Assistant(settings).provider
            )
            collected: list[str] = []
            flushed = False
            last_flush = 0.0

            def flush() -> None:
                nonlocal flushed, last_flush
                conversation.replace_last("".join(collected))
                conversation.save()
                flushed = True
                last_flush = time.monotonic()

            def rollback() -> None:
                # Undo the placeholder (and the user turn) for a failed request.
                conversation.pop_last()
                conversation.pop_last()
                if flushed:
                    conversation.save()

            conversation.add(USER, message)
            conversation.add(ASSISTANT, "")  # placeholder filled as tokens arrive
            try:
                assistant = current_assistant()
                for token in assistant.stream(history):
                    collected.append(token)
                    # Persist as we go: a client that hangs up mid-reply still
                    # leaves the user turn and the partial answer on disk.
                    if not flushed or time.monotonic() - last_flush >= FLUSH_SECONDS:
                        flush()
                    yield event("token", text=token)
            except (JarvisError, MissingAPIKey) as exc:
                rollback()
                yield event("error", message=str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - surface anything to the UI
                rollback()
                yield event("error", message=f"Unexpected failure: {exc}")
                return

            flush()
            yield event("done", reply="".join(collected))

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


def main(argv: list[str] | None = None) -> int:
    """``python -m jarvis serve`` entry point."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="jarvis serve", description="Run the Jarvis web app")
    parser.add_argument("--host", default=None, help="Bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port (default 8000)")
    parser.add_argument("--model", default=None, help="Override the model name")
    parser.add_argument("--reload", action="store_true", help="Uvicorn auto-reload")
    args = parser.parse_args(argv)

    settings = load_settings(host=args.host, port=args.port, model=args.model)
    if not settings.api_key_configured and not settings.is_mock:
        print(
            "warning: GEMINI_API_KEY is not set — chat will report a missing-key error "
            "until you add one (cp .env.example .env, or use the UI's Settings panel).\n"
            "         Tip: JARVIS_MODEL=mock runs fully offline.",
            flush=True,
        )
    print(f"Jarvis {__version__} on http://{settings.host}:{settings.port} (model={settings.model})")
    uvicorn.run(
        "jarvis.server:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=args.reload,
    )
    return 0
