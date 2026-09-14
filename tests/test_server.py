"""Tests for the FastAPI app, driven through TestClient (real ASGI calls)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from jarvis.config import Settings
from jarvis.memory import Conversation
from jarvis.server import create_app


def make_client(tmp_path, **overrides) -> TestClient:
    settings = Settings(
        api_key="",
        model=overrides.pop("model", "mock"),
        data_dir=tmp_path,
        max_turns=overrides.pop("max_turns", 40),
        **overrides,
    )
    return TestClient(create_app(settings))


def sse_payloads(body: str) -> list[dict]:
    return [json.loads(line[5:].strip()) for line in body.split("\n\n") if line.startswith("data:")]


def test_index_serves_ui(tmp_path):
    response = make_client(tmp_path).get("/")
    assert response.status_code == 200
    assert "Jarvis" in response.text
    assert "/static/app.js" in response.text


def test_static_assets_are_served(tmp_path):
    client = make_client(tmp_path)
    for path in ("/static/app.js", "/static/styles.css"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.content, path


def test_health_reports_model_and_key_state(tmp_path):
    data = make_client(tmp_path).get("/api/health").json()
    assert data["status"] == "ok"
    assert data["model"] == "mock"
    assert data["provider"] == "mock"
    assert data["api_key_configured"] is False
    assert data["messages"] == []


def test_chat_streams_tokens_and_persists(tmp_path):
    client = make_client(tmp_path)
    response = client.post("/api/chat", json={"message": "hello jarvis"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = sse_payloads(response.text)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert "token" in kinds

    reply = "".join(e["text"] for e in events if e["type"] == "token")
    assert "hello jarvis" in reply
    assert events[-1]["reply"] == reply

    messages = client.get("/api/health").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["text"] == "hello jarvis"
    assert messages[1]["text"] == reply


def test_history_is_written_to_disk(tmp_path):
    client = make_client(tmp_path)
    events = sse_payloads(client.post("/api/chat", json={"message": "persist please"}).text)
    reply = "".join(e["text"] for e in events if e["type"] == "token")

    stored = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
    assert [m["text"] for m in stored["messages"]] == ["persist please", reply]


def test_reset_clears_conversation(tmp_path):
    client = make_client(tmp_path)
    client.post("/api/chat", json={"message": "hi"})
    assert len(client.get("/api/health").json()["messages"]) == 2
    assert client.post("/api/reset").json()["messages"] == []


def test_missing_key_is_reported_as_sse_error_not_a_crash(tmp_path):
    client = make_client(tmp_path, model="gemini-2.5-flash")
    events = sse_payloads(client.post("/api/chat", json={"message": "hi"}).text)
    assert events[-1]["type"] == "error"
    assert "GEMINI_API_KEY" in events[-1]["message"]
    # A failed call must not leave a half-written history behind.
    assert client.get("/api/health").json()["messages"] == []


def test_key_endpoint_updates_state(tmp_path):
    client = make_client(tmp_path, model="gemini-2.5-flash")
    assert client.get("/api/health").json()["api_key_configured"] is False
    response = client.post("/api/key", json={"api_key": "  AIza-test  "})
    assert response.status_code == 200
    assert response.json()["api_key_configured"] is True
    assert client.get("/api/health").json()["api_key_configured"] is True


def test_key_is_never_written_to_disk(tmp_path):
    client = make_client(tmp_path)
    client.post("/api/key", json={"api_key": "AIza-secret-value"})
    client.post("/api/chat", json={"message": "say something"})  # force a history write
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written, "expected the conversation history file to exist"
    for path in written:
        assert "AIza-secret-value" not in path.read_text(encoding="utf-8")


def test_empty_message_is_rejected(tmp_path):
    client = make_client(tmp_path)
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422


def test_partial_reply_survives_client_disconnect(tmp_path):
    client = make_client(tmp_path)
    with client.stream("POST", "/api/chat", json={"message": "long answer please"}) as response:
        for chunk in response.iter_text():
            if "token" in chunk:
                break  # hang up mid-stream

    messages = client.get("/api/health").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["text"] == "long answer please"
    assert messages[1]["text"].startswith("I am running")


def test_interrupted_stream_leaves_usable_history(tmp_path):
    """A hung-up client must not leave a blank turn that poisons the next request."""
    client = make_client(tmp_path)
    with client.stream("POST", "/api/chat", json={"message": "first question"}) as response:
        for chunk in response.iter_text():
            if "token" in chunk:
                break

    client.post("/api/chat", json={"message": "second question"})
    messages = client.get("/api/health").json()["messages"]
    texts = [m["text"] for m in messages]
    assert texts[-2] == "second question"
    assert all(text.strip() for text in texts), texts
    assert "first question" in texts


def test_failed_request_leaves_no_history(tmp_path):
    client = make_client(tmp_path, model="gemini-2.5-flash")
    client.post("/api/chat", json={"message": "hi"})
    assert not (tmp_path / "history.json").exists()


def test_conversation_survives_restart(tmp_path):
    make_client(tmp_path).post("/api/chat", json={"message": "remember me"})
    reloaded = make_client(tmp_path)
    texts = [m["text"] for m in reloaded.get("/api/health").json()["messages"]]
    assert texts[0] == "remember me"
    assert len(texts) == 2


def test_conversation_helper_is_importable(tmp_path):
    assert Conversation(path=tmp_path / "x.json").history() == []


@pytest.mark.parametrize("message", ["ping", "tell me a joke"])
def test_chat_echoes_the_prompt_in_mock_mode(tmp_path, message):
    client = make_client(tmp_path)
    events = sse_payloads(client.post("/api/chat", json={"message": message}).text)
    assert message in events[-1]["reply"]


# ---------------------------------------------------------- runtime model switch
def test_model_endpoint_switches_provider(tmp_path):
    client = make_client(tmp_path, model="gemini-2.5-flash")
    assert client.get("/api/health").json()["provider"] == "google-ai-studio"

    response = client.post("/api/model", json={"model": "mock"})
    assert response.status_code == 200
    assert response.json()["model"] == "mock"
    assert response.json()["provider"] == "mock"

    events = sse_payloads(client.post("/api/chat", json={"message": "hi"}).text)
    assert events[-1]["type"] == "done"


def test_model_endpoint_ignores_blank_and_reports_health(tmp_path):
    client = make_client(tmp_path, model="gemini-2.5-flash")
    assert client.post("/api/model", json={"model": "   "}).json()["model"] == "gemini-2.5-flash"
    assert client.get("/api/health").json()["model"] == "gemini-2.5-flash"


def test_switching_to_real_model_without_key_fails_cleanly(tmp_path):
    client = make_client(tmp_path, model="mock")
    assert sse_payloads(client.post("/api/chat", json={"message": "hi"}).text)[-1]["type"] == "done"

    client.post("/api/model", json={"model": "gemini-2.5-flash"})
    events = sse_payloads(client.post("/api/chat", json={"message": "hi again"}).text)
    assert events[-1]["type"] == "error"
    assert "GEMINI_API_KEY" in events[-1]["message"]
    # The successful mock turn is still on record; the failed one is not.
    assert len(client.get("/api/health").json()["messages"]) == 2


def test_ui_exposes_model_and_route_controls(tmp_path):
    client = make_client(tmp_path)
    index = client.get("/").text
    for needle in ('id="model"', 'id="route"', 'id="apiKey"', "Browser → Google direct"):
        assert needle in index, needle
    script = client.get("/static/app.js").text
    assert "/api/model" in script and "/api/key" in script
