"""End-to-end tests: the real google-genai SDK over real HTTP.

These point the genuine SDK at tests/web/fake_gemini.py, which speaks the real
Gemini REST protocol. Everything is exercised for real — request serialisation,
HTTP transport, SSE parsing, conversation assembly — and only the model behind
the endpoint is fake.

This is the closest the suite can get to a live Gemini call without network
access to Google. It is what proves the app actually holds a conversation,
rather than each layer working in isolation.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from jarvis.config import Settings
from jarvis.llm import Assistant, JarvisError
from jarvis.memory import USER, Message
from jarvis.server import create_app

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "web"))
import fake_gemini  # noqa: E402

GOOD_KEY = "AQ.fake-key-for-e2e"
BAD_KEY = "AQ.bad-key-for-tests"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def gemini_url():
    """Run the fake Gemini endpoint on its own port for the duration of the module."""
    port = _free_port()
    config = uvicorn.Config(fake_gemini.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:  # pragma: no cover - only on a wedged test machine
        raise RuntimeError("fake Gemini endpoint never started")

    fake_gemini.REQUESTS.clear()
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _clear_log(gemini_url):
    fake_gemini.REQUESTS.clear()


def settings_for(url: str, key: str = GOOD_KEY, **kw) -> Settings:
    return Settings(api_key=key, model="gemini-2.5-flash", base_url=url, **kw)


# ------------------------------------------------------- the real SDK, streaming
def test_real_sdk_streams_a_reply(gemini_url):
    assistant = Assistant(settings_for(gemini_url))
    chunks = list(assistant.stream([Message(role=USER, text="hello there")]))
    assert len(chunks) > 1, "expected a genuine token stream, not one blob"
    assert "fake endpoint" in "".join(chunks)


def test_real_sdk_reaches_the_endpoint_with_the_right_request(gemini_url):
    assistant = Assistant(settings_for(gemini_url))
    assistant.complete([Message(role=USER, text="hello there")])

    assert len(fake_gemini.REQUESTS) == 1
    req = fake_gemini.REQUESTS[0]
    assert req["model"] == "gemini-2.5-flash"
    assert req["api_key"] == GOOD_KEY, "key must be sent as the x-goog-api-key header"
    assert "Jarvis" in req["system"], "system instruction must be sent"
    assert req["contents"][0] == {"role": "user", "parts": [{"text": "hello there"}]}


# ------------------------------------------------------------ multi-turn history
def test_history_grows_and_alternates_roles(gemini_url, tmp_path):
    """The core behaviour: each turn must carry the whole prior conversation."""
    app = create_app(settings_for(gemini_url, data_dir=tmp_path))
    client = TestClient(app)

    for message in ("hello there", "greet me again please", "one more"):
        response = client.post("/api/chat", json={"message": message})
        assert response.status_code == 200

    sizes = [len(r["contents"]) for r in fake_gemini.REQUESTS]
    assert sizes == [1, 3, 5], f"history did not grow as expected: {sizes}"

    last = fake_gemini.REQUESTS[-1]["contents"]
    assert [c["role"] for c in last] == ["user", "model", "user", "model", "user"]
    assert [c["parts"][0]["text"] for c in last][::2] == [
        "hello there",
        "greet me again please",
        "one more",
    ]


def test_stream_events_are_well_formed(gemini_url, tmp_path):
    client = TestClient(create_app(settings_for(gemini_url, data_dir=tmp_path)))
    response = client.post("/api/chat", json={"message": "hello there"})
    events = [
        json.loads(line[5:])
        for line in response.text.split("\n\n")
        if line.startswith("data:")
    ]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert kinds.count("token") > 1
    reply = "".join(e["text"] for e in events if e["type"] == "token")
    assert reply == events[-1]["reply"]


# ------------------------------------------------------------------ error paths
def test_rejected_key_produces_a_clean_error(gemini_url):
    assistant = Assistant(settings_for(gemini_url, key=BAD_KEY))
    with pytest.raises(JarvisError) as excinfo:
        assistant.complete([Message(role=USER, text="hello")])
    assert "API key" in str(excinfo.value)
    assert BAD_KEY not in str(excinfo.value), "the key must not leak into the error"


def test_rejected_key_over_http_rolls_back(gemini_url, tmp_path):
    tmp = tmp_path
    client = TestClient(create_app(settings_for(gemini_url, key=BAD_KEY, data_dir=tmp)))
    body = client.post("/api/chat", json={"message": "this will fail"}).text
    assert '"type": "error"' in body
    assert BAD_KEY not in body
    assert client.get("/api/health").json()["messages"] == []
    # Nothing was ever streamed, so rollback means the file is never created at
    # all -- or, if it exists from an earlier reset, holds no messages.
    path = tmp / "history.json"
    assert not path.exists() or json.loads(path.read_text())["messages"] == []


# ------------------------------------------------------- conversation durability
def test_conversation_survives_a_restart(gemini_url, tmp_path):
    tmp = tmp_path
    client = TestClient(create_app(settings_for(gemini_url, data_dir=tmp)))
    client.post("/api/chat", json={"message": "hello there"})

    reloaded = TestClient(create_app(settings_for(gemini_url, data_dir=tmp)))
    messages = reloaded.get("/api/health").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]

    # and the next turn replays that restored history to the model
    reloaded.post("/api/chat", json={"message": "and again"})
    assert len(fake_gemini.REQUESTS[-1]["contents"]) == 3


# ------------------------------------------------- long-conversation trimming
def test_history_trimmed_past_max_turns_stays_valid_for_gemini(gemini_url, tmp_path):
    """Once trimming kicks in, what reaches the model must still be well formed.

    max_turns bounds *stored* history; the in-flight question is appended on top,
    so a request carries at most max_turns + 1 messages. The important part is
    structural: Gemini expects the first turn to be a user turn, never two model
    turns in a row, and never an empty message.
    """
    max_turns = 4
    client = TestClient(
        create_app(settings_for(gemini_url, data_dir=tmp_path, max_turns=max_turns))
    )

    for i in range(1, 9):
        response = client.post("/api/chat", json={"message": f"question {i}"})
        assert response.status_code == 200

        sent = fake_gemini.REQUESTS[-1]["contents"]
        roles = [c["role"] for c in sent]

        assert len(sent) <= max_turns + 1, f"turn {i}: sent {len(sent)} > {max_turns + 1}"
        assert roles[0] == "user", f"turn {i}: history starts with {roles[0]!r}"
        assert roles[-1] == "user", f"turn {i}: the new question must be last"
        assert not any(
            a == b == "model" for a, b in zip(roles, roles[1:])
        ), f"turn {i}: consecutive model turns in {roles}"
        assert all(c["parts"][0]["text"].strip() for c in sent), f"turn {i}: empty message sent"

    # The oldest exchanges were dropped and the window slid forward. With
    # max_turns=4 the stored window is [q6,a6,q7,a7], plus the new question.
    last = fake_gemini.REQUESTS[-1]["contents"]
    asked = [c["parts"][0]["text"] for c in last if c["role"] == "user"]
    assert asked == ["question 6", "question 7", "question 8"], asked
