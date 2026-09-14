"""Tests for key protection: redaction, masking, and the auth gate."""

from __future__ import annotations

import json
import sys
import types

import pytest
from fastapi.testclient import TestClient

from jarvis.config import Settings
from jarvis.llm import Assistant, JarvisError, gemini_transport
from jarvis.memory import USER, Message
from jarvis.security import REDACTED, mask_key, redact
from jarvis.server import create_app

FAKE_KEY = "AQ.Ab8RthisKeyIsFakeAndUsedOnlyForTests1234567890"
TOKEN = "s3cret-server-token"


def make_client(tmp_path, **overrides) -> TestClient:
    settings = Settings(
        api_key=overrides.pop("api_key", ""),
        auth_token=overrides.pop("auth_token", ""),
        model=overrides.pop("model", "mock"),
        data_dir=tmp_path,
    )
    return TestClient(create_app(settings))


def history(*pairs):
    return [Message(role=r, text=t) for r, t in pairs]


# ---------------------------------------------------------------------- redact()
def test_redact_removes_the_exact_key():
    assert redact(f"failed for {FAKE_KEY}", FAKE_KEY) == f"failed for {REDACTED}"


def test_redact_removes_a_truncated_echo():
    assert FAKE_KEY not in redact(f"head was {FAKE_KEY[:16]} ok", FAKE_KEY)
    assert FAKE_KEY not in redact(f"tail was {FAKE_KEY[-16:]} ok", FAKE_KEY)


@pytest.mark.parametrize(
    "raw",
    [
        "https://generativelanguage.googleapis.com/v1beta/models?key=SECRET123&x=1",
        'headers: {"x-goog-api-key": "SECRET123"}',
        "Authorization: Bearer SECRET123",
        "x-goog-api-key=SECRET123",
    ],
)
def test_redact_scrubs_credential_shapes(raw):
    out = redact(raw)
    assert "SECRET123" not in out
    assert REDACTED in out


def test_redact_handles_empty_secrets_and_text():
    assert redact("nothing here", "", None) == "nothing here"
    assert redact("", FAKE_KEY) == ""


def test_redact_does_not_touch_unrelated_text():
    assert redact("rate limit exceeded", FAKE_KEY) == "rate limit exceeded"


# -------------------------------------------------------------------- mask_key()
def test_mask_key_is_not_reversible():
    hint = mask_key(FAKE_KEY)
    assert hint == f"{FAKE_KEY[:6]}\u2026{FAKE_KEY[-3:]}"
    assert FAKE_KEY not in hint
    assert len(hint) < len(FAKE_KEY) / 2


def test_mask_key_fully_masks_short_values():
    assert mask_key("abc") == "***"
    assert mask_key("") == ""


# ------------------------------------------------- the key never leaves the app
def test_no_endpoint_ever_returns_the_key(tmp_path):
    client = make_client(tmp_path, api_key=FAKE_KEY, model="gemini-2.5-flash")
    responses = [
        client.get("/"),
        client.get("/api/health"),
        client.get("/static/app.js"),
        client.post("/api/key", json={"api_key": FAKE_KEY}),
        client.post("/api/model", json={"model": "mock"}),
        client.post("/api/chat", json={"message": "hello"}),
        client.post("/api/reset"),
    ]
    for response in responses:
        assert FAKE_KEY not in response.text, response.request.url
        assert FAKE_KEY[:16] not in response.text, response.request.url


def test_key_is_still_not_written_to_disk(tmp_path):
    client = make_client(tmp_path, model="mock")
    client.post("/api/key", json={"api_key": FAKE_KEY})
    client.post("/api/chat", json={"message": "hi"})
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files
    for path in files:
        assert FAKE_KEY not in path.read_text(encoding="utf-8")


def test_health_shows_a_hint_not_the_key(tmp_path):
    data = make_client(tmp_path, api_key=FAKE_KEY).get("/api/health").json()
    assert data["api_key_configured"] is True
    assert data["api_key_hint"] == f"{FAKE_KEY[:6]}\u2026{FAKE_KEY[-3:]}"
    assert FAKE_KEY not in json.dumps(data)


# ------------------------------------------------------------- redaction at the wire
def test_sse_error_is_redacted_before_it_reaches_the_browser(tmp_path, monkeypatch):
    client = make_client(tmp_path, api_key=FAKE_KEY, model="gemini-2.5-flash")

    def boom(self, hist):
        raise RuntimeError(f"upstream said key={FAKE_KEY} for x-goog-api-key: {FAKE_KEY}")

    monkeypatch.setattr(Assistant, "stream", boom)
    body = client.post("/api/chat", json={"message": "hi"}).text
    assert FAKE_KEY not in body
    assert REDACTED in body
    # the failure must still be reported as an error event, not swallowed
    assert '"type": "error"' in body


def test_transport_redacts_the_key_from_its_own_errors(monkeypatch):
    class Boom:
        def __init__(self, api_key):
            raise RuntimeError(f"400 for request with {api_key}")

    fake_types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    fake_module = types.SimpleNamespace(Client=Boom, types=fake_types)
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=fake_module))
    monkeypatch.setitem(sys.modules, "google.genai", fake_module)

    with pytest.raises(JarvisError) as excinfo:
        list(gemini_transport(api_key=FAKE_KEY, model="gemini-2.5-flash")(history((USER, "hi")), "sys"))
    assert FAKE_KEY not in str(excinfo.value)


# ---------------------------------------------------------------------- auth gate
def test_endpoints_require_the_token_when_configured(tmp_path):
    client = make_client(tmp_path, auth_token=TOKEN)
    for method, url, payload in (
        ("post", "/api/chat", {"message": "hi"}),
        ("post", "/api/key", {"api_key": "x"}),
        ("post", "/api/model", {"model": "mock"}),
        ("post", "/api/reset", {}),
    ):
        assert getattr(client, method)(url, json=payload).status_code == 401, url
        assert getattr(client, method)(
            url, json=payload, headers={"Authorization": "Bearer wrong"}
        ).status_code == 401, url


def test_token_grants_access_via_either_header(tmp_path):
    client = make_client(tmp_path, auth_token=TOKEN)
    for headers in (
        {"Authorization": f"Bearer {TOKEN}"},
        {"X-Jarvis-Token": TOKEN},
    ):
        response = client.post("/api/chat", json={"message": "hi"}, headers=headers)
        assert response.status_code == 200, headers
        assert '"type": "done"' in response.text


def test_health_stays_open_but_reports_the_requirement(tmp_path):
    client = make_client(tmp_path, auth_token=TOKEN)
    data = client.get("/api/health").json()
    assert data["auth_required"] is True
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_auth_optional_by_default(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/api/health").json()["auth_required"] is False
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 200


def test_token_is_not_echoed_back(tmp_path):
    client = make_client(tmp_path, auth_token=TOKEN)
    for response in (
        client.get("/api/health"),
        client.post("/api/chat", json={"message": "hi"}, headers={"Authorization": f"Bearer {TOKEN}"}),
    ):
        assert TOKEN not in response.text


# ------------------------------------------------------------------ UI plumbing
def test_ui_has_token_and_remember_controls(tmp_path):
    index = make_client(tmp_path).get("/").text
    for needle in ('id="authToken"', 'id="rememberKey"', "Remember the key"):
        assert needle in index, needle
    script = make_client(tmp_path).get("/static/app.js").text
    assert "sessionStorage" in script and "Authorization" in script
    # the browser key must only reach localStorage behind the remember checkbox
    assert "function storeKey(key)" in script
    body = script.split("function storeKey(key)")[1].split("}")[0]
    assert "rememberKey.checked" in body
