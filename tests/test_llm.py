"""Tests for jarvis.llm — streaming, error mapping, and the Gemini transport."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.llm import (
    Assistant,
    JarvisError,
    MissingAPIKey,
    _chunk_text,
    _friendly,
    gemini_transport,
    mock_transport,
)
from jarvis.memory import ASSISTANT, USER, Message


def history(*pairs: tuple[str, str]) -> list[Message]:
    return [Message(role=role, text=text) for role, text in pairs]


# ------------------------------------------------------------------ mock provider
def test_mock_transport_streams_every_word():
    settings = Settings(api_key="", model="mock")
    assistant = Assistant(settings)
    assert assistant.provider == "mock"
    reply = assistant.complete(history((USER, "ping")))
    assert "ping" in reply
    assert "offline mock mode" in reply


def test_mock_transport_custom_reply():
    transport = mock_transport(reply="exactly this")
    assert "".join(transport(history((USER, "x")), "sys")) == "exactly this"


def test_stream_yields_multiple_chunks():
    settings = Settings(model="mock")
    chunks = list(Assistant(settings).stream(history((USER, "one two three"))))
    assert len(chunks) > 1
    assert "".join(chunks).startswith("I am running")


# ------------------------------------------------------------------- input checks
def test_stream_requires_a_user_last_message():
    assistant = Assistant(Settings(model="mock"))
    with pytest.raises(JarvisError, match="last message must be from the user"):
        list(assistant.stream(history((USER, "hi"), (ASSISTANT, "hello"))))


def test_stream_rejects_empty_history():
    assistant = Assistant(Settings(model="mock"))
    with pytest.raises(JarvisError, match="conversation is empty"):
        list(assistant.stream([]))


def test_missing_key_raises_helpful_error():
    assistant = Assistant(Settings(api_key="   ", model="gemini-2.5-flash"))
    with pytest.raises(MissingAPIKey, match="GEMINI_API_KEY"):
        list(assistant.stream(history((USER, "hi"))))


# ------------------------------------------------------------------- transport io
def test_injected_transport_sees_history_and_system_prompt():
    seen = {}

    def fake(messages, system_prompt):
        seen["messages"] = messages
        seen["system_prompt"] = system_prompt
        yield "ok"

    settings = Settings(model="anything", system_prompt="be brief")
    assistant = Assistant(settings, transport=fake)
    assert assistant.complete(history((USER, "hi"), (ASSISTANT, "yo"), (USER, "again"))) == "ok"
    assert [m.role for m in seen["messages"]] == [USER, ASSISTANT, USER]
    assert seen["system_prompt"] == "be brief"


def test_gemini_transport_builds_contents_and_extracts_text(monkeypatch):
    """Drive the real transport with a stubbed google-genai client (no network)."""
    calls = {}

    class FakeModels:
        def generate_content_stream(self, model, contents, config):
            calls["model"] = model
            calls["contents"] = contents
            calls["config"] = config
            yield types.SimpleNamespace(
                candidates=[types.SimpleNamespace(content=types.SimpleNamespace(parts=[types.SimpleNamespace(text="Hel")]))]
            )
            yield types.SimpleNamespace(candidates=[])
            yield types.SimpleNamespace(
                candidates=[types.SimpleNamespace(content=types.SimpleNamespace(parts=[types.SimpleNamespace(text="lo")]))]
            )

    class FakeClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    fake_module = types.SimpleNamespace(Client=FakeClient, types=fake_types)
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=fake_module))
    monkeypatch.setitem(sys.modules, "google.genai", fake_module)

    transport = gemini_transport(api_key="AIza-test", model="gemini-2.5-flash")
    text = "".join(transport(history((USER, "hi"), (ASSISTANT, "hey"), (USER, "again")), "be brief"))

    assert text == "Hello"
    assert calls["api_key"] == "AIza-test"
    assert calls["model"] == "gemini-2.5-flash"
    assert calls["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hey"}]},
        {"role": "user", "parts": [{"text": "again"}]},
    ]
    assert calls["config"].system_instruction == "be brief"


def test_gemini_transport_wraps_failures(monkeypatch):
    class Boom:
        def __init__(self, api_key):
            raise RuntimeError("API key not valid")

    fake_types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    fake_module = types.SimpleNamespace(Client=Boom, types=fake_types)
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=fake_module))
    monkeypatch.setitem(sys.modules, "google.genai", fake_module)

    transport = gemini_transport(api_key="bad", model="gemini-2.5-flash")
    with pytest.raises(JarvisError, match="rejected by Google AI Studio"):
        list(transport(history((USER, "hi")), "sys"))


# ------------------------------------------------------------------ small helpers
def test_chunk_text_handles_missing_parts():
    assert _chunk_text(types.SimpleNamespace(candidates=[])) == ""
    assert _chunk_text(types.SimpleNamespace(candidates=[types.SimpleNamespace(content=None)])) == ""
    assert (
        _chunk_text(
            types.SimpleNamespace(
                candidates=[
                    types.SimpleNamespace(
                        content=types.SimpleNamespace(parts=[types.SimpleNamespace(text="a"), types.SimpleNamespace(text="b")])
                    )
                ]
            )
        )
        == "ab"
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("400 API_KEY_INVALID: API key not valid", "the API key was rejected by Google AI Studio (check GEMINI_API_KEY)."),
        ("RESOURCE_EXHAUSTED quota exceeded", "rate limit or quota exceeded — wait a moment and retry."),
        ("[SSL: CERTIFICATE_VERIFY_FAILED]", "network/TLS blocked"),
        ("", "RuntimeError"),
    ],
)
def test_friendly_error_mapping(raw, expected):
    exc = RuntimeError(raw) if raw else RuntimeError()
    assert expected in _friendly(exc)


# ------------------------------------------------- AI Studio key format support
# AI Studio now issues "Auth keys" (AQ. prefix) instead of legacy AIza traffic
# keys. Jarvis must stay format-agnostic: the key is an opaque string.
AQ_KEY = "AQ.Ab8Rexample-not-a-real-key-value-here"
AIZA_KEY = "AIzaSyExample-not-a-real-key-value"


@pytest.mark.parametrize("key", [AQ_KEY, AIZA_KEY])
def test_assistant_accepts_any_key_format(monkeypatch, key):
    seen = {}

    class FakeModels:
        def generate_content_stream(self, model, contents, config):
            yield types.SimpleNamespace(
                candidates=[
                    types.SimpleNamespace(content=types.SimpleNamespace(parts=[types.SimpleNamespace(text="ok")]))
                ]
            )

    class FakeClient:
        def __init__(self, api_key):
            seen["api_key"] = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    fake_module = types.SimpleNamespace(Client=FakeClient, types=fake_types)
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=fake_module))
    monkeypatch.setitem(sys.modules, "google.genai", fake_module)

    assistant = Assistant(Settings(api_key=key, model="gemini-2.5-flash"))
    assert assistant.complete(history((USER, "hi"))) == "ok"
    assert seen["api_key"] == key  # passed through verbatim, unmodified


def test_no_key_format_validation_in_the_codebase():
    """Guard against anyone reintroducing an AIza regex, which would reject AQ. keys.

    security.py is excluded: it names the formats deliberately so it can scrub
    them out of chat text. It never accepts or rejects a key.
    """
    root = Path(__file__).resolve().parent.parent / "jarvis"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "security.py":
            continue
        if "AIza" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == [], f"hardcoded key format assumption in: {offenders}"
