"""Tests for jarvis.memory."""

from __future__ import annotations

import json

import pytest

from jarvis.memory import ASSISTANT, USER, Conversation, Message


def test_message_rejects_unknown_role():
    with pytest.raises(ValueError):
        Message(role="system", text="nope")


def test_gemini_role_mapping():
    assert Message(role=USER, text="a").gemini_role == "user"
    assert Message(role=ASSISTANT, text="b").gemini_role == "model"


def test_add_and_history_order(tmp_path):
    convo = Conversation(path=tmp_path / "h.json")
    convo.add(USER, "hello")
    convo.add(ASSISTANT, "hi there")
    assert [m.text for m in convo.history()] == ["hello", "hi there"]
    assert len(convo) == 2


def test_history_is_a_copy():
    convo = Conversation()
    convo.add(USER, "hello")
    convo.history().clear()
    assert len(convo) == 1


def test_trim_caps_at_max_turns_and_starts_on_user():
    convo = Conversation(max_turns=4)
    for i in range(6):
        convo.add(USER, f"q{i}")
        convo.add(ASSISTANT, f"a{i}")
    assert len(convo) == 4
    assert convo.messages[0].role == USER
    assert [m.text for m in convo.history()] == ["q4", "a4", "q5", "a5"]


def test_trim_drops_leading_assistant_message():
    convo = Conversation(max_turns=3)
    convo.messages = [
        Message(role=ASSISTANT, text="orphan"),
        Message(role=USER, text="q"),
        Message(role=ASSISTANT, text="a"),
        Message(role=USER, text="q2"),
        Message(role=ASSISTANT, text="a2"),
    ]
    convo._trim()
    # Keeps the last 3, then drops the orphaned leading assistant message.
    assert [m.text for m in convo.history()] == ["q2", "a2"]
    assert convo.messages[0].role == USER


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "nested" / "history.json"
    convo = Conversation(path=path)
    convo.add(USER, "persist me")
    convo.add(ASSISTANT, "done")

    reloaded = Conversation(path=path)
    assert [m.text for m in reloaded.history()] == ["persist me", "done"]


def test_reset_clears_disk_copy(tmp_path):
    path = tmp_path / "history.json"
    convo = Conversation(path=path)
    convo.add(USER, "hi")
    convo.reset()
    assert len(convo) == 0
    assert Conversation(path=path).history() == []


def test_corrupt_history_file_is_ignored(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")
    assert Conversation(path=path).history() == []


def test_unknown_entries_in_history_are_skipped(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "wizard", "text": "bad role"},
                    {"role": "user", "text": "good"},
                    {"nope": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    convo = Conversation(path=path)
    assert [m.text for m in convo.history()] == ["good"]


def test_in_memory_conversation_never_touches_disk():
    convo = Conversation()
    convo.add(USER, "hi")
    convo.save()  # no path -> no-op, must not raise
    assert convo.path is None


# --------------------------------------------------------- streaming durability
def test_autosave_off_defers_disk_writes(tmp_path):
    path = tmp_path / "history.json"
    convo = Conversation(path=path, autosave=False)
    convo.add(USER, "hello")
    assert not path.exists()
    convo.save()
    assert json.loads(path.read_text(encoding="utf-8"))["messages"][0]["text"] == "hello"


def test_replace_last_updates_newest_message():
    convo = Conversation()
    convo.add(USER, "hi")
    convo.add(ASSISTANT, "")
    convo.replace_last("partial")
    convo.replace_last("partial reply")
    assert convo.history()[-1].text == "partial reply"
    assert convo.history()[-1].role == ASSISTANT
    assert len(convo) == 2


def test_replace_last_on_empty_conversation_raises():
    with pytest.raises(IndexError):
        Conversation().replace_last("x")


def test_drop_blank_tail_removes_interrupted_placeholder():
    convo = Conversation()
    convo.add(USER, "hi")
    convo.add(ASSISTANT, "")
    assert convo.drop_blank_tail() is True
    assert [m.text for m in convo.history()] == ["hi"]
    assert convo.drop_blank_tail() is False


def test_drop_blank_tail_keeps_real_replies():
    convo = Conversation()
    convo.add(USER, "hi")
    convo.add(ASSISTANT, "hello")
    assert convo.drop_blank_tail() is False
    assert len(convo) == 2


def test_pop_last_returns_and_removes():
    convo = Conversation()
    convo.add(USER, "one")
    convo.add(ASSISTANT, "two")
    assert convo.pop_last().text == "two"
    assert [m.text for m in convo.history()] == ["one"]
    convo.pop_last()
    assert convo.pop_last() is None


def test_load_drops_trailing_blank_message(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "text": "hi"},
                    {"role": "assistant", "text": ""},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert [m.text for m in Conversation(path=path).history()] == ["hi"]
