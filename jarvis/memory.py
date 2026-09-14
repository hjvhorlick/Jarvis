"""Conversation memory: an ordered message list with optional JSON persistence."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .security import scrub_message

USER = "user"
ASSISTANT = "assistant"
VALID_ROLES = (USER, ASSISTANT)

# Role names understood by the Gemini API.
GEMINI_ROLE = {USER: "user", ASSISTANT: "model"}


@dataclass(frozen=True)
class Message:
    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"invalid role: {self.role!r}")

    @property
    def gemini_role(self) -> str:
        return GEMINI_ROLE[self.role]


class Conversation:
    """Keeps the chat history in memory and optionally mirrors it to disk."""

    def __init__(
        self,
        path: str | Path | None = None,
        max_turns: int = 40,
        autosave: bool = True,
    ) -> None:
        self.path = Path(path) if path else None
        self.max_turns = max(1, int(max_turns))
        self.autosave = autosave
        self.messages: list[Message] = []
        if self.path and self.path.exists():
            self.load()

    def _maybe_save(self) -> None:
        if self.autosave and self.path:
            self.save()

    # ------------------------------------------------------------------ mutating
    def add(self, role: str, text: str) -> Message:
        # Never persist a credential a human pasted into the chat box.
        message = Message(role=role, text=scrub_message(text))
        self.messages.append(message)
        self._trim()
        self._maybe_save()
        return message

    def replace_last(self, text: str) -> Message:
        """Overwrite the text of the newest message (used while streaming)."""
        if not self.messages:
            raise IndexError("cannot replace the last message of an empty conversation")
        self.messages[-1] = Message(role=self.messages[-1].role, text=text)
        return self.messages[-1]

    def drop_blank_tail(self) -> bool:
        """Remove trailing empty messages left by an interrupted stream."""
        before = len(self.messages)
        while self.messages and not self.messages[-1].text.strip():
            self.messages.pop()
        return len(self.messages) != before

    def pop_last(self) -> Message | None:
        """Remove and return the newest message, or None when empty."""
        if not self.messages:
            return None
        message = self.messages.pop()
        self._maybe_save()
        return message

    def reset(self) -> None:
        self.messages.clear()
        self._maybe_save()

    def _trim(self) -> None:
        """Cap history at ``max_turns`` messages, never starting mid-response."""
        if len(self.messages) <= self.max_turns:
            return
        dropped = self.messages[-self.max_turns :]
        while dropped and dropped[0].role != USER:
            dropped.pop(0)
        self.messages = dropped

    # ------------------------------------------------------------------- reading
    def history(self) -> list[Message]:
        return list(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"messages": [asdict(m) for m in self.messages]}

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: never leave a half-written history behind.
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            Path(tmp_name).replace(self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return  # A corrupt history file must not stop Jarvis from booting.
        loaded: list[Message] = []
        for item in raw.get("messages", []):
            try:
                loaded.append(Message(role=item["role"], text=str(item["text"])))
            except (KeyError, TypeError, ValueError):
                continue
        self.messages = loaded
        self._trim()
        self.drop_blank_tail()

    @classmethod
    def from_messages(cls, messages: Iterable[Message]) -> "Conversation":
        convo = cls()
        convo.messages = list(messages)
        return convo
