"""Streaming text generation against the Google AI Studio (Gemini) API.

The network transport is injected, so the rest of Jarvis can be exercised with
a fake transport in tests and with the ``mock`` model in offline demos.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator

from .config import Settings
from .memory import Message

# transport(history, system_prompt) -> iterator of text chunks
Transport = Callable[[list[Message], str], Iterator[str]]


class JarvisError(RuntimeError):
    """Base class for user-facing Jarvis failures."""


class MissingAPIKey(JarvisError):
    """Raised when a real model is requested but no API key is available."""


class Assistant:
    """Turns a conversation into a stream of assistant text chunks."""

    def __init__(self, settings: Settings, transport: Transport | None = None) -> None:
        self.settings = settings
        self._transport = transport

    # ------------------------------------------------------------------- helpers
    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def provider(self) -> str:
        return "mock" if self.settings.is_mock else "google-ai-studio"

    def transport(self) -> Transport:
        if self._transport is not None:
            return self._transport
        if self.settings.is_mock:
            return mock_transport()
        if not self.settings.api_key_configured:
            raise MissingAPIKey(
                "No Google AI Studio API key found. Set GEMINI_API_KEY in .env, "
                "or paste a key in the web UI's Settings panel."
            )
        return gemini_transport(api_key=self.settings.api_key, model=self.settings.model)

    # --------------------------------------------------------------------- reply
    def stream(self, history: Iterable[Message]) -> Iterator[str]:
        """Yield assistant text chunks for ``history`` (last message = latest)."""
        messages = list(history)
        if not messages:
            raise JarvisError("Nothing to answer: the conversation is empty.")
        if messages[-1].role != "user":
            raise JarvisError("The last message must be from the user.")
        yield from self.transport()(messages, self.settings.system_prompt)

    def complete(self, history: Iterable[Message]) -> str:
        """Non-streaming convenience wrapper: the full reply as one string."""
        return "".join(self.stream(history))


# --------------------------------------------------------------------- transports
def mock_transport(reply: str | None = None) -> Transport:
    """Offline transport: streams a canned reply word by word. No network use."""

    def transport(history: list[Message], system_prompt: str) -> Iterator[str]:
        prompt = history[-1].text if history else ""
        text = reply or (
            "I am running in offline mock mode, so no Gemini call was made. "
            f'You said: "{prompt}". Set JARVIS_MODEL back to a real model such as '
            "gemini-2.5-flash and provide GEMINI_API_KEY to talk to the real thing."
        )
        words = text.split(" ")
        for index, word in enumerate(words):
            yield word if index == 0 else " " + word

    return transport


def gemini_transport(api_key: str, model: str) -> Transport:
    """Real transport using the official ``google-genai`` SDK."""

    def transport(history: list[Message], system_prompt: str) -> Iterator[str]:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover - dependency is required
            raise JarvisError(
                "The google-genai package is not installed. Run: pip install -r requirements.txt"
            ) from exc

        contents = [
            {"role": message.gemini_role, "parts": [{"text": message.text}]}
            for message in history
        ]
        try:
            client = genai.Client(api_key=api_key)
            config = genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            )
            for chunk in client.models.generate_content_stream(
                model=model, contents=contents, config=config
            ):
                text = _chunk_text(chunk)
                if text:
                    yield text
        except JarvisError:
            raise
        except Exception as exc:
            raise JarvisError(f"Gemini request failed: {_friendly(exc)}") from exc

    return transport


def _chunk_text(chunk: object) -> str:
    """Pull text out of a genai chunk without assuming ``.text`` is populated."""
    candidates = getattr(chunk, "candidates", None)
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(part, "text", "") or "" for part in parts)


def _friendly(exc: BaseException) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if "api key not valid" in lowered or "api_key" in lowered:
        return "the API key was rejected by Google AI Studio (check GEMINI_API_KEY)."
    if "quota" in lowered or "429" in message:
        return "rate limit or quota exceeded — wait a moment and retry."
    if "ssl" in lowered or "timed out" in lowered or "getaddrinfo" in lowered:
        return (
            "this machine could not reach generativelanguage.googleapis.com "
            "(network/TLS blocked). Run Jarvis on a machine with open internet "
            "access, or switch the web UI's Settings route to 'Browser -> Google direct'."
        )
    return message[:400]
