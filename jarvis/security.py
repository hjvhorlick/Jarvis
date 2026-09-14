"""Keeping the API key out of anything that leaves the process.

The key must never appear in an HTTP response, an SSE error event, a log line,
or a file. Nothing here can stop the key from being used; it stops it from
being *read back*.
"""

from __future__ import annotations

import re

REDACTED = "[redacted]"

# Patterns that leak a credential embedded in a URL, a header dump, or a JSON
# blob inside an exception message.
_PATTERNS = (
    re.compile(r"(?i)\b(key|api_key|apikey)=([^&\s\"']+)", re.UNICODE),
    re.compile(r"(?i)(x-goog-api-key)([\"']?\s*[:=]\s*[\"']?)([^\"'\s,}]+)", re.UNICODE),
    re.compile(r"(?i)(authorization)([\"']?\s*[:=]\s*[\"']?bearer\s+)([^\"'\s,}]+)", re.UNICODE),
)


def redact(text: str, *secrets: str) -> str:
    """Strip secrets (and common credential shapes) out of ``text``.

    Handles the exact key, its head/tail in case an error quotes a truncated
    copy, and ``key=``/header forms whatever their value.
    """
    out = text or ""
    for secret in secrets:
        value = (secret or "").strip()
        if not value:
            continue
        out = out.replace(value, REDACTED)
        if len(value) >= 16:  # a truncated echo of the same key
            out = out.replace(value[:16], REDACTED).replace(value[-16:], REDACTED)
    for pattern in _PATTERNS:
        groups = pattern.groups
        if groups == 2:
            out = pattern.sub(rf"\1={REDACTED}", out)
        else:
            out = pattern.sub(rf"\1\2{REDACTED}", out)
    return out


# Credential shapes that should never be persisted or sent to a model, even
# when a human pastes one into the chat box by mistake.
_SECRET_SHAPES = (
    re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}\b"),   # legacy Google "traffic" key
    re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}\b"),   # Google AI Studio Auth key
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),     # OpenAI-style key
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub tokens
)

SCRUBBED = "[secret removed]"


def scrub_message(text: str) -> str:
    """Blank out credential-shaped tokens in a chat message.

    Applied before a message is persisted or sent to the model, so a key pasted
    into the chat box never reaches disk or a third party.
    """
    out = text or ""
    for pattern in _SECRET_SHAPES:
        out = pattern.sub(SCRUBBED, out)
    return out


def mask_key(key: str) -> str:
    """A short, non-reversible hint so a UI can show *which* key is loaded."""
    value = (key or "").strip()
    if not value:
        return ""
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}\u2026{value[-3:]}"
