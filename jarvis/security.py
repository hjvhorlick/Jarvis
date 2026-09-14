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


def mask_key(key: str) -> str:
    """A short, non-reversible hint so a UI can show *which* key is loaded."""
    value = (key or "").strip()
    if not value:
        return ""
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}\u2026{value[-3:]}"
