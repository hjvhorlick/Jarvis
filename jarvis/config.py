"""Runtime configuration for Jarvis.

Everything is overridable through environment variables (or a local ``.env``),
so the same code runs on a laptop, in a container, or behind a preview proxy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "gemini-2.5-flash"
MOCK_MODEL = "mock"

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a sharp and concise personal assistant. "
    "Answer directly, keep replies short unless asked for detail, "
    "and use Markdown when it helps readability."
)


def _load_dotenv() -> None:
    """Load ``.env`` if python-dotenv is available.  Never raises."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is optional
        return
    load_dotenv(REPO_ROOT / ".env")


_load_dotenv()


@dataclass
class Settings:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_turns: int = 40
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")

    @property
    def api_key_configured(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def is_mock(self) -> bool:
        return self.model.strip().lower() == MOCK_MODEL

    def resolved(self) -> "Settings":
        """A copy of these settings, used when a per-request key is supplied."""
        return Settings(
            api_key=self.api_key,
            model=self.model,
            system_prompt=self.system_prompt,
            max_turns=self.max_turns,
            host=self.host,
            port=self.port,
            data_dir=self.data_dir,
        )


def load_settings(**overrides: object) -> Settings:
    """Build :class:`Settings` from the environment plus explicit overrides."""
    env = os.environ
    settings = Settings(
        api_key=env.get("GEMINI_API_KEY", "").strip(),
        model=env.get("JARVIS_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        system_prompt=env.get("JARVIS_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()
        or DEFAULT_SYSTEM_PROMPT,
        max_turns=_as_int(env.get("JARVIS_MAX_TURNS"), 40),
        host=env.get("JARVIS_HOST", "0.0.0.0").strip() or "0.0.0.0",
        port=_as_int(env.get("JARVIS_PORT"), 8000),
        data_dir=Path(env.get("JARVIS_DATA_DIR") or (REPO_ROOT / "data")),
    )
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    return settings


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default
