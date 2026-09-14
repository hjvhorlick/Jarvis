"""``python -m jarvis [chat|serve]`` dispatcher."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else "chat"

    if command in {"chat", "cli"}:
        from .cli import main as chat_main

        return chat_main(argv[1:])
    if command in {"serve", "web", "server"}:
        from .server import main as serve_main

        return serve_main(argv[1:])
    if command in {"-h", "--help", "help"}:
        print(__doc__)
        print("commands:\n  chat    terminal chat (default)\n  serve   web UI + HTTP API")
        return 0

    print(f"unknown command: {command}\nusage: python -m jarvis [chat|serve]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
