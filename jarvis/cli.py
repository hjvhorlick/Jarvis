"""Terminal chat: ``python -m jarvis`` (or ``python -m jarvis chat``)."""

from __future__ import annotations

import argparse
import sys

from .config import load_settings
from .llm import Assistant, JarvisError
from .memory import ASSISTANT, USER, Conversation, Message

BANNER = "Jarvis — type a message, 'exit' to quit, 'reset' to clear history.\n"


def ask(assistant: Assistant, conversation: Conversation, text: str) -> int:
    """Send one message, stream the reply, persist the exchange. 0 = ok."""
    history = conversation.history() + [Message(role=USER, text=text)]
    print("jarvis: ", end="", flush=True)
    chunks: list[str] = []
    try:
        for token in assistant.stream(history):
            chunks.append(token)
            print(token, end="", flush=True)
    except JarvisError as exc:
        print(f"\nerror: {exc}")
        return 1
    print()
    conversation.add(USER, text)
    conversation.add(ASSISTANT, "".join(chunks))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="Chat with Jarvis in the terminal")
    parser.add_argument("--model", default=None, help="Override the model name")
    parser.add_argument("--no-save", action="store_true", help="Do not persist history to data/")
    parser.add_argument("-m", "--message", default=None, help="Send one message and exit")
    args = parser.parse_args(argv)

    settings = load_settings(model=args.model)
    path = None if args.no_save else settings.data_dir / "history.json"
    conversation = Conversation(path=path, max_turns=settings.max_turns)
    assistant = Assistant(settings)

    print(BANNER, end="")
    print(f"model={settings.model} provider={assistant.provider}\n", end="")

    if args.message:
        return ask(assistant, conversation, args.message)

    while True:
        try:
            line = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            return 0
        if line.lower() == "reset":
            conversation.reset()
            print("(history cleared)")
            continue
        ask(assistant, conversation, line)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
