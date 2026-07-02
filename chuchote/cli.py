"""Command-line entry point: `chuchote start`."""

from __future__ import annotations

import argparse

from . import __version__
from .config import Config


def _cmd_start(args: argparse.Namespace) -> None:
    config = Config()
    if args.model:
        config.ollama_model = args.model
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    if args.voice:
        config.piper_voice = args.voice
    if args.ptt_key:
        config.ptt_key = args.ptt_key

    if args.forget:
        # Start fresh: wipe persisted history before the session.
        from .memory import Memory

        Memory(config).clear()
        print("[memory cleared]")

    # Imported lazily so `chuchote --help` / `--version` stay instant and don't
    # need the heavy audio/ML deps present.
    from .loop import Assistant

    Assistant(config).run()


def _cmd_forget(args: argparse.Namespace) -> None:
    from .memory import Memory

    Memory(Config()).clear()
    print("Conversation memory cleared.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chuchote",
        description="Local-first, hands-free voice assistant for Ollama.",
    )
    parser.add_argument("--version", action="version", version=f"chuchote {__version__}")
    sub = parser.add_subparsers(dest="command")

    start = sub.add_parser("start", help="Start the voice assistant loop.")
    start.add_argument("--model", help="Ollama model to use (e.g. llama3.2).")
    start.add_argument("--whisper-model", help="faster-whisper model (e.g. base.en).")
    start.add_argument("--voice", help="Path to a Piper .onnx voice model.")
    start.add_argument("--ptt-key", help="Push-to-talk key to hold (default: space).")
    start.add_argument(
        "--forget",
        action="store_true",
        help="Clear conversation memory before starting.",
    )
    start.set_defaults(func=_cmd_start)

    forget = sub.add_parser("forget", help="Erase all saved conversation memory.")
    forget.set_defaults(func=_cmd_forget)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
