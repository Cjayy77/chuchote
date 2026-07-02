"""Command-line entry point: `chuchote start`."""

from __future__ import annotations

import argparse

from . import __version__
from .config import Config


def _cmd_start(args: argparse.Namespace) -> None:
    # Precedence: defaults < config file < command-line flags.
    config = Config.load(args.config)
    if args.model:
        config.ollama_model = args.model
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    if args.language:
        config.language = args.language
    if args.voice:
        config.piper_voice = args.voice
    if args.ptt_key:
        config.ptt_key = args.ptt_key
    if args.ptt:
        config.mode = "ptt"
    if args.wake_word:
        config.wake_model = args.wake_word
    if args.wake_threshold is not None:
        config.wake_threshold = args.wake_threshold
    if args.no_chime:
        config.wake_chime = False
    if args.barge_in:
        config.barge_in_mode = args.barge_in
    if args.no_banner:
        config.banner = False

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


def _cmd_doctor(args: argparse.Namespace) -> None:
    config = Config.load(args.config)
    from .doctor import run

    raise SystemExit(0 if run(config) else 1)


def _cmd_init(args: argparse.Namespace) -> None:
    import os

    from .config import default_config_path, write_default_config

    path = args.path or default_config_path()
    if os.path.exists(path) and not args.force:
        print(f"Config already exists at {path}\nUse --force to overwrite.")
        return
    write_default_config(path)
    print(f"Wrote default config to {path}\nEdit it, or override any setting with a flag.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chuchote",
        description="Local-first, hands-free voice assistant for Ollama.",
    )
    parser.add_argument("--version", action="version", version=f"chuchote {__version__}")
    sub = parser.add_subparsers(dest="command")

    start = sub.add_parser("start", help="Start the voice assistant loop.")
    start.add_argument("--config", help="Path to a config file (default: per-user config dir).")
    start.add_argument("--model", help="Ollama model to use (e.g. llama3.2).")
    start.add_argument("--whisper-model", help="faster-whisper model (e.g. small, small.en).")
    start.add_argument(
        "--language",
        help="Recognition language code (e.g. en, fr, de, zh) or 'auto'.",
    )
    start.add_argument("--voice", help="Path to a Piper .onnx voice model.")
    start.add_argument(
        "--ptt",
        action="store_true",
        help="Use push-to-talk instead of the wake word (hold --ptt-key).",
    )
    start.add_argument("--ptt-key", help="Push-to-talk key to hold (default: space).")
    start.add_argument(
        "--wake-word",
        help="Wake word model (e.g. hey_jarvis, alexa, hey_mycroft).",
    )
    start.add_argument(
        "--wake-threshold",
        type=float,
        help="Wake word sensitivity 0..1 (higher = fewer false triggers).",
    )
    start.add_argument(
        "--no-chime",
        action="store_true",
        help="Disable the acknowledgement tone after the wake word.",
    )
    start.add_argument(
        "--barge-in",
        choices=["wake", "vad", "off"],
        help=(
            "How to interrupt a reply: 'wake' (say the wake word again; default, "
            "works with speakers), 'vad' (any speech; headphones only), 'off'."
        ),
    )
    start.add_argument(
        "--no-banner",
        action="store_true",
        help="Don't print the startup banner.",
    )
    start.add_argument(
        "--forget",
        action="store_true",
        help="Clear conversation memory before starting.",
    )
    start.set_defaults(func=_cmd_start)

    forget = sub.add_parser("forget", help="Erase all saved conversation memory.")
    forget.set_defaults(func=_cmd_forget)

    doctor = sub.add_parser(
        "doctor", help="Check Ollama, voice, and audio devices are ready."
    )
    doctor.add_argument("--config", help="Path to a config file (default: per-user config dir).")
    doctor.set_defaults(func=_cmd_doctor)

    init = sub.add_parser("init", help="Write a default config file you can edit.")
    init.add_argument("--path", help="Where to write it (default: per-user config dir).")
    init.add_argument(
        "--force", action="store_true", help="Overwrite an existing config file."
    )
    init.set_defaults(func=_cmd_init)

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
