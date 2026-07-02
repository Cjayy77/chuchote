# Chuchote

Local-first, hands-free voice assistant for [Ollama](https://ollama.com). Talk
to a local LLM and get a spoken answer — wake word → speech-to-text → local
reasoning → text-to-speech, running entirely on your machine. No cloud, ever.

> **Status: Phase 5 — the CLI MVP is complete.** Wake word → speech capture
> (Silero VAD end-of-turn) → faster-whisper → Ollama → Piper → speaker, with
> memory across restarts, barge-in to interrupt a reply, overlapped synthesis,
> a config file, and an install script. Push-to-talk remains a fallback
> (`--ptt`). See [CLAUDE.md](CLAUDE.md) for what's next.

## Pipeline

```
wake word → capture (Silero VAD end-of-turn) → faster-whisper (STT)
          → Ollama (reasoning) → Piper (TTS) → speakers  ↺
```

The wake word (openWakeWord) and VAD (Silero) both run locally on onnxruntime.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with a model pulled
  (e.g. `ollama pull llama3.2`)
- A [Piper voice](https://github.com/rhasspy/piper/blob/master/VOICES.md)
  (`.onnx` + `.onnx.json`) — drop it in `./voices/`
- A working microphone and speakers

## Install

```sh
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e .
```

Or use the install script, which also downloads a default Piper voice and
writes a config file:

```sh
./scripts/install.ps1     # Windows (PowerShell)
./scripts/install.sh      # macOS/Linux
```

## Run

Make sure Ollama is running (`ollama serve`) and a voice model is in `./voices/`:

```sh
chuchote start
# or override defaults:
chuchote start --model llama3.2 --wake-word hey_jarvis --voice voices/en_US-lessac-medium.onnx
```

By default Chuchote is **always listening** for the wake word (`hey_jarvis`).
Say it, then speak your request — Silero VAD detects when you've stopped and
the reply is transcribed, thought out, and spoken back (starting as soon as the
first sentence is ready). Press **Ctrl+C** to quit.

The wake-word models download automatically on first run. Other built-in words
include `alexa`, `hey_mycroft`, and `hey_rhasspy` (`--wake-word`); raise
`--wake-threshold` if you get false triggers. A short tone confirms the wake
word — silence it with `--no-chime`.

**Barge-in.** While Chuchote is speaking, say the wake word again to cut it off
and start a new turn (`--barge-in wake`, the default — echo-robust, works with
open speakers). With headphones you can use `--barge-in vad` so *any* speech
interrupts; `--barge-in off` lets every reply finish. In push-to-talk mode,
pressing the PTT key cuts off the reply (then hold it to speak again).

### Push-to-talk fallback

Prefer holding a key? Skip the wake word entirely:

```sh
chuchote start --ptt            # hold space to talk, release to send
chuchote start --ptt --ptt-key ctrl
```

## Memory

Chuchote remembers the conversation. Each exchange is stored in a SQLite
database (`memory.db` in your per-user data dir) and the most recent turns are
fed back into the model's context every turn, so it stays coherent across turns
and restarts.

```sh
chuchote start --forget   # start a session with a clean slate
chuchote forget           # erase all saved memory
```

Tune how much history is injected via `history_messages` in
[chuchote/config.py](chuchote/config.py).

## Configuration

Persist your settings in a config file instead of passing flags every time:

```sh
chuchote init              # writes a commented config.toml to your config dir
```

Edit the file (its path is printed by `init`; typically `%APPDATA%\chuchote\`
on Windows or `~/.config/chuchote/` elsewhere), uncomment what you want to
change, and it's picked up on the next `chuchote start`. Point at a specific
file with `--config PATH`. Precedence is **defaults < config file < flags**.

Flags (over sensible defaults; see [chuchote/config.py](chuchote/config.py) for
the full list and VAD tunables):

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `llama3.2` | Ollama model |
| `--whisper-model` | `base.en` | faster-whisper model |
| `--voice` | first `.onnx` in `./voices` | Piper voice model |
| `--wake-word` | `hey_jarvis` | wake word model (`alexa`, `hey_mycroft`, `hey_rhasspy`) |
| `--wake-threshold` | `0.5` | wake sensitivity 0..1 (higher = fewer false triggers) |
| `--no-chime` | off | disable the wake-word acknowledgement tone |
| `--barge-in` | `wake` | interrupt a reply: `wake` / `vad` (headphones) / `off` |
| `--ptt` | off | use push-to-talk instead of the wake word |
| `--ptt-key` | `space` | push-to-talk key to hold (with `--ptt`) |
| `--forget` | off | clear conversation memory before starting |
| `--no-banner` | off | don't print the startup banner |
| `--config` | per-user config dir | path to a config file |

## License

MIT.
