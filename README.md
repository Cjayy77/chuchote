# Chuchote

Local-first, hands-free voice assistant for [Ollama](https://ollama.com). Talk
to a local LLM and get a spoken answer — wake word → speech-to-text → local
reasoning → text-to-speech, running entirely on your machine. No cloud, ever.

> **Status: Phase 2 — round-trip + memory.** Push-to-talk (hold a key, speak,
> release) → faster-whisper → Ollama → Piper → speaker, with recent turns
> remembered across the conversation and across restarts. Wake word, VAD, and
> barge-in come in later phases (see [CLAUDE.md](CLAUDE.md)).

## Pipeline

```
push-to-talk → faster-whisper (STT) → Ollama (reasoning) → Piper (TTS) → speakers ↺
```

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

## Run

Make sure Ollama is running (`ollama serve`) and a voice model is in `./voices/`:

```sh
chuchote start
# or override defaults:
chuchote start --model llama3.2 --whisper-model base.en --voice voices/en_US-lessac-medium.onnx
```

Hold **space** to talk, release to send. Chuchote transcribes, thinks, and
speaks the answer back — starting to speak as soon as the first sentence is
ready. Press **Ctrl+C** to quit.

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

Phase 1 takes settings via CLI flags and sensible defaults (see
[chuchote/config.py](chuchote/config.py)):

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `llama3.2` | Ollama model |
| `--whisper-model` | `base.en` | faster-whisper model |
| `--voice` | first `.onnx` in `./voices` | Piper voice model |
| `--ptt-key` | `space` | push-to-talk key to hold |

A proper config file lands in Phase 5.

## License

MIT.
