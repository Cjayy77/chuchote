#   <img src="doaBanner.png" width="45%" /> Chuchote

[![CI](https://github.com/Cjayy77/chuchote/actions/workflows/ci.yml/badge.svg)](https://github.com/Cjayy77/chuchote/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chuchote)](https://pypi.org/project/chuchote/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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
pip install chuchote
```

Then grab a [Piper voice](https://github.com/rhasspy/piper/blob/master/VOICES.md)
(`.onnx` + `.onnx.json`) into a `voices/` folder next to where you run it (or
set `piper_voice` in your config), and check everything's wired up:

```sh
chuchote doctor
```

### From source (development)

```sh
git clone https://github.com/Cjayy77/chuchote.git
cd chuchote
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

### Custom wake words

`--wake-word` accepts a **file path** as well as a built-in name, so you can
use any openWakeWord-compatible model:

```sh
chuchote start --wake-word path/to/hey_chuchote.onnx
```

To create one for your own phrase, use openWakeWord's automatic training
notebook — see [Training New Models](https://github.com/dscripka/openWakeWord#training-new-models)
in their README. It generates synthetic speech for your phrase and trains a
model in roughly an hour, no voice recordings needed; download the resulting
`.onnx` and point `--wake-word` (or `wake_model` in your config) at it.

> Note: the training notebook runs on Google Colab (a free cloud notebook) —
> that's one-time, dev-machine tooling, like downloading a Piper voice. The
> resulting model runs fully locally; nothing about the assistant touches the
> network.

### Push-to-talk fallback

Prefer holding a key? Skip the wake word entirely:

```sh
chuchote start --ptt            # hold space to talk, release to send
chuchote start --ptt --ptt-key ctrl
```

## Checking your setup

Not sure everything's wired up? Run the preflight check:

```sh
chuchote doctor
```

It verifies Ollama is reachable and the model is pulled, a Piper voice is
present, your mic and speakers are detected, and (in wake mode) the wake-word
deps are installed — reporting each as `[ ok ]` / `[fail]` and exiting non-zero
if anything's wrong.

## Languages

Chuchote isn't English-only — whisper understands ~99 languages and Piper has
voices for dozens. To run it in another language, three things need to line up:

1. **Recognition** — set `language` and use a **multilingual** whisper model
   (the plain names, *not* the `.en` ones — `base` on low-RAM machines,
   `small` for better accuracy):
   ```toml
   language = "fr"
   whisper_model = "base"
   ```
2. **Speech** — download a Piper voice for that language from
   [VOICES.md](https://github.com/rhasspy/piper/blob/master/VOICES.md) into your
   voices dir (or point `piper_voice` at it). Piper voices are one language each.
3. **Reasoning** — pick an Ollama model that's good in your language (most
   modern ones are multilingual; e.g. `qwen2.5` is strong for Chinese). Chuchote
   already asks the model to reply in whatever language you speak.

Then `chuchote doctor` will confirm the pieces match. Common starting points:

| Language | `language` | Example Piper voice |
|---|---|---|
| French | `fr` | `fr_FR-siwis-medium` |
| German | `de` | `de_DE-thorsten-medium` |
| Spanish | `es` | `es_ES-davefx-medium` |
| Italian | `it` | `it_IT-paola-medium` |
| Portuguese (BR) | `pt` | `pt_BR-faber-medium` |
| Dutch | `nl` | `nl_NL-mls-medium` |
| Chinese | `zh` | `zh_CN-huayan-medium` |
| Russian | `ru` | `ru_RU-dmitri-medium` |

**Adding any other language:** find its voice on
[VOICES.md](https://github.com/rhasspy/piper/blob/master/VOICES.md), set
`language` to whisper's [code](https://github.com/openai/whisper#available-models-and-languages)
for it, keep a multilingual `whisper_model`, and you're set. The wake word stays
an English phrase (openWakeWord's built-ins are English) — or use `--ptt`.

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
| `--whisper-model` | `base.en` | faster-whisper model (`small.en` = more accurate, needs ~1 GB free RAM) |
| `--language` | `auto` | recognition language (`en`, `fr`, `de`, `zh`, …); needs a multilingual model |
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

## Development

Run the test suite (covers the pure logic — sentence chunking, memory, config
precedence, banner styling — with no audio/model deps needed):

```sh
pip install -e ".[dev]"
pytest
```

Tests also run in CI (GitHub Actions) on Linux and Windows, Python 3.10–3.13.

### Releasing

Publishing a GitHub release triggers the `publish.yml` workflow, which builds
and uploads to PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/)
— no API token. One-time setup on pypi.org: add a *trusted publisher* for this
repo (owner `Cjayy77`, repo `chuchote`, workflow `publish.yml`, environment
`pypi`), and create a matching `pypi` environment in the repo's GitHub settings.

## License

MIT.
