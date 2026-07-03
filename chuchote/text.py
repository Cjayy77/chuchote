"""Pure text helpers (no audio/ML deps, so they're cheap to import and test)."""

from __future__ import annotations

import re

# Flush a chunk to TTS as soon as a sentence-ending punctuation is seen, so we
# start speaking before Ollama finishes generating.
_SENTENCE_END = re.compile(r".*?[.!?…\n]+", re.DOTALL)


def explain_failure(exc: Exception, ollama_model: str) -> str:
    """Turn a turn-level exception into a message with actionable advice.

    Out-of-memory failures (whisper's mkl_malloc, Ollama's ggml buffer
    allocation) are common on 8 GB machines and their raw messages are
    opaque — translate them.
    """
    msg = str(exc)
    lowered = msg.lower()
    if "alloc" in lowered or "out of memory" in lowered:
        return (
            f"{msg}\n[hint: not enough free RAM to load the model. Close some "
            f"apps, or try a smaller Ollama model than '{ollama_model}' "
            "(e.g. `ollama pull llama3.2:1b`). `chuchote doctor` shows free RAM.]"
        )
    return msg


def drain_sentences(buffer: str) -> tuple[list[str], str]:
    """Split completed sentences out of `buffer`, keep the trailing remainder."""
    sentences = []
    while True:
        match = _SENTENCE_END.match(buffer)
        if not match:
            break
        sentence = match.group(0).strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[match.end():]
    return sentences, buffer
