"""Pure text helpers (no audio/ML deps, so they're cheap to import and test)."""

from __future__ import annotations

import re

# Flush a chunk to TTS as soon as a sentence-ending punctuation is seen, so we
# start speaking before Ollama finishes generating.
_SENTENCE_END = re.compile(r".*?[.!?…\n]+", re.DOTALL)


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
