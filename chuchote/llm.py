"""Reasoning via a locally-running Ollama instance.

Talks to Ollama over localhost only (default http://localhost:11434) — this is
the "local" in local-first. No hosted fallback, ever.

Responses are streamed token-by-token so the loop can start speaking the first
sentence before the full answer is generated.
"""

from __future__ import annotations

from collections.abc import Iterator

from .config import Config


class Reasoner:
    def __init__(self, config: Config):
        self.config = config
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self.config.ollama_host)
        return self._client

    def check_ready(self) -> None:
        """Raise a helpful error if Ollama isn't reachable or the model is missing."""
        client = self._ensure_client()
        try:
            models = {m.model for m in client.list().models}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Can't reach Ollama at {self.config.ollama_host}. "
                "Is it running? Try `ollama serve`."
            ) from exc

        want = self.config.ollama_model
        # Ollama reports names as "llama3.2:latest"; match the bare name too.
        if want not in models and f"{want}:latest" not in models:
            raise RuntimeError(
                f"Model '{want}' not found in Ollama. "
                f"Pull it with `ollama pull {want}`."
            )

    def stream_reply(self, messages: list[dict]) -> Iterator[str]:
        """Yield response text chunks for a chat message history."""
        client = self._ensure_client()
        full = [{"role": "system", "content": self.config.system_prompt}, *messages]
        for chunk in client.chat(
            model=self.config.ollama_model,
            messages=full,
            stream=True,
        ):
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
