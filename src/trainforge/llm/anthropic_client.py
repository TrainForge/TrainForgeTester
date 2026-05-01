"""Anthropic / Claude implementation of :class:`LLMClient`.

Keeps the surface area minimal - one ``complete(system, user)`` method.
Prompt caching (spec open-question #4) is deferred; the whole system prompt
is a plain string today and can become a cached prefix later without
changing the protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

from trainforge.llm.base import DEFAULT_MODEL


@dataclass
class AnthropicClient:
    """Thin wrapper over ``anthropic.Anthropic``.

    Imported lazily so tests and `trainforge mock-agent` / `trainforge diff`
    don't require the ``anthropic`` package to be installed at runtime.
    """

    api_key: str
    model: str = DEFAULT_MODEL
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        from anthropic import Anthropic  # type: ignore[import-not-found]

        self._client = Anthropic(api_key=self.api_key)

    def complete(self, system: str, user: str) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: list[str] = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()
