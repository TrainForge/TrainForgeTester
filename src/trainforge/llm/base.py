"""LLM client protocol.

The runner only needs raw-string completions; parsing is handled in
:mod:`trainforge.evaluation`. This keeps the protocol minimal and trivial to
stub in tests.
"""
from __future__ import annotations

from typing import Protocol


OPENAI_COMPAT_DEFAULT_MODEL = "gpt-4o-mini"
"""Default evaluator model for any OpenAI-compatible endpoint."""


class LLMClient(Protocol):
    """Minimal LLM completion interface used by the evaluator."""

    model: str

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - Protocol
        """Return the assistant's text completion given a system + user prompt."""
        ...
