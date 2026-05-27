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


class MissingLLMCredentialsError(RuntimeError):
    """Raised by :class:`LazyMissingLLMClient` when a scenario actually
    needs a judge call but the user never configured one. The message
    tells the user exactly which flags / env vars to set.
    """


class LazyMissingLLMClient:
    """LLM client stub used when no credentials are configured.

    Construction always succeeds so deterministic scenarios (no
    ``may_diverge``, no per-turn custom checks, no ``outcome_checks``)
    run without needing an LLM at all. The error is deferred to the
    first actual ``.complete()`` call.

    Used by both the CLI (``trainforge run`` with no LLM flags) and the
    pytest plugin (``trainforge[pytest]`` fixture when the user hasn't
    set ``OPENAI_API_KEY`` / ``OPENAI_API_URL``).
    """

    model: str

    def __init__(self, model: str = OPENAI_COMPAT_DEFAULT_MODEL) -> None:
        self.model = model

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - error path
        raise MissingLLMCredentialsError(
            "this scenario needs an LLM judge but none is configured. "
            "Pass --llm-api-key + --llm-api-url, or set $OPENAI_API_KEY + "
            "$OPENAI_API_URL. Scenarios with may_diverge=True, custom turn "
            "checks, or outcome_checks all trigger the judge."
        )
