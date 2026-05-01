"""LLM client protocol.

The runner only needs raw-string completions; parsing is handled in
:mod:`trainforge.evaluation`. This keeps the protocol minimal and trivial to
stub in tests.
"""
from __future__ import annotations

from typing import Protocol


DEFAULT_MODEL = "claude-sonnet-4-6"
"""Spec's default evaluator model (see testing-spec-v1.md §CLI Interface)."""

NVIDIA_DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
"""Default evaluator model when using NVIDIA's OpenAI-compatible API.

Picked empirically (2026-04-23) from ``/v1/models`` on NVIDIA's hosted
catalog: ~2-3s per classification call, reliable. Alternatives that worked
in the same test run (slower but fine):

    google/gemma-3-27b-it      ~6-8s per call
    google/gemma-3-4b-it       ~8s per call

Avoid ``google/gemma-4-31b-it`` - the endpoint hangs on hosted NIM.
Reasoning models (``z-ai/glm4.7``, ``qwen/qwen3-next-80b-a3b-thinking``)
work but are slow for classification unless you also override
``enable_thinking=False`` in the request."""


CEREBRAS_DEFAULT_MODEL = "zai-glm-4.7"
"""Default evaluator model when using Cerebras's OpenAI-compatible API.

Benchmarked empirically (2026-04-23): ~0.8s per classification call on
``zai-glm-4.7``, ~0.4s on ``llama3.1-8b``. Cerebras is the fastest hosted
option we've measured. Full catalog at time of writing:

    zai-glm-4.7
    qwen-3-235b-a22b-instruct-2507
    gpt-oss-120b
    llama3.1-8b

Provider-specific notes:
- Cerebras rejects NVIDIA-style ``chat_template_kwargs`` (422).
- ``reasoning_effort`` (``low`` | ``medium`` | ``high``) is supported only
  by reasoning-capable models (zai-glm-4.7, qwen-3-235b); passing it to a
  non-reasoning model returns 422. We leave ``extra_body`` unset by
  default for forward-compat across model swaps."""


class LLMClient(Protocol):
    """Minimal LLM completion interface used by the evaluator."""

    model: str

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - Protocol
        """Return the assistant's text completion given a system + user prompt."""
        ...
