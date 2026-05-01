"""OpenAI-compatible Chat Completions (e.g. NVIDIA NIM / integrate.api.nvidia.com).

Defaults are tuned for speed on classification workloads:

- ``max_tokens=512`` - the evaluator prompts in :mod:`trainforge.llm.prompts`
  produce ~150 output tokens of JSON; 512 is plenty of headroom.
- ``temperature=0.0`` - classification is deterministic; high temperature
  buys us nothing and wastes tokens.
- ``timeout=60`` seconds - fail fast if the endpoint hangs (some hosted
  models on NVIDIA NIM just don't respond, e.g. ``google/gemma-4-31b-it``).
- ``extra_body=None`` by default - reasoning-model ``enable_thinking``
  mode is OPT-IN via the ``extra_body`` parameter. Enabling it by default
  murders classification latency on GLM/Qwen3 (thousands of hidden tokens
  per call).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from trainforge.llm.base import NVIDIA_DEFAULT_MODEL


@dataclass
class OpenAICompatibleClient:
    """Thin wrapper over ``openai.OpenAI`` for evaluator ``complete(system, user)``."""

    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = NVIDIA_DEFAULT_MODEL
    max_tokens: int = 512
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    extra_body: dict[str, Any] | None = None
    """Provider-specific extras (e.g. ``{"chat_template_kwargs":
    {"enable_thinking": True}}`` for reasoning models). ``None`` is the fast
    path - set this explicitly when you need thinking mode."""

    _client: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        openai_mod = import_module("openai")
        openai_cls = getattr(openai_mod, "OpenAI")

        self._client = openai_cls(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
        )

    def complete(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message
        text = getattr(message, "content", None) or ""
        return str(text).strip()
