"""LLM evaluator clients.

``LLMClient`` is the protocol the rest of the runner depends on. Production
uses :class:`trainforge.llm.anthropic_client.AnthropicClient`; tests inject
their own implementation (no live network calls in CI).
"""
from trainforge.llm.base import (
    CEREBRAS_DEFAULT_MODEL,
    DEFAULT_MODEL,
    LLMClient,
    NVIDIA_DEFAULT_MODEL,
)

__all__ = [
    "LLMClient",
    "DEFAULT_MODEL",
    "NVIDIA_DEFAULT_MODEL",
    "CEREBRAS_DEFAULT_MODEL",
]
