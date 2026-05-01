"""LLM evaluator clients.

``LLMClient`` is the protocol the rest of the runner depends on. Production
uses an OpenAI-compatible endpoint via
:class:`trainforge.llm.openai_compatible_client.OpenAICompatibleClient`;
tests inject their own implementation (no live network calls in CI).
"""
from trainforge.llm.base import LLMClient, OPENAI_COMPAT_DEFAULT_MODEL

__all__ = [
    "LLMClient",
    "OPENAI_COMPAT_DEFAULT_MODEL",
]
