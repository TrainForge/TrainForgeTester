"""Toy agent used by the README quickstart and the smoke test.

A real agent (LangChain, CrewAI, etc.) is more involved, but the contract
TrainForge sees is the same: an async function that takes a list of
messages and returns a dict with ``response`` and optionally
``tool_calls``.

This particular agent honors two TrainForge hot-swap env vars so the
``--override-prompt`` / ``--override-model`` flags do something visible
in the quickstart demo.
"""
from __future__ import annotations

import os
from typing import Any

DEFAULT_PROMPT = "You are a friendly concierge. Greet the user."
DEFAULT_MODEL = "echo-1"


async def run(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """One-turn echo agent: greets, then echoes whatever the user said.

    The agent doesn't actually call an LLM; it just demonstrates the
    in-process contract TrainForge expects. Replace the body of this
    function with a real LangChain / CrewAI / OpenAI Agents SDK call to
    test a real agent.

    Note: env vars are read **inside the function**, not at module load
    time. This matters for ``--override-model`` / ``--override-prompt``:
    if your agent caches the env at import, the override won't take
    effect on the second run because the module is already loaded.
    """
    prompt = os.environ.get("TRAINFORGE_OVERRIDE_PROMPT", DEFAULT_PROMPT)
    model = os.environ.get("TRAINFORGE_OVERRIDE_MODEL", DEFAULT_MODEL)

    last_user = next(
        (m for m in reversed(messages) if m.get("role") == "user"),
        None,
    )
    user_text = (last_user or {}).get("content", "") or ""

    reply = f"[model={model}] Hello! I heard: {user_text!r}. ({prompt})"
    return {"response": reply}
