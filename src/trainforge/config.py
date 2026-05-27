"""Per-run override config for hot-swap flags.

The ``--override-model`` and ``--override-prompt`` flags on
``trainforge run`` set two state sources, in this priority:

1. **Environment variables** ``TRAINFORGE_OVERRIDE_MODEL`` and
   ``TRAINFORGE_OVERRIDE_PROMPT``. Most agent code that reads a model name
   from env (the 90% case) works without any change.
2. **ContextVars** exposed by :func:`current_overrides`. Power users whose
   agent doesn't read env can import ``from trainforge.config import
   current_overrides`` and look up the value explicitly.

Both sources reflect the same value and are set/cleared together by
:func:`override_scope`. The dual delivery is intentional — env vars are
universal, ContextVars are async-safe (asyncio tasks inherit the
context). Use whichever your agent supports.

Caveat: under ``pytest-xdist`` (process-pool parallel testing), subprocess
workers inherit env but not ContextVars. If you run scenarios under
xdist, the env path is the only one that survives.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

ENV_MODEL = "TRAINFORGE_OVERRIDE_MODEL"
ENV_PROMPT = "TRAINFORGE_OVERRIDE_PROMPT"


@dataclass(frozen=True)
class Overrides:
    """Snapshot of the active overrides for one run."""

    model: str | None = None
    prompt: str | None = None

    @property
    def is_empty(self) -> bool:
        return self.model is None and self.prompt is None


_CURRENT: ContextVar[Overrides] = ContextVar("trainforge_overrides", default=Overrides())


def current_overrides() -> Overrides:
    """Return the active :class:`Overrides` snapshot.

    Agents can call this to discover the model/prompt the runner wants
    them to use for the current scenario. Returns an empty
    :class:`Overrides` (both fields ``None``) when no overrides are
    active.
    """
    return _CURRENT.get()


@contextmanager
def override_scope(*, model: str | None = None, prompt: str | None = None) -> Iterator[Overrides]:
    """Set env vars + ContextVar for the duration of a block.

    Used by the CLI around scenario execution. On exit, env vars are
    restored to their previous values (or removed if they weren't set)
    and the ContextVar reverts.
    """
    overrides = Overrides(model=model, prompt=prompt)
    token = _CURRENT.set(overrides)

    prev_model = os.environ.get(ENV_MODEL)
    prev_prompt = os.environ.get(ENV_PROMPT)
    if model is not None:
        os.environ[ENV_MODEL] = model
    if prompt is not None:
        os.environ[ENV_PROMPT] = prompt

    try:
        yield overrides
    finally:
        # Restore env first, then ContextVar, in opposite order of setup.
        if model is not None:
            if prev_model is None:
                os.environ.pop(ENV_MODEL, None)
            else:
                os.environ[ENV_MODEL] = prev_model
        if prompt is not None:
            if prev_prompt is None:
                os.environ.pop(ENV_PROMPT, None)
            else:
                os.environ[ENV_PROMPT] = prev_prompt
        _CURRENT.reset(token)


def load_prompt_file(path: str) -> str:
    """Read a prompt file from disk. Used by the CLI to materialize
    ``--override-prompt PATH`` into the env-var / ContextVar string value.

    The file is read as UTF-8 text and trailing whitespace is preserved
    (prompts often end with deliberate newlines).
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
