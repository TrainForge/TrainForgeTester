"""trainforge.config override_scope tests: env + ContextVar set and cleanup."""
from __future__ import annotations

import os

from trainforge.config import (
    ENV_MODEL,
    ENV_PROMPT,
    current_overrides,
    override_scope,
)


def test_override_scope_sets_env_and_contextvar() -> None:
    assert ENV_MODEL not in os.environ
    assert current_overrides().model is None

    with override_scope(model="claude-sonnet-4-7"):
        assert os.environ[ENV_MODEL] == "claude-sonnet-4-7"
        assert current_overrides().model == "claude-sonnet-4-7"

    # Both cleaned up.
    assert ENV_MODEL not in os.environ
    assert current_overrides().model is None


def test_override_scope_restores_prior_env() -> None:
    os.environ[ENV_MODEL] = "preexisting"
    try:
        with override_scope(model="temp"):
            assert os.environ[ENV_MODEL] == "temp"
        # Restored to the prior value, not removed.
        assert os.environ[ENV_MODEL] == "preexisting"
    finally:
        os.environ.pop(ENV_MODEL, None)


def test_override_scope_both_model_and_prompt() -> None:
    with override_scope(model="claude", prompt="system prompt content"):
        assert os.environ[ENV_MODEL] == "claude"
        assert os.environ[ENV_PROMPT] == "system prompt content"
        snap = current_overrides()
        assert snap.model == "claude"
        assert snap.prompt == "system prompt content"
        assert snap.is_empty is False

    assert ENV_MODEL not in os.environ
    assert ENV_PROMPT not in os.environ
    assert current_overrides().is_empty


def test_override_scope_empty_is_noop() -> None:
    with override_scope():
        assert ENV_MODEL not in os.environ
        assert ENV_PROMPT not in os.environ
        assert current_overrides().is_empty
