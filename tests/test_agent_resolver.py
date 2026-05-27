"""agent_resolver tests: module:fn syntax + factory pattern."""
from __future__ import annotations

import sys
import types

import pytest

from trainforge.agent_resolver import AgentResolutionError, resolve_in_process_agent


@pytest.fixture
def fake_module(request):
    name = "_trainforge_fake_agent_module"
    mod = types.ModuleType(name)

    async def run(messages):
        return {"response": "ok"}

    def make_run():
        return run

    def not_callable():
        return "not a callable"

    mod.run = run
    mod.make_run = make_run
    mod.bad_factory = not_callable
    mod.nested = types.SimpleNamespace(deep=run)
    sys.modules[name] = mod
    yield name
    sys.modules.pop(name, None)


def test_resolves_module_callable(fake_module) -> None:
    fn = resolve_in_process_agent(f"{fake_module}:run")
    assert callable(fn)


def test_resolves_dotted_attribute_path(fake_module) -> None:
    fn = resolve_in_process_agent(f"{fake_module}:nested.deep")
    assert callable(fn)


def test_factory_pattern(fake_module) -> None:
    fn = resolve_in_process_agent(f"{fake_module}:make_run()")
    assert callable(fn)


def test_factory_returning_non_callable_raises(fake_module) -> None:
    with pytest.raises(AgentResolutionError):
        resolve_in_process_agent(f"{fake_module}:bad_factory()")


def test_unknown_module_raises() -> None:
    with pytest.raises(AgentResolutionError):
        resolve_in_process_agent("trainforge_no_such_module_xyz:fn")


def test_unknown_attribute_raises(fake_module) -> None:
    with pytest.raises(AgentResolutionError):
        resolve_in_process_agent(f"{fake_module}:missing_attr")


def test_invalid_syntax_raises() -> None:
    with pytest.raises(AgentResolutionError):
        resolve_in_process_agent("not a valid spec")


def test_non_callable_attribute_raises(fake_module) -> None:
    # Add an int attribute to the fake module
    sys.modules[fake_module].an_int = 42
    with pytest.raises(AgentResolutionError):
        resolve_in_process_agent(f"{fake_module}:an_int")
