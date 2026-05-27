"""InProcessTransport tests.

The transport calls the user's Python callable directly. Async and sync
callables both supported; sync are wrapped via ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio

import pytest

from trainforge.errors import AgentError, AgentTimeoutError
from trainforge.transport import InProcessTransport, Message, Role


def _user(text: str) -> Message:
    return Message(role=Role.USER, content=text)


async def test_async_callable_returns_text() -> None:
    async def agent(messages):
        return {"response": "hello"}

    reply = await InProcessTransport(agent=agent).chat([_user("hi")])
    assert reply.text == "hello"
    assert reply.tool_calls == []


async def test_sync_callable_wrapped_via_to_thread() -> None:
    def agent(messages):
        return {"response": "from sync"}

    reply = await InProcessTransport(agent=agent).chat([_user("hi")])
    assert reply.text == "from sync"


async def test_tool_call_reply_parsed() -> None:
    async def agent(messages):
        return {
            "response": "",
            "tool_calls": [
                {"id": "c1", "name": "lookup", "arguments": {"q": "x"}},
            ],
        }

    reply = await InProcessTransport(agent=agent).chat([_user("hi")])
    assert reply.is_tool_round
    assert reply.tool_calls[0].name == "lookup"


async def test_user_exception_maps_to_agent_error() -> None:
    async def agent(messages):
        raise KeyError("missing")

    with pytest.raises(AgentError):
        await InProcessTransport(agent=agent).chat([_user("hi")])


async def test_wrong_shape_maps_to_agent_error() -> None:
    async def agent(messages):
        return "not a dict"

    with pytest.raises(AgentError):
        await InProcessTransport(agent=agent).chat([_user("hi")])


async def test_missing_both_response_and_tool_calls_raises() -> None:
    async def agent(messages):
        return {}

    with pytest.raises(AgentError):
        await InProcessTransport(agent=agent).chat([_user("hi")])


async def test_slow_agent_hits_timeout() -> None:
    async def agent(messages):
        await asyncio.sleep(0.5)
        return {"response": "too late"}

    with pytest.raises(AgentTimeoutError):
        await InProcessTransport(agent=agent, timeout_seconds=0.05).chat([_user("hi")])


async def test_factory_pattern_callable_of_callable() -> None:
    """A factory returns a callable that is then used as the agent.

    This is how :func:`trainforge.agent_resolver.resolve_in_process_agent`
    invokes ``module:make_fn()`` — InProcessTransport doesn't see the
    factory layer; by the time it's constructed, ``agent`` is the
    already-resolved callable.
    """
    def make_agent():
        async def fn(messages):
            return {"response": "from factory"}

        return fn

    agent = make_agent()
    reply = await InProcessTransport(agent=agent).chat([_user("hi")])
    assert reply.text == "from factory"


async def test_defensive_message_copy() -> None:
    """The user's callable mutating its messages arg must not corrupt the runner."""
    sent = [_user("hi")]

    async def agent(messages):
        messages.append({"role": "agent", "content": "INJECTED"})
        return {"response": "ok"}

    await InProcessTransport(agent=agent).chat(sent)
    # Original list unchanged.
    assert len(sent) == 1
