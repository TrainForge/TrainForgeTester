"""HttpTransport tests, honoring the spec's error-handling table.

Uses ``respx`` to mock the ``httpx.AsyncClient`` requests. The contract
the runner sees is identical to the previous ``AgentClient`` contract;
only the wire-level client changed.
"""
from __future__ import annotations

import json as _json

import httpx
import pytest
import respx

from trainforge.errors import AgentError, AgentTimeoutError, AgentUnreachableError
from trainforge.transport import HttpTransport, Message, Role

AGENT_URL = "http://agent.test/chat"


def _user(content: str) -> Message:
    return Message(role=Role.USER, content=content)


@respx.mock
async def test_happy_path_returns_text_reply() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(200, json={"response": "hi there"})
    )
    reply = await HttpTransport(url=AGENT_URL).chat([_user("hello")])
    assert reply.text == "hi there"
    assert reply.tool_calls == []
    assert reply.is_tool_round is False


@respx.mock
async def test_tool_call_reply_parsed() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "response": "",
                "tool_calls": [
                    {"id": "call_42", "name": "check_weather", "arguments": {"when": "tonight"}},
                    {"name": "check_availability", "arguments": {"party_size": 2}},
                ],
            },
        )
    )
    reply = await HttpTransport(url=AGENT_URL).chat([_user("hi")])
    assert reply.is_tool_round is True
    assert len(reply.tool_calls) == 2
    assert reply.tool_calls[0].name == "check_weather"
    assert reply.tool_calls[0].arguments == {"when": "tonight"}
    assert reply.tool_calls[0].id == "call_42"
    assert reply.tool_calls[1].id is None


@respx.mock
async def test_reply_with_text_and_tool_calls_is_tool_round() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "response": "Let me check a couple of things",
                "tool_calls": [{"name": "check_weather", "arguments": {}}],
            },
        )
    )
    reply = await HttpTransport(url=AGENT_URL).chat([_user("hi")])
    assert reply.is_tool_round is True
    assert reply.text == "Let me check a couple of things"


@respx.mock
async def test_empty_reply_raises_agent_error() -> None:
    respx.post(AGENT_URL).mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_tool_calls_not_a_list_raises() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(200, json={"tool_calls": "oops"})
    )
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_tool_call_without_name_raises() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(200, json={"tool_calls": [{"arguments": {}}]})
    )
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_non_2xx_raises_agent_error() -> None:
    respx.post(AGENT_URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_non_json_body_raises_agent_error() -> None:
    respx.post(AGENT_URL).mock(return_value=httpx.Response(200, text="not json"))
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_timeout_retries_once_then_raises() -> None:
    route = respx.post(AGENT_URL).mock(side_effect=httpx.TimeoutException("slow"))
    with pytest.raises(AgentTimeoutError):
        await HttpTransport(url=AGENT_URL, timeout_seconds=0.1).chat([_user("hi")])
    assert route.call_count == 2  # one initial + one retry


@respx.mock
async def test_timeout_then_success_returns_reply() -> None:
    respx.post(AGENT_URL).mock(
        side_effect=[
            httpx.TimeoutException("slow"),
            httpx.Response(200, json={"response": "on retry"}),
        ]
    )
    reply = await HttpTransport(url=AGENT_URL, timeout_seconds=0.1).chat([_user("hi")])
    assert reply.text == "on retry"


@respx.mock
async def test_connection_error_raises_unreachable() -> None:
    respx.post(AGENT_URL).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(AgentUnreachableError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_sends_full_history() -> None:
    route = respx.post(AGENT_URL).mock(
        return_value=httpx.Response(200, json={"response": "ok"})
    )
    history: list[Message] = [
        _user("book a table"),
        Message(role=Role.AGENT, content="for how many?"),
        _user("2 please"),
    ]
    await HttpTransport(url=AGENT_URL).chat(history)
    sent = _json.loads(route.calls[0].request.content)
    assert sent == {"messages": history}


@respx.mock
async def test_agent_reply_is_empty_property() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(200, json={"response": "   "})
    )
    reply = await HttpTransport(url=AGENT_URL).chat([_user("hello")])
    assert reply.is_empty is True


@respx.mock
async def test_timeout_then_connection_error_is_unreachable() -> None:
    respx.post(AGENT_URL).mock(
        side_effect=[httpx.TimeoutException("slow"), httpx.ConnectError("boom")]
    )
    with pytest.raises(AgentUnreachableError):
        await HttpTransport(url=AGENT_URL, timeout_seconds=0.1).chat([_user("hi")])


@respx.mock
async def test_response_must_be_object() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(200, json=["not", "an", "object"])
    )
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_response_field_must_be_string() -> None:
    respx.post(AGENT_URL).mock(return_value=httpx.Response(200, json={"response": 123}))
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_tool_call_item_must_be_object() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(200, json={"tool_calls": ["oops"]})
    )
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_tool_call_arguments_must_be_object() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(
            200, json={"tool_calls": [{"name": "t", "arguments": "nope"}]}
        )
    )
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_tool_call_id_must_be_string_if_present() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(
            200, json={"tool_calls": [{"id": 42, "name": "t", "arguments": {}}]}
        )
    )
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_tool_call_arguments_none_defaults_to_empty_dict() -> None:
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(
            200, json={"tool_calls": [{"name": "t", "arguments": None}]}
        )
    )
    reply = await HttpTransport(url=AGENT_URL).chat([_user("hi")])
    assert reply.tool_calls[0].arguments == {}
