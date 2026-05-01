"""Agent HTTP client tests, honoring the spec's error-handling table."""
from __future__ import annotations

import json as _json

import pytest
import requests
import responses

from trainforge.agent_client import AgentClient, Message
from trainforge.errors import AgentError, AgentTimeoutError, AgentUnreachableError


AGENT_URL = "http://agent.test/chat"


def _customer(content: str) -> Message:
    return Message(role="customer", content=content)


@responses.activate
def test_happy_path_returns_text_reply() -> None:
    responses.post(AGENT_URL, json={"response": "hi there"}, status=200)

    reply = AgentClient(url=AGENT_URL).chat([_customer("hello")])
    assert reply.text == "hi there"
    assert reply.tool_calls == []
    assert reply.is_tool_round is False
    assert len(responses.calls) == 1


@responses.activate
def test_tool_call_reply_parsed() -> None:
    responses.post(
        AGENT_URL,
        json={
            "response": "",
            "tool_calls": [
                {"id": "call_42", "name": "check_weather", "arguments": {"when": "tonight"}},
                {"name": "check_availability", "arguments": {"party_size": 2}},
            ],
        },
        status=200,
    )
    reply = AgentClient(url=AGENT_URL).chat([_customer("hi")])
    assert reply.is_tool_round is True
    assert len(reply.tool_calls) == 2
    assert reply.tool_calls[0].name == "check_weather"
    assert reply.tool_calls[0].arguments == {"when": "tonight"}
    assert reply.tool_calls[0].id == "call_42"
    assert reply.tool_calls[1].id is None  # optional field


@responses.activate
def test_reply_with_text_and_tool_calls_is_tool_round() -> None:
    responses.post(
        AGENT_URL,
        json={
            "response": "Let me check a couple of things",
            "tool_calls": [{"name": "check_weather", "arguments": {}}],
        },
        status=200,
    )
    reply = AgentClient(url=AGENT_URL).chat([_customer("hi")])
    assert reply.is_tool_round is True
    assert reply.text == "Let me check a couple of things"


@responses.activate
def test_empty_reply_raises_agent_error() -> None:
    responses.post(AGENT_URL, json={}, status=200)
    with pytest.raises(AgentError):
        AgentClient(url=AGENT_URL).chat([_customer("hi")])


@responses.activate
def test_tool_calls_not_a_list_raises() -> None:
    responses.post(AGENT_URL, json={"tool_calls": "oops"}, status=200)
    with pytest.raises(AgentError):
        AgentClient(url=AGENT_URL).chat([_customer("hi")])


@responses.activate
def test_tool_call_without_name_raises() -> None:
    responses.post(
        AGENT_URL,
        json={"tool_calls": [{"arguments": {}}]},
        status=200,
    )
    with pytest.raises(AgentError):
        AgentClient(url=AGENT_URL).chat([_customer("hi")])


@responses.activate
def test_non_2xx_raises_agent_error() -> None:
    responses.post(AGENT_URL, json={"error": "boom"}, status=500)
    with pytest.raises(AgentError):
        AgentClient(url=AGENT_URL).chat([_customer("hi")])


@responses.activate
def test_non_json_body_raises_agent_error() -> None:
    responses.post(AGENT_URL, body="not json", status=200)
    with pytest.raises(AgentError):
        AgentClient(url=AGENT_URL).chat([_customer("hi")])


@responses.activate
def test_timeout_retries_once_then_raises() -> None:
    responses.post(AGENT_URL, body=requests.exceptions.Timeout("slow"))
    responses.post(AGENT_URL, body=requests.exceptions.Timeout("slow again"))
    with pytest.raises(AgentTimeoutError):
        AgentClient(url=AGENT_URL, timeout_seconds=0.1).chat([_customer("hi")])
    assert len(responses.calls) == 2


@responses.activate
def test_timeout_then_success_returns_reply() -> None:
    responses.post(AGENT_URL, body=requests.exceptions.Timeout("slow"))
    responses.post(AGENT_URL, json={"response": "on retry"}, status=200)
    reply = AgentClient(url=AGENT_URL, timeout_seconds=0.1).chat([_customer("hi")])
    assert reply.text == "on retry"


@responses.activate
def test_connection_error_raises_unreachable() -> None:
    responses.post(AGENT_URL, body=requests.exceptions.ConnectionError("boom"))
    with pytest.raises(AgentUnreachableError):
        AgentClient(url=AGENT_URL).chat([_customer("hi")])


@responses.activate
def test_sends_full_history() -> None:
    responses.post(AGENT_URL, json={"response": "ok"}, status=200)
    history: list[Message] = [
        _customer("book a table"),
        Message(role="agent", content="for how many?"),
        _customer("2 please"),
    ]
    AgentClient(url=AGENT_URL).chat(history)
    call = responses.calls[0]
    body = _json.loads(call.request.body)
    assert body == {"messages": history}
