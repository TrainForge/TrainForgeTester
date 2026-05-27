"""HttpTransport edge cases not covered by test_transport_http.py:

- 3xx responses (httpx.AsyncClient does NOT auto-follow redirects).
- Custom timeout actually propagated to httpx.
- Header content-type is set on the request.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from trainforge.errors import AgentError
from trainforge.transport import HttpTransport, Message, Role

AGENT_URL = "http://agent.test/chat"


def _user(text: str) -> Message:
    return Message(role=Role.USER, content=text)


@respx.mock
async def test_301_response_raises_agent_error() -> None:
    """3xx is non-2xx; HttpTransport must reject (was previously leaking
    through to the JSON-parse path with a misleading error)."""
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(301, headers={"Location": "http://elsewhere"})
    )
    with pytest.raises(AgentError) as exc:
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])
    assert "HTTP 301" in str(exc.value)


@respx.mock
async def test_302_response_raises_agent_error() -> None:
    respx.post(AGENT_URL).mock(return_value=httpx.Response(302))
    with pytest.raises(AgentError) as exc:
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])
    assert "HTTP 302" in str(exc.value)


@respx.mock
async def test_204_no_content_raises_agent_error_on_empty_body() -> None:
    """204 is 2xx but has no body; HttpTransport should fail JSON parsing
    on the empty body with a clear message."""
    respx.post(AGENT_URL).mock(return_value=httpx.Response(204))
    with pytest.raises(AgentError):
        await HttpTransport(url=AGENT_URL).chat([_user("hi")])


@respx.mock
async def test_request_sets_json_content_type() -> None:
    route = respx.post(AGENT_URL).mock(
        return_value=httpx.Response(200, json={"response": "ok"})
    )
    await HttpTransport(url=AGENT_URL).chat([_user("hi")])
    sent = route.calls[0].request
    assert sent.headers["content-type"] == "application/json"


@respx.mock
async def test_custom_timeout_is_propagated() -> None:
    """The transport must use the configured timeout, not httpx's default
    (which is 5s, too short for some local LLM endpoints)."""
    transport = HttpTransport(url=AGENT_URL, timeout_seconds=12.5)
    respx.post(AGENT_URL).mock(
        return_value=httpx.Response(200, json={"response": "ok"})
    )
    # Smoke: should not raise on a slow-ish timeout.
    reply = await transport.chat([_user("hi")])
    assert reply.text == "ok"
