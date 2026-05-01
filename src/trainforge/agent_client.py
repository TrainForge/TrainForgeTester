"""HTTP client for the agent-under-test.

Contract (testing-spec-v1.md section "Agent API Contract", extended in v1.1
for tool_calls using OpenAI-style function-calling shape):

    POST {agent-url}
    Request  : {"messages": [Message, ...]}
    Response : {"response": "optional text", "tool_calls": [ToolCall, ...]}
               (either `response` or `tool_calls` may be absent, but at
                least one must carry signal)

Message variants (order preserved in ``messages``):

- customer:     {"role": "customer", "content": str}
- agent text:   {"role": "agent",    "content": str}
- agent tool:   {"role": "agent",    "content": str | None,
                  "tool_calls": [{"id": str, "name": str, "arguments": dict}, ...]}
- tool result:  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

ToolCall shape (in both requests and responses):
    {"id": str, "name": str, "arguments": dict}

Error handling follows the spec's "Error Handling" table:
- Agent returns non-2xx      -> :class:`AgentError`
- Agent times out            -> retry once; on second timeout, :class:`AgentTimeoutError`
- Agent unreachable          -> :class:`AgentUnreachableError`
- Agent body missing both    -> :class:`AgentError` (no text, no tool_calls)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import requests

from trainforge.errors import AgentError, AgentTimeoutError, AgentUnreachableError
from trainforge.tool_validator import AgentToolCall

Role = Literal["customer", "agent", "tool"]


class Message(TypedDict, total=False):
    role: Role
    content: str | None
    tool_calls: list[dict]
    tool_call_id: str
    name: str


@dataclass(frozen=True)
class AgentReply:
    """Parsed agent response.

    Either ``text`` is non-None (final text turn) or ``tool_calls`` is
    non-empty (tool round), or both (rare: agent prefaces tool calls with
    text). The runner drives loops based on whether ``tool_calls`` is set.
    """

    text: str | None
    tool_calls: list[AgentToolCall] = field(default_factory=list)

    @property
    def is_tool_round(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_empty(self) -> bool:
        return not self.tool_calls and not (self.text and self.text.strip())


@dataclass(frozen=True)
class AgentClient:
    """Thin wrapper around ``requests`` that enforces the spec's retry rules."""

    url: str
    timeout_seconds: float = 30.0

    def chat(self, messages: list[Message]) -> AgentReply:
        """Send ``messages`` to the agent and parse its response.

        One silent retry on timeout, per the spec.
        """
        payload = {"messages": messages}

        try:
            response = self._post(payload)
        except requests.exceptions.Timeout:
            try:
                response = self._post(payload)
            except requests.exceptions.Timeout as exc:
                raise AgentTimeoutError(
                    f"agent at {self.url} timed out twice after {self.timeout_seconds}s"
                ) from exc
            except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as exc:
                raise AgentUnreachableError(str(exc)) from exc
        except requests.exceptions.ConnectionError as exc:
            raise AgentUnreachableError(str(exc)) from exc
        except requests.exceptions.RequestException as exc:
            raise AgentUnreachableError(str(exc)) from exc

        if not response.ok:
            raise AgentError(
                f"agent at {self.url} returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise AgentError(f"agent at {self.url} returned non-JSON body") from exc

        if not isinstance(data, dict):
            raise AgentError(
                f"agent at {self.url} response is not an object: {type(data).__name__}"
            )

        return _parse_reply(data, url=self.url)

    def _post(self, payload: dict) -> requests.Response:
        return requests.post(
            self.url,
            json=payload,
            timeout=self.timeout_seconds,
            headers={"Content-Type": "application/json"},
        )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_reply(data: dict, *, url: str) -> AgentReply:
    text = data.get("response")
    if text is not None and not isinstance(text, str):
        raise AgentError(
            f"agent at {url} 'response' must be a string, got {type(text).__name__}"
        )

    raw_calls = data.get("tool_calls", []) or []
    if not isinstance(raw_calls, list):
        raise AgentError(
            f"agent at {url} 'tool_calls' must be an array, got {type(raw_calls).__name__}"
        )

    tool_calls: list[AgentToolCall] = []
    for i, raw in enumerate(raw_calls):
        tool_calls.append(_parse_tool_call(raw, url=url, index=i))

    if text is None and not tool_calls:
        raise AgentError(
            f"agent at {url} response had neither 'response' nor 'tool_calls'"
        )

    return AgentReply(text=text, tool_calls=tool_calls)


def _parse_tool_call(raw: Any, *, url: str, index: int) -> AgentToolCall:
    if not isinstance(raw, dict):
        raise AgentError(
            f"agent at {url} tool_calls[{index}] is not an object: {raw!r}"
        )
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise AgentError(
            f"agent at {url} tool_calls[{index}] missing 'name' string"
        )
    arguments = raw.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise AgentError(
            f"agent at {url} tool_calls[{index}] 'arguments' must be an object"
        )
    call_id = raw.get("id")
    if call_id is not None and not isinstance(call_id, str):
        raise AgentError(
            f"agent at {url} tool_calls[{index}] 'id' must be a string if present"
        )
    return AgentToolCall(name=name, arguments=dict(arguments), id=call_id)
