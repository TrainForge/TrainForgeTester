"""Transport types and protocol.

Wire contract (unchanged from v0.1, originally lived in agent_client.py):

- ``Message`` is the on-the-wire shape passed to and returned by the agent.
- ``Role`` enumerates ``user`` / ``agent`` / ``tool``.
- ``AgentReply`` is the parsed response from one turn of the agent.
- ``Transport`` is the Protocol every transport must satisfy.

Both :class:`HttpTransport` and :class:`InProcessTransport` produce identical
``AgentReply`` shapes; the runner never branches on transport type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypedDict

from trainforge.errors import AgentError
from trainforge.tool_validator import AgentToolCall

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11
    class StrEnum(str, Enum):
        pass


class Role(StrEnum):
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"


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


class Transport(Protocol):
    """Anything the runner can call to get an agent reply.

    Every transport implements ``async def chat(messages) -> AgentReply``.
    Implementations are responsible for mapping their own failure modes
    (HTTP errors, Python exceptions, timeouts) to the canonical
    :class:`~trainforge.errors.AgentError`,
    :class:`~trainforge.errors.AgentTimeoutError`, and
    :class:`~trainforge.errors.AgentUnreachableError`.
    """

    async def chat(self, messages: list[Message]) -> AgentReply: ...


# ---------------------------------------------------------------------------
# Response parsing (shared between HTTP and in-process transports)
# ---------------------------------------------------------------------------


def parse_reply(data: dict, *, source: str) -> AgentReply:
    """Parse a raw agent-response dict into :class:`AgentReply`.

    ``source`` is used in error messages so users see whether the bad shape
    came from an HTTP endpoint or an in-process callable.
    """
    text = data.get("response")
    if text is not None and not isinstance(text, str):
        raise AgentError(
            f"agent at {source} 'response' must be a string, got {type(text).__name__}"
        )

    raw_calls = data.get("tool_calls", []) or []
    if not isinstance(raw_calls, list):
        raise AgentError(
            f"agent at {source} 'tool_calls' must be an array, got {type(raw_calls).__name__}"
        )

    tool_calls: list[AgentToolCall] = []
    for i, raw in enumerate(raw_calls):
        tool_calls.append(_parse_tool_call(raw, source=source, index=i))

    if text is None and not tool_calls:
        raise AgentError(
            f"agent at {source} response had neither 'response' nor 'tool_calls'"
        )

    return AgentReply(text=text, tool_calls=tool_calls)


def _parse_tool_call(raw: Any, *, source: str, index: int) -> AgentToolCall:
    if not isinstance(raw, dict):
        raise AgentError(
            f"agent at {source} tool_calls[{index}] is not an object: {raw!r}"
        )
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise AgentError(
            f"agent at {source} tool_calls[{index}] missing 'name' string"
        )
    arguments = raw.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise AgentError(
            f"agent at {source} tool_calls[{index}] 'arguments' must be an object"
        )
    call_id = raw.get("id")
    if call_id is not None and not isinstance(call_id, str):
        raise AgentError(
            f"agent at {source} tool_calls[{index}] 'id' must be a string if present"
        )
    return AgentToolCall(name=name, arguments=dict(arguments), id=call_id)
