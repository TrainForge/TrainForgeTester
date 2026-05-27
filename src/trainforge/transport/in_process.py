"""In-process Python transport.

Calls the user's Python callable directly. The callable receives a list of
``Message`` dicts and returns a dict shaped like the HTTP response
(``{"response": str?, "tool_calls": [...]?}``). Both ``async def`` and
``def`` callables are supported; sync callables are wrapped via
``asyncio.to_thread`` so the runner main loop stays free.

Failure modes are mapped to the same error classes as the HTTP transport:

- callable raises any non-asyncio exception -> :class:`AgentError`
- callable exceeds ``timeout_seconds``      -> :class:`AgentTimeoutError`
- callable returns the wrong shape          -> :class:`AgentError`
"""
from __future__ import annotations

import asyncio
import copy
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Union

from trainforge.errors import AgentError, AgentTimeoutError
from trainforge.transport.base import AgentReply, Message, parse_reply

# An in-process agent callable returns either a dict directly (sync) or a
# coroutine that resolves to a dict (async). We dispatch on inspect.
AgentCallable = Union[
    Callable[[list[Message]], dict[str, Any]],
    Callable[[list[Message]], Awaitable[dict[str, Any]]],
]


@dataclass(frozen=True)
class InProcessTransport:
    """Calls a Python callable as if it were an agent.

    The callable shape:

        async def agent(messages: list[dict]) -> dict
        # or
        def agent(messages: list[dict]) -> dict

    Return value must match the HTTP response shape:

        {"response": "...", "tool_calls": [{"id": "...", "name": "...", "arguments": {...}}, ...]}

    At least one of ``response`` / ``tool_calls`` must carry signal, same
    rule as the HTTP transport.
    """

    agent: AgentCallable
    timeout_seconds: float = 30.0
    source: str = "in-process"

    async def chat(self, messages: list[Message]) -> AgentReply:
        try:
            data = await asyncio.wait_for(
                self._invoke(messages), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise AgentTimeoutError(
                f"agent at {self.source} did not return within {self.timeout_seconds}s"
            ) from exc
        except (AgentError, AgentTimeoutError):
            raise
        except Exception as exc:  # noqa: BLE001 — wrap any user exception
            raise AgentError(
                f"agent at {self.source} raised {type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise AgentError(
                f"agent at {self.source} returned {type(data).__name__}, expected dict"
            )

        return parse_reply(data, source=self.source)

    async def _invoke(self, messages: list[Message]) -> Any:
        # Deep-copy so the user's callable can mutate messages OR any of
        # the nested dicts/lists (e.g., `messages[0]["content"] = ...`)
        # without corrupting the runner's golden/actual histories.
        # Shallow copying the outer list would still alias the inner
        # dicts; deep copy is the only safe answer.
        defensive = copy.deepcopy(messages)
        if inspect.iscoroutinefunction(self.agent):
            return await self.agent(defensive)
        result = await asyncio.to_thread(self.agent, defensive)
        # If a sync callable accidentally returns a coroutine (e.g., user
        # wrote `def agent(...): return async_thing(...)`), await it.
        if inspect.iscoroutine(result):
            return await result
        return result
