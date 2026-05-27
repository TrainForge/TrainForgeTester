"""HTTP transport implementation.

Talks to an agent at a URL over HTTP using ``httpx.AsyncClient``. The wire
contract is unchanged from v0.1:

    POST {agent-url}
    Request  : {"messages": [Message, ...]}
    Response : {"response": "optional text", "tool_calls": [ToolCall, ...]}

Either ``response`` or ``tool_calls`` may be absent, but at least one must
carry signal.

Error handling matches the spec's "Error Handling" table:

- Agent returns non-2xx        -> :class:`AgentError`
- Agent times out              -> retry once; on second timeout, :class:`AgentTimeoutError`
- Agent unreachable / network  -> :class:`AgentUnreachableError`
- Body missing both fields     -> :class:`AgentError`
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from trainforge.errors import AgentError, AgentTimeoutError, AgentUnreachableError
from trainforge.transport.base import AgentReply, Message, parse_reply


@dataclass(frozen=True)
class HttpTransport:
    """Async HTTP transport for the agent-under-test.

    Stateless per-call: no persistent connection pool is kept across
    ``chat`` invocations. Callers running large suites should consider the
    ``--parallel`` flag, which uses ``asyncio.gather`` to overlap requests.
    """

    url: str
    timeout_seconds: float = 30.0

    async def chat(self, messages: list[Message]) -> AgentReply:
        """Send ``messages`` to the agent and parse its response.

        One silent retry on timeout, per the spec.
        """
        payload = {"messages": messages}

        try:
            response = await self._post(payload)
        except httpx.TimeoutException:
            try:
                response = await self._post(payload)
            except httpx.TimeoutException as exc:
                raise AgentTimeoutError(
                    f"agent at {self.url} timed out twice after {self.timeout_seconds}s"
                ) from exc
            except httpx.RequestError as exc:
                raise AgentUnreachableError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise AgentUnreachableError(str(exc)) from exc

        # Non-2xx is an error per spec. httpx.AsyncClient does NOT follow
        # redirects by default, so 3xx responses would otherwise leak
        # through to the JSON-parse path with a confusing "non-JSON body"
        # error. Use `is_success` (== 2xx) instead of a `>= 400` check.
        if not response.is_success:
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

        return parse_reply(data, source=self.url)

    async def _post(self, payload: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            return await client.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
