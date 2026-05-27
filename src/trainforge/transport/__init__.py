"""Agent transport layer.

A :class:`Transport` is whatever can answer ``await transport.chat(messages)``
with an :class:`AgentReply`. Two implementations ship today:

- :class:`HttpTransport` (in :mod:`trainforge.transport.http`) talks to an
  HTTP agent endpoint via ``httpx.AsyncClient``.
- :class:`InProcessTransport` (in :mod:`trainforge.transport.in_process`)
  awaits a Python callable directly. Sync callables are wrapped via
  ``asyncio.to_thread``.

Both produce the same :class:`AgentReply` shape so the runner does not need
to know which transport produced a reply. New transports (stdio, gRPC, WASM)
slot in by implementing the same Protocol.
"""
from __future__ import annotations

from trainforge.transport.base import (
    AgentReply,
    Message,
    Role,
    Transport,
)
from trainforge.transport.http import HttpTransport
from trainforge.transport.in_process import InProcessTransport

__all__ = [
    "AgentReply",
    "HttpTransport",
    "InProcessTransport",
    "Message",
    "Role",
    "Transport",
]
