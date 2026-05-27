"""In-process node observer for multi-agent assertions.

Usage from an agent function:

    from trainforge import observer

    async def run_agent(messages):
        with observer.node("intent_classifier"):
            intent = classify(messages)
        if intent == "refund":
            with observer.node("refund_handler", args={"reason": "..."}):
                return handle_refund(messages)

The runner reads :func:`captured_nodes` after each agent turn to evaluate
``AgentTurn.node_assertions``.

Isolation:
    State is held in a :class:`~contextvars.ContextVar`, so concurrent
    scenarios under ``asyncio.gather`` each see their own list. Within a
    single scenario, the runner resets the captured list before each turn
    so node fires from turn N don't leak into turn N+1.

Caveats:
    Under multiprocessing / pytest-xdist (separate processes), each worker
    has its own ContextVar. The observer is in-process only by design;
    multi-agent assertions are an in-process feature.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class NodeFire:
    """One observed node entry / exit pair."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


_CAPTURED: ContextVar[list[NodeFire] | None] = ContextVar(
    "trainforge_observer_captured", default=None
)


def captured_nodes() -> list[NodeFire]:
    """Return the list of nodes fired since the last :func:`reset_captured`.

    Returns an empty list if no scenario is active (i.e., the observer is
    being called outside a TrainForge turn) so user code that calls this
    in production has zero behavior change.
    """
    state = _CAPTURED.get()
    if state is None:
        return []
    return list(state)


@contextmanager
def capture() -> Iterator[list[NodeFire]]:
    """Establish a fresh capture scope.

    The runner enters this scope around each agent invocation so node
    fires from turn N don't bleed into turn N+1, and so concurrent
    scenarios (asyncio.gather) each see their own list.
    """
    state: list[NodeFire] = []
    token = _CAPTURED.set(state)
    try:
        yield state
    finally:
        _CAPTURED.reset(token)


@contextmanager
def node(name: str, *, args: dict[str, Any] | None = None) -> Iterator[None]:
    """Mark that a named sub-agent / node ran during this scope.

    Outside an active :func:`capture` scope (i.e., user code running in
    production), this is a no-op so it's always safe to leave in place.
    """
    state = _CAPTURED.get()
    if state is None:
        # Production / not under a TrainForge scenario: no-op.
        yield
        return
    state.append(NodeFire(name=name, args=dict(args or {})))
    yield


def reset_captured() -> None:
    """Clear the captured list within the current scope.

    Provided for tests; the runner uses :func:`capture` as the canonical
    reset (entering a new scope is cleaner than mutating an existing one).
    """
    state = _CAPTURED.get()
    if state is not None:
        state.clear()
