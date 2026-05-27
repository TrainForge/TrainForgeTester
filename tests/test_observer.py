"""trainforge.observer tests: ContextVar capture, isolation, no-op safety."""
from __future__ import annotations

import asyncio

from trainforge import observer


def test_node_outside_capture_is_noop() -> None:
    # No active scenario; observer.node must be a no-op so user code is
    # safe to leave in place in production.
    with observer.node("free_node"):
        pass
    assert observer.captured_nodes() == []


def test_capture_records_node_fires() -> None:
    with observer.capture() as captured:
        with observer.node("a"):
            pass
        with observer.node("b", args={"x": 1}):
            pass
    assert [n.name for n in captured] == ["a", "b"]
    assert captured[1].args == {"x": 1}


def test_capture_isolates_between_scopes() -> None:
    with observer.capture() as first:
        with observer.node("a"):
            pass
    # New scope: prior captures should not leak in.
    with observer.capture() as second:
        with observer.node("b"):
            pass

    assert [n.name for n in first] == ["a"]
    assert [n.name for n in second] == ["b"]


async def test_capture_isolates_across_asyncio_gather() -> None:
    """ContextVar copies are independent per task — proves no cross-talk
    when --parallel runs multiple scenarios concurrently."""

    async def run_one(name: str):
        with observer.capture() as captured:
            await asyncio.sleep(0)  # yield so tasks interleave
            with observer.node(name):
                await asyncio.sleep(0)
        return [n.name for n in captured]

    results = await asyncio.gather(
        run_one("alpha"), run_one("beta"), run_one("gamma")
    )
    assert results == [["alpha"], ["beta"], ["gamma"]]


def test_captured_nodes_inside_scope_reflects_state() -> None:
    with observer.capture():
        with observer.node("first"):
            pass
        snapshot_after_first = [n.name for n in observer.captured_nodes()]
        with observer.node("second"):
            pass
        snapshot_after_second = [n.name for n in observer.captured_nodes()]
    assert snapshot_after_first == ["first"]
    assert snapshot_after_second == ["first", "second"]
