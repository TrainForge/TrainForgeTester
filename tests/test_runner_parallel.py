"""Parallel scenario execution via asyncio.gather + Semaphore.

The CLI `--parallel N` flag is wired in cli.py; here we test the
runner-level behavior directly so regressions don't slip past the CLI
layer (which is fiddlier to isolate).
"""
from __future__ import annotations

import asyncio
import time

from trainforge.runner import ScenarioRunner
from trainforge.schema import parse_scenarios
from trainforge.transport import InProcessTransport


def _make_minimal_scenario(scenario_id: str):
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": scenario_id,
                "name": "parallel",
                "turns": [
                    {"role": "user", "message": "ping"},
                    {"role": "agent", "golden_response": "pong"},
                ],
                "expected_outcome": "ack",
                "outcome_checks": [],
            }
        ],
    }
    return parse_scenarios(raw).scenarios[0]


async def test_asyncio_gather_runs_scenarios_concurrently(fake_llm) -> None:
    """Four scenarios with a 100ms agent delay should finish in roughly
    100ms wall-time when run via asyncio.gather, not 400ms."""
    delay = 0.1

    async def slow_agent(messages):
        await asyncio.sleep(delay)
        return {"response": "pong"}

    runner = ScenarioRunner(
        agent=InProcessTransport(agent=slow_agent), llm=fake_llm
    )
    scenarios = [_make_minimal_scenario(f"sc-{i}") for i in range(4)]

    started = time.perf_counter()
    results = await asyncio.gather(*(runner.run_scenario(sc, runs=1) for sc in scenarios))
    elapsed = time.perf_counter() - started

    assert len(results) == 4
    # Generous upper bound (3x serial-min) to avoid CI flake; tight enough
    # to fail loudly if scenarios serialized accidentally (4 * 100ms = 400ms).
    assert elapsed < delay * 3, f"expected concurrent execution; took {elapsed:.2f}s"


async def test_parallel_scenarios_each_see_isolated_observer_state(fake_llm) -> None:
    """ContextVar isolation across asyncio.gather: each scenario's observed
    nodes don't bleed into other scenarios' results."""
    from trainforge import observer

    async def agent(messages):
        # The "name" we record is the user message so we can correlate.
        user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            {"content": "?"},
        )["content"]
        with observer.node(f"node-{user_msg}"):
            await asyncio.sleep(0.01)
        return {"response": "pong"}

    raw_template = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "sc-iso",
                "name": "iso",
                "turns": [
                    {"role": "user", "message": "MARKER"},
                    {
                        "role": "agent",
                        "golden_response": "pong",
                        "node_assertions": [{"node_name": "node-MARKER"}],
                    },
                ],
                "expected_outcome": "ok",
                "outcome_checks": [],
            }
        ],
    }

    scenarios = []
    for marker in ("alpha", "beta", "gamma"):
        raw = {
            "version": "2.0",
            "scenarios": [
                {
                    **raw_template["scenarios"][0],
                    "id": f"sc-{marker}",
                    "turns": [
                        {"role": "user", "message": marker},
                        {
                            "role": "agent",
                            "golden_response": "pong",
                            "node_assertions": [{"node_name": f"node-{marker}"}],
                        },
                    ],
                }
            ],
        }
        scenarios.append(parse_scenarios(raw).scenarios[0])

    runner = ScenarioRunner(agent=InProcessTransport(agent=agent), llm=fake_llm)
    results = await asyncio.gather(
        *(runner.run_scenario(sc, runs=1) for sc in scenarios)
    )

    # All three node assertions pass — each scenario saw only its own node.
    for result in results:
        nars = result.runs[0].turns[0].node_assertion_results
        assert len(nars) == 1
        assert nars[0].passed is True


async def test_runner_run_scenario_is_idempotent_when_called_twice(fake_llm) -> None:
    """Sanity: the runner has no hidden state that breaks back-to-back
    invocations against the same scenario."""

    async def agent(messages):
        return {"response": "pong"}

    runner = ScenarioRunner(agent=InProcessTransport(agent=agent), llm=fake_llm)
    scenario = _make_minimal_scenario("sc-idempotent")

    a = await runner.run_scenario(scenario, runs=1)
    b = await runner.run_scenario(scenario, runs=1)

    assert a.runs[0].status == b.runs[0].status
    assert a.runs[0].turns[0].exact_match == b.runs[0].turns[0].exact_match
