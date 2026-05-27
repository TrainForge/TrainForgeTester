"""End-to-end node_assertions tests via the runner.

Uses an in-process agent that wraps node calls in ``observer.node`` and
verifies the runner emits the right ``NodeAssertionResult`` for each
``AgentTurn.node_assertions`` declaration.
"""
from __future__ import annotations

import pytest

from trainforge import observer
from trainforge.runner import ScenarioRunner
from trainforge.schema import parse_scenarios
from trainforge.transport import InProcessTransport


@pytest.fixture
def must_fire_scenario():
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "sc-node-1",
                "name": "must-fire",
                "turns": [
                    {"role": "user", "message": "hi"},
                    {
                        "role": "agent",
                        "golden_response": "hello",
                        "node_assertions": [{"node_name": "greeter"}],
                    },
                ],
                "expected_outcome": "greeted",
                "outcome_checks": [],
            }
        ],
    }
    return parse_scenarios(raw).scenarios[0]


@pytest.fixture
def must_not_fire_scenario():
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "sc-node-2",
                "name": "must-not-fire",
                "turns": [
                    {"role": "user", "message": "hi"},
                    {
                        "role": "agent",
                        "golden_response": "hello",
                        "node_assertions": [
                            {"node_name": "danger_node", "must_fire": False}
                        ],
                    },
                ],
                "expected_outcome": "safe",
                "outcome_checks": [],
            }
        ],
    }
    return parse_scenarios(raw).scenarios[0]


@pytest.fixture
def args_match_scenario():
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "sc-node-3",
                "name": "args-match",
                "turns": [
                    {"role": "user", "message": "refund inv-1"},
                    {
                        "role": "agent",
                        "golden_response": "refunded",
                        "node_assertions": [
                            {
                                "node_name": "refund_handler",
                                "args_match": {"invoice_id": "inv-1"},
                            }
                        ],
                    },
                ],
                "expected_outcome": "ok",
                "outcome_checks": [],
            }
        ],
    }
    return parse_scenarios(raw).scenarios[0]


async def test_must_fire_passes_when_node_fires(must_fire_scenario, fake_llm) -> None:
    async def agent(messages):
        with observer.node("greeter"):
            pass
        return {"response": "hello"}

    runner = ScenarioRunner(agent=InProcessTransport(agent=agent), llm=fake_llm)
    result = await runner.run_scenario(must_fire_scenario, runs=1)
    nars = result.runs[0].turns[0].node_assertion_results
    assert len(nars) == 1
    assert nars[0].passed is True
    assert nars[0].fired is True


async def test_must_fire_fails_when_node_missing(must_fire_scenario, fake_llm) -> None:
    async def agent(messages):
        # No observer.node call.
        return {"response": "hello"}

    runner = ScenarioRunner(agent=InProcessTransport(agent=agent), llm=fake_llm)
    result = await runner.run_scenario(must_fire_scenario, runs=1)
    nars = result.runs[0].turns[0].node_assertion_results
    assert len(nars) == 1
    assert nars[0].passed is False
    assert "never fired" in nars[0].explanation


async def test_must_not_fire_passes_when_absent(must_not_fire_scenario, fake_llm) -> None:
    async def agent(messages):
        return {"response": "hello"}

    runner = ScenarioRunner(agent=InProcessTransport(agent=agent), llm=fake_llm)
    result = await runner.run_scenario(must_not_fire_scenario, runs=1)
    nars = result.runs[0].turns[0].node_assertion_results
    assert nars[0].passed is True
    assert nars[0].fired is False


async def test_must_not_fire_fails_when_node_fires(must_not_fire_scenario, fake_llm) -> None:
    async def agent(messages):
        with observer.node("danger_node"):
            pass
        return {"response": "hello"}

    runner = ScenarioRunner(agent=InProcessTransport(agent=agent), llm=fake_llm)
    result = await runner.run_scenario(must_not_fire_scenario, runs=1)
    nars = result.runs[0].turns[0].node_assertion_results
    assert nars[0].passed is False
    assert "required it NOT to fire" in nars[0].explanation


async def test_args_match_passes_with_matching_args(args_match_scenario, fake_llm) -> None:
    async def agent(messages):
        with observer.node("refund_handler", args={"invoice_id": "inv-1", "amount": 50}):
            pass
        return {"response": "refunded"}

    runner = ScenarioRunner(agent=InProcessTransport(agent=agent), llm=fake_llm)
    result = await runner.run_scenario(args_match_scenario, runs=1)
    assert result.runs[0].turns[0].node_assertion_results[0].passed is True


async def test_args_match_fails_with_wrong_args(args_match_scenario, fake_llm) -> None:
    async def agent(messages):
        with observer.node("refund_handler", args={"invoice_id": "inv-WRONG"}):
            pass
        return {"response": "refunded"}

    runner = ScenarioRunner(agent=InProcessTransport(agent=agent), llm=fake_llm)
    result = await runner.run_scenario(args_match_scenario, runs=1)
    nar = result.runs[0].turns[0].node_assertion_results[0]
    assert nar.passed is False
    assert "no invocation matched args_match" in nar.explanation
