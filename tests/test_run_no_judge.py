"""`trainforge run --no-judge` and the deferred-labeling pipeline.

Covers the runner side: no_judge=True must:

- Skip every LLM-dependent evaluation (custom checks, standard NLP,
  outcome) and emit them with ``pending=True``.
- Still run the agent and capture actual vs golden.
- Still evaluate tool calls deterministically.
- Treat pending checks as "not yet judged" in scoring — they don't
  trigger failures in the deterministic-only summary.
"""
from __future__ import annotations

import pytest

from trainforge.runner import ScenarioRunner
from trainforge.schema import ScenarioStatus, parse_scenarios
from trainforge.transport import InProcessTransport


@pytest.fixture
def scenario_with_judge_dependencies():
    """A scenario that touches every LLM-dependent path:
    - one may_diverge=True turn (triggers 20 standard checks + custom)
    - outcome_checks (triggers outcome eval)
    """
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "sc-no-judge-1",
                "name": "no-judge",
                "turns": [
                    {"role": "user", "message": "hi"},
                    {
                        "role": "agent",
                        "golden_response": "hello",
                        "may_diverge": True,
                        "checks": ["agent greeted politely"],
                    },
                ],
                "expected_outcome": "greeted",
                "outcome_checks": ["a greeting was returned"],
            }
        ],
    }
    return parse_scenarios(raw).scenarios[0]


async def test_no_judge_emits_pending_standard_checks(
    scenario_with_judge_dependencies, fake_llm
) -> None:
    async def agent(messages):
        return {"response": "hello"}

    runner = ScenarioRunner(
        agent=InProcessTransport(agent=agent), llm=fake_llm, no_judge=True
    )
    result = await runner.run_scenario(scenario_with_judge_dependencies, runs=1)

    turn = result.runs[0].turns[0]
    assert len(turn.standard_check_results) == 20
    assert all(s.pending for s in turn.standard_check_results)
    assert all(not s.passed for s in turn.standard_check_results)


async def test_no_judge_emits_pending_custom_checks(
    scenario_with_judge_dependencies, fake_llm
) -> None:
    async def agent(messages):
        return {"response": "hello"}

    runner = ScenarioRunner(
        agent=InProcessTransport(agent=agent), llm=fake_llm, no_judge=True
    )
    result = await runner.run_scenario(scenario_with_judge_dependencies, runs=1)

    turn = result.runs[0].turns[0]
    assert len(turn.checks) == 1
    assert turn.checks[0].pending is True
    assert turn.checks[0].passed is False


async def test_no_judge_emits_pending_outcome_checks(
    scenario_with_judge_dependencies, fake_llm
) -> None:
    async def agent(messages):
        return {"response": "hello"}

    runner = ScenarioRunner(
        agent=InProcessTransport(agent=agent), llm=fake_llm, no_judge=True
    )
    result = await runner.run_scenario(scenario_with_judge_dependencies, runs=1)

    outcome = result.runs[0].outcome
    assert len(outcome.checks) == 1
    assert outcome.checks[0].pending is True
    assert outcome.checks[0].passed is False


async def test_no_judge_does_not_call_llm(
    scenario_with_judge_dependencies, fake_llm
) -> None:
    """Important: no_judge=True must produce ZERO LLM calls. The fixture
    starts with `fake_llm.calls == []`; it must stay that way."""
    async def agent(messages):
        return {"response": "hello"}

    runner = ScenarioRunner(
        agent=InProcessTransport(agent=agent), llm=fake_llm, no_judge=True
    )
    await runner.run_scenario(scenario_with_judge_dependencies, runs=1)

    assert fake_llm.calls == []


async def test_no_judge_classifies_scenarios_as_pass_when_deterministic_clean(
    scenario_with_judge_dependencies, fake_llm
) -> None:
    """With every LLM-judged check pending and no deterministic failures,
    the scenario should be PASS in the intermediate summary. The coding
    agent's labeling step is what produces the final verdict."""
    async def agent(messages):
        return {"response": "hello"}

    runner = ScenarioRunner(
        agent=InProcessTransport(agent=agent), llm=fake_llm, no_judge=True
    )
    result = await runner.run_scenario(scenario_with_judge_dependencies, runs=1)
    assert result.runs[0].status == ScenarioStatus.PASS


async def test_no_judge_does_not_mask_deterministic_failures(fake_llm) -> None:
    """A tool-call failure must still surface even in no-judge mode —
    those checks are deterministic, no LLM involved."""
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "sc-no-judge-tool-fail",
                "name": "tool-fail",
                "turns": [
                    {"role": "user", "message": "book"},
                    {
                        "role": "agent",
                        "golden_response": "booked",
                        "tool_loops": [
                            {
                                "ordered": False,
                                "tools": [
                                    {
                                        "name": "book_table",
                                        "arguments_schema": {
                                            "time": {"type": "string", "expected": "7pm"}
                                        },
                                        "expected_response": "OK",
                                    }
                                ],
                            }
                        ],
                    },
                ],
                "expected_outcome": "booked",
                "outcome_checks": [],
            }
        ],
    }
    scenario = parse_scenarios(raw).scenarios[0]

    async def agent(messages):
        # First call returns the WRONG tool; second call returns the
        # final text once the runner injects the canned tool response.
        if not any(m.get("role") == "tool" for m in messages):
            return {
                "tool_calls": [
                    {"id": "c1", "name": "delete_table", "arguments": {"time": "7pm"}}
                ]
            }
        return {"response": "booked"}

    runner = ScenarioRunner(
        agent=InProcessTransport(agent=agent), llm=fake_llm, no_judge=True
    )
    result = await runner.run_scenario(scenario, runs=1)
    # Wrong tool was deterministically caught → scenario fails.
    assert result.runs[0].status != ScenarioStatus.PASS
