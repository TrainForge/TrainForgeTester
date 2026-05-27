"""The single most important invariant: GOLDEN injection.

After each agent turn, the runner replaces the agent's actual response with
the golden response in the conversation history. The agent at turn N always
sees the correct conversation up to turn N-1.

Adapted to TrainForge 0.1: the small fixture uses ``may_diverge=False`` (exact-match
default), so the runner only calls the LLM for the per-scenario CUSTOM
checks on each turn (each custom check list is non-empty in the fixture)
plus the outcome eval.
"""
from __future__ import annotations

from pathlib import Path

from trainforge.transport import AgentReply, HttpTransport as AgentClient
from trainforge.runner import ScenarioRunner
from trainforge.schema import load_scenarios


class RecordingAgent:
    """Stand-in for :class:`AgentClient` that records every request."""

    def __init__(self, replies) -> None:  # type: ignore[no-untyped-def]
        self._replies = [
            r if isinstance(r, AgentReply) else AgentReply(text=r) for r in replies
        ]
        self.received_histories: list[list[dict]] = []

    async def chat(self, messages):  # type: ignore[no-untyped-def]
        self.received_histories.append([dict(m) for m in messages])
        return self._replies.pop(0)


def _queue_small_fixture_passing_run(fake_llm) -> None:
    """Queue LLM responses for one full pass of the small fixture.

    Small fixture (TrainForge 0.1):
    - 2 agent turns, both may_diverge=False.
    - Turn 1 has 2 custom checks -> 1 LLM call (compact, length 2)
    - Turn 2 has 1 custom check  -> 1 LLM call (compact, length 1)
    - Outcome has 2 checks       -> 1 LLM call (compact, length 2)
    """
    fake_llm.queue_compact([1, 1])  # turn 1 custom checks
    fake_llm.queue_compact([1])     # turn 2 custom check
    fake_llm.queue_compact([1, 1])  # outcome


async def test_agent_sees_golden_history_not_actual(
    small_scenarios_path: Path, fake_llm
) -> None:
    scenarios = load_scenarios(small_scenarios_path).scenarios
    scenario = scenarios[0]

    # Agent deliberately answers wrong on every turn (exact-match WILL fail).
    wrong = ["WRONG FIRST", "WRONG SECOND"]
    agent = RecordingAgent(replies=wrong)
    _queue_small_fixture_passing_run(fake_llm)  # custom checks + outcome

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = await runner.run_scenario(scenario, runs=1)
    assert len(result.runs) == 1

    # Both turns have exact_match=False (text didn't match).
    assert result.runs[0].turns[0].exact_match is False
    assert result.runs[0].turns[1].exact_match is False

    # The SECOND request the agent saw must contain the GOLDEN response from
    # turn 1, NOT the wrong string the agent actually returned. This is the
    # golden-injection invariant.
    assert len(agent.received_histories) == 2
    second_history = agent.received_histories[1]
    assert {
        "role": "agent",
        "content": scenario.turns[1].golden_response,  # type: ignore[union-attr]
    } in second_history
    assert {"role": "agent", "content": "WRONG FIRST"} not in second_history


async def test_outcome_eval_sees_actual_responses_not_golden(
    small_scenarios_path: Path, fake_llm
) -> None:
    """Outcome eval must receive the actual transcript."""
    scenarios = load_scenarios(small_scenarios_path).scenarios
    scenario = scenarios[0]

    wrong = ["wrong response A", "wrong response B"]
    agent = RecordingAgent(replies=wrong)
    _queue_small_fixture_passing_run(fake_llm)

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    await runner.run_scenario(scenario, runs=1)

    # Last LLM call is the outcome eval; its user prompt should include the
    # actual ("wrong") agent text and NOT the golden text.
    _system, user = fake_llm.calls[-1]
    assert "wrong response A" in user
    assert "wrong response B" in user
    assert scenario.turns[1].golden_response not in user  # type: ignore[union-attr]


async def test_runner_returns_per_turn_golden_and_actual(
    small_scenarios_path: Path, fake_llm
) -> None:
    scenarios = load_scenarios(small_scenarios_path).scenarios
    scenario = scenarios[0]
    # Agent text exactly matches the golden -> exact_match=True.
    agent = RecordingAgent(
        replies=[
            scenario.turns[1].golden_response,  # type: ignore[union-attr]
            scenario.turns[3].golden_response,  # type: ignore[union-attr]
        ]
    )
    _queue_small_fixture_passing_run(fake_llm)

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = await runner.run_scenario(scenario, runs=1)

    turn0 = result.runs[0].turns[0]
    assert turn0.actual_response == scenario.turns[1].golden_response  # type: ignore[union-attr]
    assert turn0.golden_response == scenario.turns[1].golden_response  # type: ignore[union-attr]
    assert turn0.status == "evaluated"
    assert turn0.exact_match is True
    assert turn0.standard_check_results == []  # may_diverge=False -> no standard checks
    assert all(c.passed for c in turn0.checks)


async def test_runner_multi_run(small_scenarios_path: Path, fake_llm) -> None:
    scenarios = load_scenarios(small_scenarios_path).scenarios
    scenario = scenarios[0]
    # All 3 runs send exactly the golden text.
    g1 = scenario.turns[1].golden_response  # type: ignore[union-attr]
    g2 = scenario.turns[3].golden_response  # type: ignore[union-attr]
    agent = RecordingAgent(replies=[g1, g2, g1, g2, g1, g2])

    for _ in range(3):
        _queue_small_fixture_passing_run(fake_llm)

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = await runner.run_scenario(scenario, runs=3)
    assert len(result.runs) == 3
    assert all(r.status == "pass" for r in result.runs), [r.status for r in result.runs]
    assert result.consistency == 1.0
    assert result.inconsistent is False


async def test_empty_response_is_treated_as_failure(
    small_scenarios_path: Path, fake_llm
) -> None:
    scenarios = load_scenarios(small_scenarios_path).scenarios
    scenario = scenarios[0]

    # Turn 1: empty (no LLM call needed; short-circuit). Turn 2: normal.
    agent = RecordingAgent(
        replies=["", scenario.turns[3].golden_response]  # type: ignore[union-attr]
    )
    fake_llm.queue_compact([1])     # turn 2 custom check
    fake_llm.queue_compact([1, 1])  # outcome

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = await runner.run_scenario(scenario, runs=1)
    run = result.runs[0]
    assert run.turns[0].status == "empty_response"
    assert run.turns[0].exact_match is False
    assert all(not c.passed for c in run.turns[0].checks)


def test_agent_client_constructible() -> None:
    _ = AgentClient(url="http://example.test/chat")
