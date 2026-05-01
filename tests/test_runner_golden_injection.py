"""The single most important invariant: GOLDEN injection.

Spec §"The Golden Injection Mechanism":

    After each agent turn, the runner replaces the agent's actual response
    with the golden response in the conversation history. The agent at turn
    N always sees the correct conversation up to turn N-1.

We verify this by:
  1. Giving the agent a deliberately-wrong response string on every turn.
  2. Asserting that every subsequent request the agent receives contains the
     GOLDEN response (not the wrong one) in its history.
"""
from __future__ import annotations

from pathlib import Path

from trainforge.agent_client import AgentClient, AgentReply
from trainforge.runner import ScenarioRunner
from trainforge.schema import load_scenarios


class RecordingAgent:
    """Stand-in for :class:`AgentClient` that records every request.

    ``replies`` accepts either raw text strings (wrapped into a text-only
    ``AgentReply``) or :class:`AgentReply` instances directly (for tool
    rounds).
    """

    def __init__(self, replies) -> None:  # type: ignore[no-untyped-def]
        self._replies = [
            r if isinstance(r, AgentReply) else AgentReply(text=r) for r in replies
        ]
        self.received_histories: list[list[dict]] = []

    def chat(self, messages):  # type: ignore[no-untyped-def]
        self.received_histories.append([dict(m) for m in messages])
        return self._replies.pop(0)


def _queue_small_fixture_passing_run(fake_llm) -> None:
    """Queue the 3 LLM responses for one pass run of the small fixture.

    The fixture has 2 agent turns (2 checks, 1 check) and 2 outcome checks.
    """
    fake_llm.queue_turn(
        consistency_score=5,
        divergence_type="none",
        checks=[("c1", True, ""), ("c2", True, "")],
    )
    fake_llm.queue_turn(
        consistency_score=5,
        divergence_type="none",
        checks=[("c1", True, "")],
    )
    fake_llm.queue_outcome([("o1", True, ""), ("o2", True, "")])


def test_agent_sees_golden_history_not_actual(
    small_scenarios_path: Path, fake_llm
) -> None:
    scenarios = load_scenarios(small_scenarios_path).scenarios
    scenario = scenarios[0]

    # The agent deliberately answers with a wrong string at every turn.
    wrong = ["WRONG FIRST", "WRONG SECOND"]
    agent = RecordingAgent(replies=wrong)

    # Queue two agent-turn evals + one outcome eval.
    fake_llm.queue_turn(consistency_score=3, divergence_type="factual_difference", checks=[("c", True, ""), ("c2", True, "")])
    fake_llm.queue_turn(consistency_score=3, divergence_type="factual_difference", checks=[("c", True, "")])
    fake_llm.queue_outcome([("o", True, ""), ("o2", True, "")])

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = runner.run_scenario(scenario, runs=1)
    assert len(result.runs) == 1

    # The SECOND request the agent saw should contain the GOLDEN response
    # from turn 1, NOT the wrong string it actually returned.
    assert len(agent.received_histories) == 2
    second_history = agent.received_histories[1]
    assert {
        "role": "agent",
        "content": scenario.turns[1].golden_response,  # type: ignore[union-attr]
    } in second_history
    assert {"role": "agent", "content": "WRONG FIRST"} not in second_history


def test_outcome_eval_sees_actual_responses_not_golden(
    small_scenarios_path: Path, fake_llm
) -> None:
    """Outcome eval MUST receive the actual transcript, per spec §Step 3."""
    scenarios = load_scenarios(small_scenarios_path).scenarios
    scenario = scenarios[0]

    wrong = ["wrong response A", "wrong response B"]
    agent = RecordingAgent(replies=wrong)

    fake_llm.queue_turn(consistency_score=3, divergence_type="factual_difference", checks=[("c", True, ""), ("c2", True, "")])
    fake_llm.queue_turn(consistency_score=3, divergence_type="factual_difference", checks=[("c", True, "")])
    fake_llm.queue_outcome([("o", True, ""), ("o2", True, "")])

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    runner.run_scenario(scenario, runs=1)

    # The outcome eval user prompt should include the actual ("wrong") responses.
    _system, user = fake_llm.calls[-1]
    assert "wrong response A" in user
    assert "wrong response B" in user
    # And must NOT include the golden responses in the conversation block.
    assert scenario.turns[1].golden_response not in user  # type: ignore[union-attr]


def test_runner_returns_per_turn_golden_and_actual(
    small_scenarios_path: Path, fake_llm
) -> None:
    scenarios = load_scenarios(small_scenarios_path).scenarios
    scenario = scenarios[0]
    agent = RecordingAgent(replies=["agent a1", "agent a2"])

    fake_llm.queue_turn(consistency_score=5, divergence_type="none", checks=[("c", True, ""), ("c2", True, "")])
    fake_llm.queue_turn(consistency_score=5, divergence_type="none", checks=[("c", True, "")])
    fake_llm.queue_outcome([("o", True, ""), ("o2", True, "")])

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = runner.run_scenario(scenario, runs=1)

    turn0 = result.runs[0].turns[0]
    assert turn0.actual_response == "agent a1"
    assert turn0.golden_response == scenario.turns[1].golden_response  # type: ignore[union-attr]
    assert turn0.status == "evaluated"
    assert turn0.diverged is False


def test_runner_multi_run(small_scenarios_path: Path, fake_llm) -> None:
    scenarios = load_scenarios(small_scenarios_path).scenarios
    scenario = scenarios[0]
    agent = RecordingAgent(replies=["a1", "a2", "a1", "a2", "a1", "a2"])

    for _ in range(3):
        _queue_small_fixture_passing_run(fake_llm)

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = runner.run_scenario(scenario, runs=3)
    assert len(result.runs) == 3
    assert all(r.status == "pass" for r in result.runs), [r.status for r in result.runs]
    assert result.consistency == 1.0
    assert result.inconsistent is False


def test_empty_response_is_treated_as_divergence(
    small_scenarios_path: Path, fake_llm
) -> None:
    scenarios = load_scenarios(small_scenarios_path).scenarios
    scenario = scenarios[0]

    # First turn returns empty, second turn normal.
    agent = RecordingAgent(replies=["", "agent a2"])
    # Only the second turn goes through the evaluator; the first short-circuits.
    fake_llm.queue_turn(consistency_score=5, divergence_type="none", checks=[("c", True, "")])
    fake_llm.queue_outcome([("o", True, ""), ("o2", True, "")])

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = runner.run_scenario(scenario, runs=1)
    run = result.runs[0]
    assert run.turns[0].status == "empty_response"
    assert run.turns[0].diverged is True
    # All checks on the empty turn should be failed.
    assert all(not c.passed for c in run.turns[0].checks)


def test_agent_url_accepts_agentclient_interface() -> None:
    """Sanity: ensure the runner type hints are satisfied by AgentClient too."""
    _ = AgentClient(url="http://example.test/chat")
