from __future__ import annotations

from pathlib import Path

from trainforge.transport import AgentReply
from trainforge.errors import AgentError, AgentTimeoutError, AgentUnreachableError
from trainforge.runner import (
    ScenarioRunner,
    _as_turn_status,
    _call_to_dict,
    _default_for,
    _ensure_call_id,
)
from trainforge.schema import (
    ArgumentType,
    OutcomeStatus,
    TurnStatus,
    load_scenarios,
    parse_scenarios,
)
from trainforge.tool_validator import AgentToolCall


class _TimeoutAgent:
    async def chat(self, messages):  # type: ignore[no-untyped-def]
        raise AgentTimeoutError("slow")


class _UnreachableAgent:
    async def chat(self, messages):  # type: ignore[no-untyped-def]
        raise AgentUnreachableError("down")


class _AgentErrorAgent:
    async def chat(self, messages):  # type: ignore[no-untyped-def]
        raise AgentError("boom")


async def test_tool_loop_timeout_marks_turn_timeout(tools_scenarios_path: Path, fake_llm) -> None:
    scenario = load_scenarios(tools_scenarios_path).scenarios[0]
    fake_llm.queue_compact([1, 1])
    runner = ScenarioRunner(agent=_TimeoutAgent(), llm=fake_llm)  # type: ignore[arg-type]
    result = await runner.run_scenario(scenario, runs=1)
    run = result.runs[0]
    assert run.turns[0].status == "agent_timeout"
    assert run.status in {"partial_pass", "fail"}


async def test_tool_loop_unreachable_marks_run_unreachable(tools_scenarios_path: Path, fake_llm) -> None:
    scenario = load_scenarios(tools_scenarios_path).scenarios[0]
    runner = ScenarioRunner(agent=_UnreachableAgent(), llm=fake_llm)  # type: ignore[arg-type]
    result = await runner.run_scenario(scenario, runs=1)
    run = result.runs[0]
    assert run.status == "agent_unreachable"
    assert run.error == "agent_unreachable"
    assert run.outcome.status == "agent_unreachable"


def test_runner_helper_defaults_and_ids() -> None:
    assert _default_for(ArgumentType.ARRAY) == []
    assert _default_for(ArgumentType.OBJECT) == {}
    assert _default_for(ArgumentType.ANY) == ""

    generated = _ensure_call_id(None)
    assert generated.startswith("call_")
    assert _ensure_call_id("abc") == "abc"

    call = AgentToolCall(name="t", arguments={"a": 1}, id=None)
    as_dict = _call_to_dict(call)
    assert as_dict["name"] == "t"
    assert as_dict["arguments"] == {"a": 1}
    assert as_dict["id"].startswith("call_")


def test_as_turn_status_fallback() -> None:
    assert _as_turn_status(TurnStatus.AGENT_ERROR) == TurnStatus.AGENT_ERROR
    assert _as_turn_status(OutcomeStatus.AGENT_UNREACHABLE) == TurnStatus.AGENT_ERROR


async def test_unreachable_on_text_round_sets_error_flag(small_scenarios_path: Path, fake_llm) -> None:
    scenario = load_scenarios(small_scenarios_path).scenarios[0]

    class _TextThenUnreachable:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return AgentReply(text="ok")
            raise AgentUnreachableError("offline")

    fake_llm.queue_compact([1, 1])
    runner = ScenarioRunner(agent=_TextThenUnreachable(), llm=fake_llm)  # type: ignore[arg-type]
    result = await runner.run_scenario(scenario, runs=1)
    assert result.runs[0].status == "agent_unreachable"


async def test_text_round_agent_error_marks_turn(small_scenarios_path: Path, fake_llm) -> None:
    scenario = load_scenarios(small_scenarios_path).scenarios[0]
    fake_llm.queue_compact([1, 1])
    runner = ScenarioRunner(agent=_AgentErrorAgent(), llm=fake_llm)  # type: ignore[arg-type]
    result = await runner.run_scenario(scenario, runs=1)
    assert result.runs[0].turns[0].status == "agent_error"


async def test_may_diverge_true_path_populates_standard_checks(fake_llm) -> None:
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "s-md-1",
                "name": "md",
                "turns": [
                    {"role": "user", "message": "hi"},
                    {
                        "role": "agent",
                        "golden_response": "hello",
                        "may_diverge": True,
                        "checks": ["custom ok"],
                    },
                ],
                "expected_outcome": "ok",
                "outcome_checks": ["done"],
            }
        ],
    }
    scenario = parse_scenarios(raw).scenarios[0]

    class _OneReply:
        async def chat(self, messages):  # type: ignore[no-untyped-def]
            return AgentReply(text="hello")

    fake_llm.queue_compact([1] * 21)
    fake_llm.queue_compact([1])

    runner = ScenarioRunner(agent=_OneReply(), llm=fake_llm)  # type: ignore[arg-type]
    res = await runner.run_scenario(scenario, runs=1)
    turn = res.runs[0].turns[0]
    assert len(turn.standard_check_results) == 20
    assert len(turn.checks) == 1
    assert turn.status == "evaluated"


async def test_may_diverge_true_eval_error_marks_turn(fake_llm) -> None:
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "s-md-2",
                "name": "md2",
                "turns": [
                    {"role": "user", "message": "hi"},
                    {"role": "agent", "golden_response": "hello", "may_diverge": True, "checks": ["c"]},
                ],
                "expected_outcome": "ok",
                "outcome_checks": [],
            }
        ],
    }
    scenario = parse_scenarios(raw).scenarios[0]

    class _OneReply:
        async def chat(self, messages):  # type: ignore[no-untyped-def]
            return AgentReply(text="hello")

    fake_llm.queue_raw("not json")
    fake_llm.queue_raw("still not json")

    runner = ScenarioRunner(agent=_OneReply(), llm=fake_llm)  # type: ignore[arg-type]
    res = await runner.run_scenario(scenario, runs=1)
    turn = res.runs[0].turns[0]
    assert turn.status == "eval_error"
    assert all(not s.passed for s in turn.standard_check_results)


async def test_custom_check_eval_error_on_exact_match_path(fake_llm) -> None:
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "s-md-3",
                "name": "md3",
                "turns": [
                    {"role": "user", "message": "hi"},
                    {"role": "agent", "golden_response": "hello", "checks": ["c1", "c2"]},
                ],
                "expected_outcome": "ok",
                "outcome_checks": [],
            }
        ],
    }
    scenario = parse_scenarios(raw).scenarios[0]

    class _OneReply:
        async def chat(self, messages):  # type: ignore[no-untyped-def]
            return AgentReply(text="hello")

    fake_llm.queue_raw("bad")
    fake_llm.queue_raw("bad")

    runner = ScenarioRunner(agent=_OneReply(), llm=fake_llm)  # type: ignore[arg-type]
    res = await runner.run_scenario(scenario, runs=1)
    turn = res.runs[0].turns[0]
    assert turn.exact_match is True
    assert [c.explanation for c in turn.checks] == ["eval_error", "eval_error"]


async def test_no_outcome_checks_short_circuits_outcome_eval(fake_llm) -> None:
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "s-md-4",
                "name": "md4",
                "turns": [
                    {"role": "user", "message": "hi"},
                    {"role": "agent", "golden_response": "hello"},
                ],
                "expected_outcome": "ok",
                "outcome_checks": [],
            }
        ],
    }
    scenario = parse_scenarios(raw).scenarios[0]

    class _OneReply:
        async def chat(self, messages):  # type: ignore[no-untyped-def]
            return AgentReply(text="hello")

    runner = ScenarioRunner(agent=_OneReply(), llm=fake_llm)  # type: ignore[arg-type]
    res = await runner.run_scenario(scenario, runs=1)
    assert res.runs[0].outcome.status == "evaluated"
    assert res.runs[0].outcome.checks == []


async def test_text_round_timeout_marks_turn_timeout(small_scenarios_path: Path, fake_llm) -> None:
    scenario = load_scenarios(small_scenarios_path).scenarios[0]

    class _TextTimeout:
        async def chat(self, messages):  # type: ignore[no-untyped-def]
            raise AgentTimeoutError("slow")

    fake_llm.queue_compact([1, 1])
    runner = ScenarioRunner(agent=_TextTimeout(), llm=fake_llm)  # type: ignore[arg-type]
    res = await runner.run_scenario(scenario, runs=1)
    assert res.runs[0].turns[0].status == "agent_timeout"


async def test_tool_loop_agent_error_branch(tools_scenarios_path: Path, fake_llm) -> None:
    scenario = load_scenarios(tools_scenarios_path).scenarios[0]

    class _ToolLoopAgentError:
        async def chat(self, messages):  # type: ignore[no-untyped-def]
            raise AgentError("tool round failed")

    fake_llm.queue_compact([1, 1])
    runner = ScenarioRunner(agent=_ToolLoopAgentError(), llm=fake_llm)  # type: ignore[arg-type]
    res = await runner.run_scenario(scenario, runs=1)
    assert res.runs[0].turns[0].status == "agent_error"


async def test_unexpected_tool_in_loop_is_recorded(fake_llm) -> None:
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "s-md-5",
                "name": "extra-tool",
                "turns": [
                    {"role": "user", "message": "hi"},
                    {
                        "role": "agent",
                        "tool_loops": [
                            {
                                "ordered": True,
                                "tools": [
                                    {
                                        "name": "t1",
                                        "expected_response": "ok",
                                    }
                                ],
                            }
                        ],
                        "golden_response": "done",
                    },
                ],
                "expected_outcome": "ok",
                "outcome_checks": [],
            }
        ],
    }
    scenario = parse_scenarios(raw).scenarios[0]

    class _Agent:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return AgentReply(
                    text="",
                    tool_calls=[
                        AgentToolCall(name="t1", arguments={}),
                        AgentToolCall(name="extra", arguments={}),
                    ],
                )
            return AgentReply(text="done")

    runner = ScenarioRunner(agent=_Agent(), llm=fake_llm)  # type: ignore[arg-type]
    res = await runner.run_scenario(scenario, runs=1)
    statuses = [t.status for t in res.runs[0].turns[0].tool_calls]
    assert "unexpected_tool" in statuses


async def test_outcome_eval_error_branch(small_scenarios_path: Path, fake_llm) -> None:
    scenario = load_scenarios(small_scenarios_path).scenarios[0]

    class _Agent:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return AgentReply(text="What date and time would you like?")
            return AgentReply(text="Great, booked for 7pm on Friday for 2 people.")

    # turn1 custom checks pass
    fake_llm.queue_compact([1, 1])
    # turn2 custom checks pass
    fake_llm.queue_compact([1])
    # outcome fails to parse twice -> eval_error branch
    fake_llm.queue_raw("not-json")
    fake_llm.queue_raw("still-not-json")

    runner = ScenarioRunner(agent=_Agent(), llm=fake_llm)  # type: ignore[arg-type]
    res = await runner.run_scenario(scenario, runs=1)
    assert res.runs[0].outcome.status == "eval_error"
    assert all(not c.passed for c in res.runs[0].outcome.checks)
