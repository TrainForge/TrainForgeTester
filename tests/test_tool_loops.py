"""Tool-loop semantics and golden-injection for tool_calls (TrainForge 0.1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from trainforge.agent_client import AgentReply
from trainforge.runner import ScenarioRunner
from trainforge.schema import load_scenarios
from trainforge.tool_validator import AgentToolCall, LoopMatcher, validate_arguments


# ---------------------------------------------------------------------------
# Low-level: validate_arguments
# ---------------------------------------------------------------------------


def test_validate_arguments_happy_path(tools_scenarios_path: Path) -> None:
    sc = load_scenarios(tools_scenarios_path).scenarios[0]
    weather_tool = sc.turns[1].tool_loops[0].tools[0]  # type: ignore[union-attr]
    v = validate_arguments(weather_tool, {"when": "tonight"})
    assert v.ok


def test_validate_arguments_missing_required(tools_scenarios_path: Path) -> None:
    sc = load_scenarios(tools_scenarios_path).scenarios[0]
    availability = sc.turns[1].tool_loops[0].tools[1]  # type: ignore[union-attr]
    v = validate_arguments(availability, {"party_size": 2})  # missing "time"
    assert not v.ok
    assert "time" in v.missing_keys


def test_validate_arguments_wrong_type(tools_scenarios_path: Path) -> None:
    sc = load_scenarios(tools_scenarios_path).scenarios[0]
    availability = sc.turns[1].tool_loops[0].tools[1]  # type: ignore[union-attr]
    v = validate_arguments(availability, {"party_size": "two", "time": "7pm"})
    assert not v.ok
    assert "party_size" in v.wrong_type_keys


def test_validate_arguments_extra_keys_allowed(tools_scenarios_path: Path) -> None:
    sc = load_scenarios(tools_scenarios_path).scenarios[0]
    weather_tool = sc.turns[1].tool_loops[0].tools[0]  # type: ignore[union-attr]
    v = validate_arguments(weather_tool, {"when": "tonight", "extra": "ok"})
    assert v.ok


# ---------------------------------------------------------------------------
# LoopMatcher: ordered vs unordered
# ---------------------------------------------------------------------------


def test_unordered_loop_any_order_passes(tools_scenarios_path: Path) -> None:
    sc = load_scenarios(tools_scenarios_path).scenarios[0]
    loop = sc.turns[1].tool_loops[0]  # type: ignore[union-attr]
    matcher = LoopMatcher(loop)

    decisions = matcher.process(
        [
            AgentToolCall(name="check_availability", arguments={"party_size": 2, "time": "7pm"}),
            AgentToolCall(name="check_weather", arguments={"when": "tonight"}),
        ]
    )
    assert [d.status for d in decisions] == ["pass", "pass"]
    assert matcher.done


def test_unordered_loop_wrong_name_reports_wrong_tool(tools_scenarios_path: Path) -> None:
    sc = load_scenarios(tools_scenarios_path).scenarios[0]
    loop = sc.turns[1].tool_loops[0]  # type: ignore[union-attr]
    matcher = LoopMatcher(loop)

    decisions = matcher.process(
        [AgentToolCall(name="check_weather", arguments={"when": "tonight"}),
         AgentToolCall(name="some_other_tool", arguments={})]
    )
    assert decisions[0].status == "pass"
    # The second call doesn't match any remaining tool; runner consumes the
    # remaining slot and reports wrong_tool.
    assert decisions[1].status == "wrong_tool"
    assert decisions[1].matched_tool is not None
    assert decisions[1].matched_tool.name == "check_availability"
    assert matcher.done


def test_unordered_loop_invalid_arguments(tools_scenarios_path: Path) -> None:
    sc = load_scenarios(tools_scenarios_path).scenarios[0]
    loop = sc.turns[1].tool_loops[0]  # type: ignore[union-attr]
    matcher = LoopMatcher(loop)

    decisions = matcher.process(
        [AgentToolCall(name="check_availability", arguments={"party_size": 2})]
    )
    assert decisions[0].status == "invalid_arguments"
    assert "time" in decisions[0].explanation


def test_ordered_loop_requires_exact_order() -> None:
    from trainforge.schema import ExpectedTool, ToolLoop

    loop = ToolLoop(
        ordered=True,
        tools=[
            ExpectedTool(name="a", expected_response="A"),
            ExpectedTool(name="b", expected_response="B"),
        ],
    )
    matcher = LoopMatcher(loop)
    decisions = matcher.process([AgentToolCall(name="b", arguments={})])
    assert decisions[0].status == "wrong_tool"
    assert decisions[0].matched_tool is not None
    assert decisions[0].matched_tool.name == "a"
    # Second round: agent now calls "a" - but position 0 is consumed; matcher
    # expects "b" next.
    decisions2 = matcher.process([AgentToolCall(name="a", arguments={})])
    assert decisions2[0].status == "wrong_tool"
    assert decisions2[0].matched_tool.name == "b"
    assert matcher.done


def test_ordered_loop_in_order_passes() -> None:
    from trainforge.schema import ExpectedTool, ToolLoop

    loop = ToolLoop(
        ordered=True,
        tools=[
            ExpectedTool(name="a", expected_response="A"),
            ExpectedTool(name="b", expected_response="B"),
        ],
    )
    matcher = LoopMatcher(loop)
    decisions = matcher.process(
        [AgentToolCall(name="a", arguments={}), AgentToolCall(name="b", arguments={})]
    )
    assert [d.status for d in decisions] == ["pass", "pass"]
    assert matcher.done


def test_unexpected_tool_once_loop_is_done() -> None:
    from trainforge.schema import ExpectedTool, ToolLoop

    loop = ToolLoop(
        ordered=False,
        tools=[ExpectedTool(name="a", expected_response="A")],
    )
    matcher = LoopMatcher(loop)
    assert matcher.process([AgentToolCall(name="a", arguments={})])[0].status == "pass"
    assert matcher.done
    # Now any extra call is unexpected_tool.
    extra = matcher.process([AgentToolCall(name="b", arguments={})])
    assert extra[0].status == "unexpected_tool"


def test_finalize_returns_unmatched_positions() -> None:
    from trainforge.schema import ExpectedTool, ToolLoop

    loop = ToolLoop(
        ordered=False,
        tools=[
            ExpectedTool(name="a", expected_response="A"),
            ExpectedTool(name="b", expected_response="B"),
            ExpectedTool(name="c", expected_response="C"),
        ],
    )
    matcher = LoopMatcher(loop)
    matcher.process([AgentToolCall(name="b", arguments={})])
    remaining = matcher.finalize()
    assert remaining == [0, 2]  # a and c un-matched


# ---------------------------------------------------------------------------
# Runner: end-to-end with tool_loops (stub LLM + scripted agent)
# ---------------------------------------------------------------------------


class ScriptedAgent:
    """Returns pre-baked AgentReply objects in order, recording history."""

    def __init__(self, replies: list[AgentReply]) -> None:
        self._replies = list(replies)
        self.received_histories: list[list[dict]] = []

    def chat(self, messages):  # type: ignore[no-untyped-def]
        self.received_histories.append([dict(m) for m in messages])
        return self._replies.pop(0)


def _tool_call(name: str, arguments: dict, call_id: str | None = None) -> AgentToolCall:
    return AgentToolCall(name=name, arguments=arguments, id=call_id)


def test_full_run_tool_loops_pass(tools_scenarios_path: Path, fake_llm) -> None:
    scenario = load_scenarios(tools_scenarios_path).scenarios[0]
    agent = ScriptedAgent(
        [
            # Round 1: unordered loop, agent emits both tools together.
            AgentReply(
                text="",
                tool_calls=[
                    _tool_call("check_weather", {"when": "tonight"}),
                    _tool_call("check_availability", {"party_size": 2, "time": "7pm"}),
                ],
            ),
            # Round 2: ordered loop, agent emits book_table.
            AgentReply(
                text="",
                tool_calls=[
                    _tool_call(
                        "book_table",
                        {"party_size": 2, "time": "7pm", "seating": "indoor"},
                    )
                ],
            ),
            # Round 3: final text turn.
            AgentReply(text="Booked - corner table for 2 at 7pm, indoor (reference X-7)."),
        ]
    )
    # may_diverge=False on the agent turn -> exact-match path. The 1 custom
    # check still gets evaluated via LLM. Outcome eval also runs.
    fake_llm.queue_compact([1])      # 1 custom check passes
    fake_llm.queue_compact([1, 1])   # outcome: both checks pass

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = runner.run_scenario(scenario, runs=1)

    run = result.runs[0]
    assert run.status == "pass", [tc.status for tc in run.turns[0].tool_calls]
    assert len(run.turns) == 1
    turn = run.turns[0]
    assert [tc.status for tc in turn.tool_calls] == ["pass", "pass", "pass"]
    assert [tc.loop_index for tc in turn.tool_calls] == [0, 0, 1]
    assert turn.exact_match is True


def test_wrong_tool_reports_failure_but_injects_golden(
    tools_scenarios_path: Path, fake_llm
) -> None:
    """Golden-injection invariant for tool_calls: even when the agent calls
    the wrong tool, the history it sees on subsequent rounds contains the
    GOLDEN tool_call and the GOLDEN tool response, never the agent's mistake.
    """
    scenario = load_scenarios(tools_scenarios_path).scenarios[0]
    agent = ScriptedAgent(
        [
            # Round 1: agent calls a garbage tool instead of a weather tool.
            AgentReply(
                text="",
                tool_calls=[_tool_call("garbage_tool", {"foo": "bar"})],
            ),
            # Round 2: second unordered slot - still wrong.
            AgentReply(
                text="",
                tool_calls=[_tool_call("another_bad_one", {})],
            ),
            # Round 3: ordered loop book_table done correctly.
            AgentReply(
                text="",
                tool_calls=[
                    _tool_call(
                        "book_table",
                        {"party_size": 2, "time": "7pm", "seating": "indoor"},
                    )
                ],
            ),
            # Final text (intentionally different from golden -> exact_match=False).
            AgentReply(text="Booked - X-7."),
        ]
    )
    fake_llm.queue_compact([1])      # 1 custom check
    fake_llm.queue_compact([1, 1])   # outcome

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = runner.run_scenario(scenario, runs=1)

    tool_results = result.runs[0].turns[0].tool_calls
    statuses = [tc.status for tc in tool_results]
    # Two wrong_tool failures in the unordered loop, then one pass in the
    # ordered loop.
    assert statuses == ["wrong_tool", "wrong_tool", "pass"]
    # Scenario status: tool failures block PASS.
    assert result.runs[0].status != "pass"

    # GOLDEN INJECTION INVARIANT: by the time the agent is asked for the
    # final text turn, its history must contain the GOLDEN tool names and
    # GOLDEN tool responses, NOT the garbage ones.
    final_request_history = agent.received_histories[-1]
    tool_msgs = [m for m in final_request_history if m.get("role") == "tool"]
    tool_contents = [m["content"] for m in tool_msgs]
    assert "Tonight: cold and rainy." in tool_contents
    assert "Tables: corner (indoor), window (indoor)." in tool_contents
    assert "Booking confirmed. Reference: X-7." in tool_contents
    # Agent tool_calls in history should reference the GOLDEN names.
    agent_msgs_with_tools = [
        m for m in final_request_history
        if m.get("role") == "agent" and m.get("tool_calls")
    ]
    all_names = {c["name"] for m in agent_msgs_with_tools for c in m["tool_calls"]}
    assert "check_weather" in all_names
    assert "check_availability" in all_names
    assert "book_table" in all_names
    assert "garbage_tool" not in all_names
    assert "another_bad_one" not in all_names


def test_missing_tool_when_agent_emits_text_early(
    tools_scenarios_path: Path, fake_llm
) -> None:
    scenario = load_scenarios(tools_scenarios_path).scenarios[0]
    agent = ScriptedAgent(
        [
            # Agent only calls one of the two weather+availability tools, then
            # goes straight to text.
            AgentReply(
                text="",
                tool_calls=[_tool_call("check_weather", {"when": "tonight"})],
            ),
            AgentReply(text="Let me just guess - want the corner table?"),
            # Runner still expects the ordered book_table loop, then final text.
            # Because the agent emitted text mid-loop 0, remaining tools are
            # marked missing. The runner then proceeds to loop 1.
            AgentReply(
                text="",
                tool_calls=[
                    _tool_call(
                        "book_table",
                        {"party_size": 2, "time": "7pm", "seating": "indoor"},
                    )
                ],
            ),
            AgentReply(text="Booked - X-7."),
        ]
    )
    fake_llm.queue_compact([1])      # 1 custom check
    fake_llm.queue_compact([1, 1])   # outcome

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = runner.run_scenario(scenario, runs=1)
    tool_results = result.runs[0].turns[0].tool_calls
    statuses = [tc.status for tc in tool_results]
    assert "pass" in statuses
    assert "missing" in statuses
    # check_availability is the one that was missed.
    missed = [tc for tc in tool_results if tc.status == "missing"]
    assert missed[0].expected_name == "check_availability"


def test_invalid_arguments_records_failure(
    tools_scenarios_path: Path, fake_llm
) -> None:
    scenario = load_scenarios(tools_scenarios_path).scenarios[0]
    agent = ScriptedAgent(
        [
            AgentReply(
                text="",
                tool_calls=[
                    _tool_call("check_weather", {"when": "tonight"}),
                    # party_size missing.
                    _tool_call("check_availability", {"time": "7pm"}),
                ],
            ),
            AgentReply(
                text="",
                tool_calls=[
                    _tool_call(
                        "book_table",
                        {"party_size": 2, "time": "7pm", "seating": "indoor"},
                    )
                ],
            ),
            AgentReply(text="Booked - X-7."),
        ]
    )
    fake_llm.queue_compact([1])      # 1 custom check
    fake_llm.queue_compact([1, 1])   # outcome

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = runner.run_scenario(scenario, runs=1)
    tool_results = result.runs[0].turns[0].tool_calls
    statuses = [tc.status for tc in tool_results]
    assert statuses.count("invalid_arguments") == 1
    invalid = [tc for tc in tool_results if tc.status == "invalid_arguments"][0]
    assert invalid.expected_name == "check_availability"
    assert "party_size" in invalid.explanation


# ---------------------------------------------------------------------------
# Schema validation updates
# ---------------------------------------------------------------------------


def test_scenarios_without_tool_loops_still_valid(small_scenarios_path: Path) -> None:
    """Backward compatibility: scenarios pre-dating the tool_loops field
    continue to parse and have an empty tool_loops list per agent turn."""
    sc = load_scenarios(small_scenarios_path).scenarios[0]
    for turn in sc.turns:
        if turn.role == "agent":
            assert turn.tool_loops == []  # type: ignore[union-attr]


def test_example_scenario_exercises_tool_loops(example_scenarios_path: Path) -> None:
    from trainforge.schema import AgentTurn

    sc = load_scenarios(example_scenarios_path).scenarios[0]
    agent_turns = [t for t in sc.turns if isinstance(t, AgentTurn)]
    has_loops = [at for at in agent_turns if at.tool_loops]
    assert len(has_loops) == 2
    # Weather + availability: unordered, 2 tools.
    assert not has_loops[0].tool_loops[0].ordered
    assert len(has_loops[0].tool_loops[0].tools) == 2
    # Book_table: ordered, 1 tool.
    assert has_loops[1].tool_loops[0].ordered
    assert len(has_loops[1].tool_loops[0].tools) == 1


def test_empty_tool_loop_rejected() -> None:
    from pydantic import ValidationError

    from trainforge.schema import ToolLoop

    with pytest.raises(ValidationError):
        ToolLoop(ordered=False, tools=[])


# ---------------------------------------------------------------------------
# `expected` literal value checks (deterministic)
# ---------------------------------------------------------------------------


def test_expected_literal_value_mismatch_is_invalid_arguments() -> None:
    from trainforge.schema import ExpectedTool, ToolArgumentSchema

    tool = ExpectedTool(
        name="book_table",
        arguments_schema={
            "party_size": ToolArgumentSchema(type="integer", expected=2),
        },
        expected_response="ok",
    )
    v = validate_arguments(tool, {"party_size": 4})
    assert not v.ok
    assert "party_size" in v.wrong_value_keys


def test_expected_literal_value_matches_passes() -> None:
    from trainforge.schema import ExpectedTool, ToolArgumentSchema

    tool = ExpectedTool(
        name="book_table",
        arguments_schema={
            "party_size": ToolArgumentSchema(type="integer", expected=2),
        },
        expected_response="ok",
    )
    v = validate_arguments(tool, {"party_size": 2})
    assert v.ok


def test_expected_literal_value_takes_precedence_over_type_only() -> None:
    """If both type and expected are set, BOTH must pass."""
    from trainforge.schema import ExpectedTool, ToolArgumentSchema

    tool = ExpectedTool(
        name="t",
        arguments_schema={"seating": ToolArgumentSchema(type="string", expected="indoor")},
        expected_response="",
    )
    # Wrong type: caught as wrong_type, not wrong_value.
    assert "seating" in validate_arguments(tool, {"seating": 7}).wrong_type_keys
    # Right type, wrong value.
    assert "seating" in validate_arguments(tool, {"seating": "outdoor"}).wrong_value_keys
    # Correct.
    assert validate_arguments(tool, {"seating": "indoor"}).ok


# ---------------------------------------------------------------------------
# Deterministic `expected` mismatch -> invalid_arguments end-to-end
# ---------------------------------------------------------------------------


def test_wrong_expected_value_fails_run(
    tools_scenarios_path: Path, fake_llm
) -> None:
    """The case the user called out: tool call is structurally valid but the
    argument value doesn't match the declared literal. Status must be
    ``invalid_arguments``.
    """
    scenario = load_scenarios(tools_scenarios_path).scenarios[0]
    agent = ScriptedAgent(
        [
            AgentReply(
                text="",
                tool_calls=[
                    _tool_call("check_weather", {"when": "tonight"}),
                    # party_size has expected=2; agent sent 4 -> invalid_arguments.
                    _tool_call("check_availability", {"party_size": 4, "time": "7pm"}),
                ],
            ),
            AgentReply(
                text="",
                tool_calls=[
                    _tool_call(
                        "book_table",
                        {"party_size": 2, "time": "7pm", "seating": "indoor"},
                    )
                ],
            ),
            AgentReply(text="Booked!"),
        ]
    )
    fake_llm.queue_compact([1])      # 1 custom check
    fake_llm.queue_compact([1, 1])   # outcome

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    result = runner.run_scenario(scenario, runs=1)

    avail = next(
        tc for tc in result.runs[0].turns[0].tool_calls
        if tc.expected_name == "check_availability"
    )
    assert avail.status == "invalid_arguments"
    assert "party_size" in avail.explanation
    # Scenario cannot pass when any tool's status is not pass.
    assert result.runs[0].status != "pass"


def test_no_llm_calls_for_tool_argument_validation(
    tools_scenarios_path: Path, fake_llm
) -> None:
    """Tool-argument checking is purely deterministic: for a fully-passing
    run the LLM is called exactly twice (turn eval + outcome eval), never
    for tool arguments."""
    scenario = load_scenarios(tools_scenarios_path).scenarios[0]
    agent = ScriptedAgent(
        [
            AgentReply(
                text="",
                tool_calls=[
                    _tool_call("check_weather", {"when": "tonight"}),
                    _tool_call("check_availability", {"party_size": 2, "time": "7pm"}),
                ],
            ),
            AgentReply(
                text="",
                tool_calls=[
                    _tool_call(
                        "book_table",
                        {"party_size": 2, "time": "7pm", "seating": "indoor"},
                    )
                ],
            ),
            AgentReply(text="Booked - X-7."),
        ]
    )
    fake_llm.queue_compact([1])      # 1 custom check
    fake_llm.queue_compact([1, 1])   # outcome

    runner = ScenarioRunner(agent=agent, llm=fake_llm)  # type: ignore[arg-type]
    runner.run_scenario(scenario, runs=1)
    # 1 custom-check eval + 1 outcome eval -> exactly 2 LLM calls.
    # Tool-call validation never hits the LLM (deterministic).
    assert len(fake_llm.calls) == 2
