"""Unit coverage for the pure helpers inside ``trainforge.record``.

The full REPL is interactive (reads from stdin), so we don't drive it
end-to-end here — we test the pieces that materialize a captured
session into a valid scenario JSON. The runner already covers the
in-process transport behavior :class:`_RecordSession` uses internally.
"""
from __future__ import annotations

from trainforge.record import (
    _CapturedTurn,
    _infer_type,
    _materialize_scenario,
    _RecordSession,
    _slug,
)
from trainforge.schema import ScenariosFile, parse_scenarios
from trainforge.transport import InProcessTransport


def _make_session_with_two_turns() -> _RecordSession:
    """A session with one text-only turn and one turn-with-tool-call.

    We don't actually drive the transport here; the data is set up
    declaratively so we exercise the JSON materializer in isolation.
    """
    session = _RecordSession(transport=InProcessTransport(agent=lambda m: {"response": ""}))
    session.turns.append(
        _CapturedTurn(
            user_message="hi",
            agent_text="hello",
            tool_calls=[],
            tool_responses=[],
        )
    )
    session.turns.append(
        _CapturedTurn(
            user_message="book one for 7pm",
            agent_text="booked corner table",
            tool_calls=[
                {"id": "c1", "name": "book", "arguments": {"time": "7pm", "party": 2}}
            ],
            tool_responses=[
                {
                    "tool_call_id": "c1",
                    "name": "book",
                    "arguments": {"time": "7pm", "party": 2},
                    "expected_response": "BOOKED #A1",
                }
            ],
        )
    )
    return session


def test_slug_strips_punctuation_and_dedupes_dashes() -> None:
    assert _slug("Hello, World!") == "hello-world"
    assert _slug("   trailing whitespace   ") == "trailing-whitespace"
    assert _slug("---only-dashes---") == "only-dashes"
    assert _slug("") == "scenario"
    assert _slug("!!!") == "scenario"


def test_infer_type_covers_basic_python_types() -> None:
    assert _infer_type("x") == "string"
    assert _infer_type(7) == "integer"
    assert _infer_type(1.5) == "number"
    assert _infer_type(True) == "boolean"
    assert _infer_type([1, 2]) == "array"
    assert _infer_type({"k": 1}) == "object"
    assert _infer_type(None) == "any"


def test_infer_type_bool_is_not_integer() -> None:
    """isinstance(True, int) is True in Python; the helper must check
    bool first or every boolean gets typed as integer."""
    assert _infer_type(True) == "boolean"
    assert _infer_type(False) == "boolean"


def test_materialize_scenario_writes_valid_schema() -> None:
    """The materialized JSON must validate against ScenariosFile."""
    session = _make_session_with_two_turns()
    doc = _materialize_scenario(
        session,
        scenario_id="sc-rec-1",
        name="Recorded test",
        description="captured via REPL",
        tags=["recorded"],
        expected_outcome="A table was booked.",
        outcome_checks=["A booking was confirmed"],
        per_turn_may_diverge=[False, True],
        per_turn_checks=[[], ["agent confirmed the booking"]],
    )

    # Validate against the schema (would raise on any field mismatch).
    parsed = parse_scenarios(doc)
    assert isinstance(parsed, ScenariosFile)
    assert parsed.version == "2.0"
    assert len(parsed.scenarios) == 1
    scenario = parsed.scenarios[0]
    assert scenario.id == "sc-rec-1"
    assert scenario.outcome_checks == ["A booking was confirmed"]


def test_materialize_scenario_alternates_user_agent_turns() -> None:
    session = _make_session_with_two_turns()
    doc = _materialize_scenario(
        session,
        scenario_id="sc-rec-2",
        name="alt",
        description="",
        tags=[],
        expected_outcome="ok",
        outcome_checks=[],
        per_turn_may_diverge=[False, False],
        per_turn_checks=[[], []],
    )
    turns = doc["scenarios"][0]["turns"]
    assert [t["role"] for t in turns] == ["user", "agent", "user", "agent"]


def test_materialize_scenario_tool_call_becomes_tool_loop() -> None:
    session = _make_session_with_two_turns()
    doc = _materialize_scenario(
        session,
        scenario_id="sc-rec-3",
        name="tools",
        description="",
        tags=[],
        expected_outcome="ok",
        outcome_checks=[],
        per_turn_may_diverge=[False, False],
        per_turn_checks=[[], []],
    )
    second_agent = doc["scenarios"][0]["turns"][3]
    assert "tool_loops" in second_agent
    loop = second_agent["tool_loops"][0]
    assert loop["ordered"] is False
    tool = loop["tools"][0]
    assert tool["name"] == "book"
    assert tool["expected_response"] == "BOOKED #A1"
    # Each declared arg gets typed and expected-bound.
    assert tool["arguments_schema"]["time"]["type"] == "string"
    assert tool["arguments_schema"]["time"]["expected"] == "7pm"
    assert tool["arguments_schema"]["party"]["type"] == "integer"
    assert tool["arguments_schema"]["party"]["expected"] == 2


def test_materialize_scenario_first_turn_no_tools_emits_no_tool_loops() -> None:
    """Turn 1 in the fixture has zero tool calls; the scenario must NOT
    declare an empty tool_loops list there (otherwise the runner would
    expect tool calls that never arrive)."""
    session = _make_session_with_two_turns()
    doc = _materialize_scenario(
        session,
        scenario_id="sc-rec-4",
        name="no-tools-turn",
        description="",
        tags=[],
        expected_outcome="ok",
        outcome_checks=[],
        per_turn_may_diverge=[False, False],
        per_turn_checks=[[], []],
    )
    first_agent = doc["scenarios"][0]["turns"][1]
    assert "tool_loops" not in first_agent
