from __future__ import annotations

from trainforge.schema import ExpectedTool, ToolArgumentSchema, ToolLoop
from trainforge.tool_validator import AgentToolCall, LoopMatcher, validate_arguments


def test_args_validation_explanation_contains_all_groups() -> None:
    tool = ExpectedTool(
        name="t",
        arguments_schema={
            "a": ToolArgumentSchema(type="integer"),
            "b": ToolArgumentSchema(type="string", expected="x"),
            "c": ToolArgumentSchema(type="boolean"),
        },
        expected_response="ok",
    )
    v = validate_arguments(tool, {"a": "1", "b": "y"})
    assert not v.ok
    assert "missing arguments" in v.explanation
    assert "wrong-typed arguments" in v.explanation
    assert "wrong value" in v.explanation


def test_value_type_checks_cover_all_argument_types() -> None:
    tool = ExpectedTool(
        name="t",
        arguments_schema={
            "s": ToolArgumentSchema(type="string"),
            "i": ToolArgumentSchema(type="integer"),
            "n": ToolArgumentSchema(type="number"),
            "b": ToolArgumentSchema(type="boolean"),
            "a": ToolArgumentSchema(type="array"),
            "o": ToolArgumentSchema(type="object"),
            "x": ToolArgumentSchema(type="any"),
        },
        expected_response="ok",
    )
    ok = validate_arguments(
        tool,
        {"s": "a", "i": 1, "n": 1.2, "b": True, "a": [], "o": {}, "x": object()},
    )
    assert ok.ok


def test_ordered_matcher_unexpected_after_done() -> None:
    loop = ToolLoop(
        ordered=True,
        tools=[ExpectedTool(name="a", expected_response="A")],
    )
    matcher = LoopMatcher(loop)
    first = matcher.process([AgentToolCall(name="a", arguments={})])
    assert first[0].status == "pass"
    extra = matcher.process([AgentToolCall(name="x", arguments={})])
    assert extra[0].status == "unexpected_tool"

