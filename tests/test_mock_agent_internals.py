from __future__ import annotations

import random

import pytest

from trainforge.mock_agent import (
    Mode,
    MockAgentServer,
    _is_user_role,
    _last_user_content,
    _pending_tool_position,
    _perturb_text,
    _perturb_tool_call,
    _plan_response,
    _sample,
)
from trainforge.schema import AgentTurn, ArgumentType, ExpectedTool


def test_invalid_mode_is_rejected(small_scenarios_path) -> None:
    with pytest.raises(ValueError):
        MockAgentServer(str(small_scenarios_path), mode="bad-mode")


def test_sample_supports_all_types() -> None:
    rng = random.Random(1)
    assert isinstance(_sample(ArgumentType.STRING, rng), str)
    assert isinstance(_sample(ArgumentType.INTEGER, rng), int)
    assert isinstance(_sample(ArgumentType.NUMBER, rng), float)
    assert isinstance(_sample(ArgumentType.BOOLEAN, rng), bool)
    assert _sample(ArgumentType.ARRAY, rng) == []
    assert _sample(ArgumentType.OBJECT, rng) == {}
    assert _sample(ArgumentType.ANY, rng) == "value"


def test_last_user_content_and_user_role_aliases() -> None:
    assert _is_user_role("user") is True
    assert _is_user_role("customer") is True
    assert _is_user_role("agent") is False

    history = [{"role": "customer", "content": "hello"}, {"role": "agent", "content": "x"}]
    assert _last_user_content(history) == "hello"

    no_text = [{"role": "user", "content": 123}]
    assert _last_user_content(no_text) is None


def test_pending_tool_position_and_plan_response_for_plain_turn() -> None:
    turn = AgentTurn(role="agent", golden_response="done")
    assert _pending_tool_position(turn, [{"role": "user", "content": "x"}]) is None

    resp = _plan_response(turn, [{"role": "user", "content": "x"}], Mode.GOLDEN, random.Random(1))
    assert resp == {"response": "done"}


def test_pending_tool_position_counts_tool_calls(tools_scenarios_path) -> None:
    from trainforge.schema import load_scenarios

    scenario = load_scenarios(tools_scenarios_path).scenarios[0]
    turn = scenario.turns[1]
    assert isinstance(turn, AgentTurn)

    # No tool rounds yet -> first tool in first loop.
    pos0 = _pending_tool_position(turn, [{"role": "user", "content": "x"}])
    assert pos0 == (0, 0)

    # One tool call consumed -> second position in loop 0.
    hist = [
        {"role": "user", "content": "x"},
        {"role": "agent", "content": "", "tool_calls": [{"id": "1", "name": "check_weather", "arguments": {}}]},
    ]
    pos1 = _pending_tool_position(turn, hist)
    assert pos1 == (0, 1)


def test_start_is_idempotent(small_scenarios_path) -> None:
    server = MockAgentServer(str(small_scenarios_path), port=0, mode="golden")
    try:
        server.start()
        first_thread = server._thread
        server.start()
        assert server._thread is first_thread
    finally:
        server.stop()


def test_stop_without_start_is_noop(small_scenarios_path) -> None:
    server = MockAgentServer(str(small_scenarios_path), port=0, mode="golden")
    server.stop()


def test_serve_forever_closes_server(monkeypatch, small_scenarios_path) -> None:
    server = MockAgentServer(str(small_scenarios_path), port=0, mode="golden")

    called = {"close": 0}

    def _serve_forever() -> None:
        return

    def _close() -> None:
        called["close"] += 1

    monkeypatch.setattr(server._server, "serve_forever", _serve_forever)
    monkeypatch.setattr(server._server, "server_close", _close)

    server.serve_forever()
    assert called["close"] == 1


def test_perturb_text_swapped_branch() -> None:
    class _Rng:
        @staticmethod
        def random() -> float:
            return 0.1

    out = _perturb_text("cold rainy indoor", _Rng())
    assert out == "warm sunny outdoor"


def test_perturb_tool_call_unrelated_branch() -> None:
    class _Rng:
        @staticmethod
        def random() -> float:
            return 0.95

    expected = ExpectedTool(name="check_weather", expected_response="ok")
    name, args = _perturb_tool_call(expected, {"when": "tonight"}, _Rng())
    assert name == "totally_unrelated_tool"
    assert args == {"bogus": True}
