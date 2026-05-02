from __future__ import annotations

from trainforge.diff import DiffReport
from trainforge.report.html import (
    _collect_failures,
    _collect_standard_failures,
    _first_failing_turn,
    _pct,
    _plain,
    _status_label,
    render_diff_html,
    render_report_html,
)
from trainforge.schema import (
    CheckResult,
    OutcomeResult,
    RunConfig,
    RunResults,
    RunSummary,
    ScenarioResult,
    ScenarioRunResult,
    StandardCheckResult,
    TurnResult,
)


def _turn(*, status: str = "evaluated", exact_match: bool | None = True) -> TurnResult:
    return TurnResult(
        turn_index=1,
        user_message="u",
        golden_response="g",
        actual_response="a",
        may_diverge=False,
        status=status,  # type: ignore[arg-type]
        exact_match=exact_match,
        checks=[CheckResult(check="c", passed=True)],
    )


def _results() -> RunResults:
    bad_turn = TurnResult(
        turn_index=1,
        user_message="u",
        golden_response="g",
        actual_response="a",
        may_diverge=True,
        status="evaluated",
        exact_match=None,
        standard_check_results=[
            StandardCheckResult(id="same_language", question="q", passed=False, explanation="x"),
            StandardCheckResult(id="same_intent", question="q2", passed=True, explanation=""),
        ],
        checks=[],
    )
    run = ScenarioRunResult(
        run_index=0,
        status="fail",
        turns=[bad_turn],
        outcome=OutcomeResult(
            status="evaluated",
            checks=[CheckResult(check="o", passed=False, explanation="no")],
        ),
    )
    sc = ScenarioResult(
        scenario_id="s1",
        name="Scenario 1",
        runs=[run],
        consistency=0.0,
        inconsistent=False,
    )
    return RunResults(
        config=RunConfig(agent_url="http://a", llm_model="m", runs=1, timeout_seconds=1.0),
        summary=RunSummary(
            total_scenarios=1,
            passed=0,
            partial=0,
            failed=1,
            unreachable=0,
            inconsistent=0,
            pass_rate=0.0,
            overall_consistency=0.0,
        ),
        scenarios=[sc],
    )


def test_status_label_mapping() -> None:
    assert _status_label("agent_timeout") == "AGENT TIMEOUT"
    assert _status_label("custom") == "CUSTOM"


def test_collectors_and_renderers() -> None:
    results = _results()
    plain = _plain(results)
    assert isinstance(plain, dict)

    failures = _collect_failures(results)
    assert len(failures) == 1
    assert failures[0]["first_failing"].turn_index == 1

    std = _collect_standard_failures(results)
    assert len(std) == 1
    assert std[0]["id"] == "same_language"

    assert "TrainForge Report" in render_report_html(results)

    diff = DiffReport(before_path="b.json", after_path="a.json", before=results, after=results)
    assert "Regression" in render_diff_html(diff)


def test_first_failing_turn_none_when_run_clean() -> None:
    run = ScenarioRunResult(
        run_index=0,
        status="pass",
        turns=[_turn(status="evaluated", exact_match=True)],
        outcome=OutcomeResult(status="evaluated", checks=[]),
    )
    assert _first_failing_turn(run) is None


def test_plain_handles_dict_and_list_recursively() -> None:
    payload = {"a": [1, {"b": 2}], "c": "x"}
    out = _plain(payload)
    assert out == payload


def test_pct_filter() -> None:
    assert _pct(0.42) == "42%"


def test_first_failing_turn_axes() -> None:
    base = TurnResult(
        turn_index=1,
        user_message="u",
        golden_response="g",
        actual_response="a",
        may_diverge=True,
        status="evaluated",
        exact_match=None,
        standard_check_results=[StandardCheckResult(id="id", question="q", passed=True)],
        checks=[CheckResult(check="c", passed=True)],
    )

    run_status = ScenarioRunResult(
        run_index=0,
        status="fail",
        turns=[TurnResult(**{**base.model_dump(), "status": "agent_error"})],
        outcome=OutcomeResult(status="evaluated", checks=[]),
    )
    assert _first_failing_turn(run_status) is not None

    run_tool = ScenarioRunResult(
        run_index=0,
        status="fail",
        turns=[TurnResult(**{**base.model_dump(), "tool_calls": [{"loop_index": 0, "position": 0, "expected_name": "t", "status": "wrong_tool"}]})],
        outcome=OutcomeResult(status="evaluated", checks=[]),
    )
    assert _first_failing_turn(run_tool) is not None

    run_exact = ScenarioRunResult(
        run_index=0,
        status="fail",
        turns=[TurnResult(**{**base.model_dump(), "may_diverge": False, "exact_match": False})],
        outcome=OutcomeResult(status="evaluated", checks=[]),
    )
    assert _first_failing_turn(run_exact) is not None

    run_standard = ScenarioRunResult(
        run_index=0,
        status="fail",
        turns=[TurnResult(**{**base.model_dump(), "standard_check_results": [{"id": "id", "question": "q", "passed": False}]})],
        outcome=OutcomeResult(status="evaluated", checks=[]),
    )
    assert _first_failing_turn(run_standard) is not None

    run_custom = ScenarioRunResult(
        run_index=0,
        status="fail",
        turns=[TurnResult(**{**base.model_dump(), "checks": [{"check": "c", "passed": False}]})],
        outcome=OutcomeResult(status="evaluated", checks=[]),
    )
    assert _first_failing_turn(run_custom) is not None

