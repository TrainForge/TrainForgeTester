"""PASS / PARTIAL PASS / FAIL rules from spec Evaluation & Scoring."""
from __future__ import annotations

import pytest

from trainforge.schema import (
    CheckResult,
    OutcomeResult,
    ScenarioResult,
    ScenarioRunResult,
    TurnResult,
)
from trainforge.scoring import (
    INCONSISTENCY_THRESHOLD,
    aggregate_consistency,
    classify_scenario_run,
    summarize,
)


def _turn(
    *,
    index: int = 0,
    may_diverge: bool = False,
    diverged: bool = False,
    checks: list[tuple[str, bool]] | None = None,
    status: str = "evaluated",
    consistency: int | None = 5,
) -> TurnResult:
    return TurnResult(
        turn_index=index,
        customer_message="x",
        golden_response="g",
        actual_response="a",
        may_diverge=may_diverge,
        status=status,  # type: ignore[arg-type]
        consistency_score=consistency,
        divergence_type="none" if not diverged else "factual_difference",
        checks=[CheckResult(check=c[0], passed=c[1]) for c in (checks or [])],
        diverged=diverged,
    )


def _outcome(*, passed: list[bool], status: str = "evaluated", error: str | None = None) -> OutcomeResult:
    return OutcomeResult(
        status=status,  # type: ignore[arg-type]
        checks=[CheckResult(check=f"o{i}", passed=p) for i, p in enumerate(passed)],
        error=error,
    )


# ---------------------------------------------------------------------------
# classify_scenario_run - the heart of scoring
# ---------------------------------------------------------------------------


def test_pass_when_all_turn_checks_and_outcomes_pass() -> None:
    turns = [_turn(checks=[("c", True)]), _turn(index=2, checks=[("c2", True)])]
    outcome = _outcome(passed=[True, True])
    assert classify_scenario_run(turns, outcome) == "pass"


def test_may_diverge_turn_failures_do_not_cause_failure() -> None:
    """Spec: Divergences on may_diverge turns are NOT counted as failures."""
    turns = [
        _turn(checks=[("c1", True)]),
        _turn(
            index=2,
            may_diverge=True,
            diverged=True,
            consistency=2,
            checks=[("c2", False)],
        ),
    ]
    outcome = _outcome(passed=[True])
    assert classify_scenario_run(turns, outcome) == "pass"


def test_partial_when_non_divergent_turn_check_fails_but_outcome_ok() -> None:
    turns = [_turn(checks=[("c", False)])]
    outcome = _outcome(passed=[True])
    assert classify_scenario_run(turns, outcome) == "partial_pass"


def test_fail_when_any_outcome_check_fails() -> None:
    turns = [_turn(checks=[("c", True)])]
    outcome = _outcome(passed=[True, False])
    assert classify_scenario_run(turns, outcome) == "fail"


def test_fail_when_outcome_eval_errored() -> None:
    turns = [_turn(checks=[("c", True)])]
    outcome = _outcome(passed=[], status="eval_error", error="boom")
    assert classify_scenario_run(turns, outcome) == "fail"


def test_agent_unreachable_propagates() -> None:
    turns = [_turn(status="agent_error")]
    outcome = _outcome(passed=[], status="agent_unreachable", error="connection refused")
    assert classify_scenario_run(turns, outcome) == "agent_unreachable"


@pytest.mark.parametrize("status", ["agent_error", "agent_timeout", "eval_error", "empty_response"])
def test_any_turn_level_error_prevents_full_pass(status: str) -> None:
    turns = [_turn(status=status, checks=[])]
    outcome = _outcome(passed=[True])
    assert classify_scenario_run(turns, outcome) == "partial_pass"


# ---------------------------------------------------------------------------
# aggregate_consistency
# ---------------------------------------------------------------------------


def _run(status: str, index: int = 0) -> ScenarioRunResult:
    return ScenarioRunResult(
        run_index=index,
        status=status,  # type: ignore[arg-type]
        turns=[],
        outcome=OutcomeResult(status="evaluated", checks=[]),
    )


def test_single_run_never_flagged_inconsistent() -> None:
    rate, inconsistent = aggregate_consistency([_run("fail")])
    assert rate == 0.0
    assert inconsistent is False


def test_consistency_threshold_matches_spec() -> None:
    assert INCONSISTENCY_THRESHOLD == 0.80


def test_inconsistent_below_threshold() -> None:
    runs = [_run("pass", 0), _run("pass", 1), _run("fail", 2), _run("fail", 3), _run("fail", 4)]
    rate, inconsistent = aggregate_consistency(runs)
    assert rate == pytest.approx(0.4)
    assert inconsistent is True


def test_consistent_at_threshold() -> None:
    runs = [_run("pass"), _run("pass"), _run("pass"), _run("pass"), _run("fail")]
    rate, inconsistent = aggregate_consistency(runs)
    assert rate == pytest.approx(0.8)
    assert inconsistent is False


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_counts_divergences_by_category() -> None:
    turns = [
        _turn(diverged=True, may_diverge=False),
        _turn(diverged=True, may_diverge=True),
        _turn(diverged=False),
    ]
    run = ScenarioRunResult(
        run_index=0,
        status="pass",
        turns=turns,
        outcome=OutcomeResult(status="evaluated", checks=[]),
    )
    sc = ScenarioResult(
        scenario_id="a",
        name="a",
        runs=[run],
        consistency=1.0,
        inconsistent=False,
    )
    summary = summarize([sc])
    assert summary["unexpected_divergences"] == 1
    assert summary["expected_divergences"] == 1
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["total_scenarios"] == 1
