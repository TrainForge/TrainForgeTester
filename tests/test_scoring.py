"""PASS / PARTIAL PASS / FAIL rules under the TrainForge 0.1 deterministic-first model."""
from __future__ import annotations

import pytest

from trainforge.schema import (
    CheckResult,
    OutcomeResult,
    ScenarioResult,
    ScenarioRunResult,
    StandardCheckResult,
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
    exact_match: bool | None = True,
    standard: list[tuple[str, bool]] | None = None,
    custom: list[tuple[str, bool]] | None = None,
    status: str = "evaluated",
) -> TurnResult:
    """Build a TurnResult for scoring tests.

    For may_diverge=False: pass exact_match=True/False (default True).
    For may_diverge=True: pass exact_match=None and supply standard checks.
    """
    return TurnResult(
        turn_index=index,
        user_message="x",
        golden_response="g",
        actual_response="a" if exact_match is False else "g",
        may_diverge=may_diverge,
        status=status,  # type: ignore[arg-type]
        exact_match=None if may_diverge else exact_match,
        standard_check_results=[
            StandardCheckResult(id=s[0], question=s[0], passed=s[1])
            for s in (standard or [])
        ],
        checks=[CheckResult(check=c[0], passed=c[1]) for c in (custom or [])],
    )


def _outcome(*, passed: list[bool], status: str = "evaluated", error: str | None = None) -> OutcomeResult:
    return OutcomeResult(
        status=status,  # type: ignore[arg-type]
        checks=[CheckResult(check=f"o{i}", passed=p) for i, p in enumerate(passed)],
        error=error,
    )


# ---------------------------------------------------------------------------
# classify_scenario_run
# ---------------------------------------------------------------------------


def test_pass_when_exact_match_and_outcomes_pass() -> None:
    """Default may_diverge=False path: text matched verbatim, outcome OK."""
    turns = [_turn(custom=[("c", True)]), _turn(index=2, custom=[("c2", True)])]
    outcome = _outcome(passed=[True, True])
    assert classify_scenario_run(turns, outcome) == "pass"


def test_pass_when_may_diverge_and_all_standard_checks_pass() -> None:
    turns = [
        _turn(
            may_diverge=True,
            standard=[("same_language", True), ("same_intent", True)],
            custom=[("c", True)],
        )
    ]
    outcome = _outcome(passed=[True])
    assert classify_scenario_run(turns, outcome) == "pass"


def test_partial_when_exact_match_fails_but_outcome_passes() -> None:
    """TrainForge 0.1 specific: an exact-match miss on a may_diverge=False turn is a
    structural failure that prevents PASS. Outcome OK -> PARTIAL."""
    turns = [_turn(exact_match=False, custom=[("c", True)])]
    outcome = _outcome(passed=[True])
    assert classify_scenario_run(turns, outcome) == "partial_pass"


def test_partial_when_standard_check_fails_but_outcome_passes() -> None:
    """may_diverge=True turn where one of the 20 standard NLP checks fails."""
    turns = [
        _turn(
            may_diverge=True,
            standard=[("same_language", False), ("same_intent", True)],
        )
    ]
    outcome = _outcome(passed=[True])
    assert classify_scenario_run(turns, outcome) == "partial_pass"


def test_partial_when_custom_check_fails_but_outcome_passes() -> None:
    turns = [_turn(custom=[("c", False)])]
    outcome = _outcome(passed=[True])
    assert classify_scenario_run(turns, outcome) == "partial_pass"


def test_fail_when_any_outcome_check_fails() -> None:
    turns = [_turn(custom=[("c", True)])]
    outcome = _outcome(passed=[True, False])
    assert classify_scenario_run(turns, outcome) == "fail"


def test_fail_when_outcome_eval_errored() -> None:
    turns = [_turn(custom=[("c", True)])]
    outcome = _outcome(passed=[], status="eval_error", error="boom")
    assert classify_scenario_run(turns, outcome) == "fail"


def test_agent_unreachable_propagates() -> None:
    turns = [_turn(status="agent_error")]
    outcome = _outcome(passed=[], status="agent_unreachable", error="connection refused")
    assert classify_scenario_run(turns, outcome) == "agent_unreachable"


@pytest.mark.parametrize(
    "status", ["agent_error", "agent_timeout", "eval_error", "empty_response"]
)
def test_any_turn_level_error_prevents_full_pass(status: str) -> None:
    turns = [_turn(status=status, custom=[])]
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


def test_aggregate_consistency_empty_runs() -> None:
    rate, inconsistent = aggregate_consistency([])
    assert rate == 0.0
    assert inconsistent is False


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_counts_failures_by_category() -> None:
    turns = [
        _turn(exact_match=False),  # 1 exact_match failure
        _turn(
            index=2,
            may_diverge=True,
            standard=[("same_language", False), ("same_intent", True)],
            custom=[("c", False)],
        ),  # 1 standard failure + 1 custom failure
    ]
    run = ScenarioRunResult(
        run_index=0,
        status="partial_pass",
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
    assert summary["exact_match_failures"] == 1
    assert summary["standard_check_failures"] == 1
    assert summary["custom_check_failures"] == 1
    assert summary["total_scenarios"] == 1
    assert summary["partial"] == 1


def test_summarize_counts_unreachable_and_inconsistent() -> None:
    run = ScenarioRunResult(
        run_index=0,
        status="agent_unreachable",
        turns=[],
        outcome=OutcomeResult(status="agent_unreachable", checks=[]),
    )
    sc = ScenarioResult(
        scenario_id="u1",
        name="u1",
        runs=[run],
        consistency=0.0,
        inconsistent=True,
    )
    summary = summarize([sc])
    assert summary["unreachable"] == 1
    assert summary["inconsistent"] == 1

