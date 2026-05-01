"""Regression-diff bucketization tests."""
from __future__ import annotations

from trainforge.diff import compute_diff
from trainforge.schema import (
    OutcomeResult,
    RunConfig,
    RunResults,
    RunSummary,
    ScenarioResult,
    ScenarioRunResult,
)


def _scenario(
    scenario_id: str,
    *,
    passed: bool,
    consistency: float = 1.0,
    inconsistent: bool = False,
    name: str | None = None,
) -> ScenarioResult:
    run = ScenarioRunResult(
        run_index=0,
        status="pass" if passed else "fail",
        turns=[],
        outcome=OutcomeResult(status="evaluated", checks=[]),
    )
    return ScenarioResult(
        scenario_id=scenario_id,
        name=name or scenario_id,
        runs=[run],
        consistency=consistency,
        inconsistent=inconsistent,
    )


def _results(*scenarios: ScenarioResult) -> RunResults:
    summary = RunSummary(
        total_scenarios=len(scenarios),
        passed=sum(1 for s in scenarios if any(r.status == "pass" for r in s.runs)),
        partial=0,
        failed=sum(1 for s in scenarios if not any(r.status == "pass" for r in s.runs)),
        unreachable=0,
        inconsistent=sum(1 for s in scenarios if s.inconsistent),
        pass_rate=0.0,
        overall_consistency=0.0,
    )
    return RunResults(
        config=RunConfig(
            agent_url="http://example", llm_model="fake", runs=1, timeout_seconds=30.0
        ),
        summary=summary,
        scenarios=list(scenarios),
    )


def test_bucketization() -> None:
    before = _results(
        _scenario("a", passed=True),
        _scenario("b", passed=False),
        _scenario("c", passed=True),
        _scenario("d", passed=False),
        _scenario("e", passed=True, consistency=1.0),
        _scenario("f", passed=True, consistency=0.4, inconsistent=True),
    )
    after = _results(
        _scenario("a", passed=True),          # still_passing
        _scenario("b", passed=True),          # newly_passing
        _scenario("c", passed=False),         # newly_failing
        _scenario("d", passed=False),         # still_failing
        _scenario("e", passed=True, consistency=0.5),  # consistency_changed
        _scenario("f", passed=True, consistency=0.9),  # consistency_changed
        _scenario("g", passed=True),          # only_in_after
    )
    # Nothing "only_in_before" here.

    report = compute_diff(before, after, consistency_epsilon=0.1)

    assert {e.scenario_id for e in report.still_passing} == {"a"}
    assert {e.scenario_id for e in report.newly_passing} == {"b"}
    assert {e.scenario_id for e in report.newly_failing} == {"c"}
    assert {e.scenario_id for e in report.still_failing} == {"d"}
    assert {e.scenario_id for e in report.consistency_changed} == {"e", "f"}
    assert {e.scenario_id for e in report.only_in_after} == {"g"}
    assert report.only_in_before == []
    assert report.regressed_count == 1
    assert report.fixed_count == 1


def test_only_in_before_detected() -> None:
    before = _results(_scenario("a", passed=True), _scenario("b", passed=False))
    after = _results(_scenario("a", passed=True))

    report = compute_diff(before, after)
    assert [e.scenario_id for e in report.only_in_before] == ["b"]


def test_consistency_epsilon_respected() -> None:
    before = _results(_scenario("a", passed=True, consistency=0.9))
    after = _results(_scenario("a", passed=True, consistency=0.85))
    report = compute_diff(before, after, consistency_epsilon=0.1)
    # 0.05 < 0.1 -> still_passing, not consistency_changed.
    assert [e.scenario_id for e in report.still_passing] == ["a"]
    assert report.consistency_changed == []
