"""Scoring rules from testing-spec-v1.md section "Evaluation & Scoring".

Kept in one small module so the PASS / PARTIAL / FAIL table is unambiguous
and easy to unit-test.

Rules (verbatim from spec):

    Scenario PASS if:
      - ALL checks on non-divergent turns pass
      - ALL outcome_checks pass
      - Divergences on may_diverge turns are NOT counted as failures

    Scenario PARTIAL PASS if:
      - outcome_checks pass (agent reached the right result)
      - BUT some non-divergent turn checks failed (agent took a weird path)

    Scenario FAIL if:
      - Any outcome_check fails (agent did not reach the right result)

    Consistency:
      Per-scenario consistency = (runs where scenario PASSED) / N
      Flag as INCONSISTENT if consistency < 80%

v1.1 addition: a turn's tool_calls are treated like checks for PASS/PARTIAL
classification. A turn is considered fully successful only when every
``ToolCallRecord`` has status ``pass`` (alongside its non-divergent LLM
checks). ``may_diverge`` covers *text* divergence only; wrong tool calls
always count against the run.
"""
from __future__ import annotations

from trainforge.schema import (
    OutcomeResult,
    ScenarioResult,
    ScenarioRunResult,
    ScenarioStatus,
    TurnResult,
)


INCONSISTENCY_THRESHOLD = 0.80


def classify_scenario_run(
    turn_results: list[TurnResult], outcome: OutcomeResult
) -> ScenarioStatus:
    """Map one run's turn + outcome results to a :data:`ScenarioStatus`."""
    if outcome.status == "agent_unreachable":
        return "agent_unreachable"

    outcome_ok = outcome.status == "evaluated" and all(c.passed for c in outcome.checks)

    turn_checks_ok = _all_non_divergent_turn_checks_pass(turn_results)

    if outcome_ok and turn_checks_ok:
        return "pass"
    if outcome_ok and not turn_checks_ok:
        return "partial_pass"
    return "fail"


def aggregate_consistency(runs: list[ScenarioRunResult]) -> tuple[float, bool]:
    """Return ``(pass_rate, inconsistent?)`` across ``runs``.

    ``inconsistent`` is only meaningful when there are multiple runs;
    single-run scenarios are never flagged inconsistent.
    """
    if not runs:
        return 0.0, False
    passed = sum(1 for r in runs if r.status == "pass")
    rate = passed / len(runs)
    inconsistent = len(runs) > 1 and rate < INCONSISTENCY_THRESHOLD
    return rate, inconsistent


def summarize(scenarios: list[ScenarioResult]) -> dict[str, int | float]:
    """Build a dict suitable for :class:`trainforge.schema.RunSummary`.

    Uses the *best* run per scenario for pass/partial/fail bucketing
    (i.e. a scenario counts as "passed" if it passed in at least one run;
    "inconsistent" is carried separately). Pass rate is the fraction of
    scenarios with at least one passing run.
    """
    total = len(scenarios)
    passed = 0
    partial = 0
    failed = 0
    unreachable = 0
    inconsistent = 0
    pass_rates: list[float] = []
    unexpected_div = 0
    expected_div = 0
    tool_failures = 0

    for sc in scenarios:
        pass_rates.append(sc.consistency)
        statuses = [r.status for r in sc.runs]
        if all(s == "agent_unreachable" for s in statuses):
            unreachable += 1
        elif any(s == "pass" for s in statuses):
            passed += 1
        elif any(s == "partial_pass" for s in statuses):
            partial += 1
        else:
            failed += 1
        if sc.inconsistent:
            inconsistent += 1

        for run in sc.runs:
            for turn in run.turns:
                for tc in turn.tool_calls:
                    if tc.status != "pass":
                        tool_failures += 1
                if not turn.diverged:
                    continue
                if turn.may_diverge:
                    expected_div += 1
                else:
                    unexpected_div += 1

    pass_rate = (passed / total) if total else 0.0
    overall_consistency = (sum(pass_rates) / len(pass_rates)) if pass_rates else 0.0

    return {
        "total_scenarios": total,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "unreachable": unreachable,
        "inconsistent": inconsistent,
        "pass_rate": pass_rate,
        "overall_consistency": overall_consistency,
        "unexpected_divergences": unexpected_div,
        "expected_divergences": expected_div,
        "tool_call_failures": tool_failures,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _all_non_divergent_turn_checks_pass(turn_results: list[TurnResult]) -> bool:
    """True if every check on a turn that did NOT diverge (or was not marked
    may_diverge) passed, AND every tool_call on the turn passed.

    Spec: "Divergences on may_diverge turns are NOT counted as failures"
    applies to the LLM-evaluated text checks only. Tool call failures
    (wrong_tool, invalid_arguments, missing, unexpected_tool) always count
    because they are structural, not semantic.

    A turn whose checks failed because the agent errored / timed out DOES
    count against the run.
    """
    for t in turn_results:
        if t.status in {"agent_error", "agent_timeout", "eval_error", "empty_response"}:
            return False
        for record in t.tool_calls:
            if record.status != "pass":
                return False
        if t.may_diverge:
            continue
        for c in t.checks:
            if not c.passed:
                return False
    return True
