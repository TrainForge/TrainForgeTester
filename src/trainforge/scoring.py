"""Scenario PASS / PARTIAL / FAIL classification (v0.2 deterministic-first).

A run PASSes when ALL of:

1. Outcome checks all pass (LLM-judged).
2. Every agent turn is "clean":
   - All ``tool_calls`` have status ``pass`` (deterministic).
   - Either ``exact_match=True`` (deterministic, when may_diverge=False)
     OR every ``standard_check_results`` entry passes (LLM-judged binary).
   - Every per-scenario custom ``checks`` entry passes (LLM-judged binary).

A run PARTIAL_PASSes when the OUTCOME checks pass but at least one turn
is not clean. (Same intent as v0.1: agent reached the right end-state
but took a weird path.)

A run FAILs otherwise (any outcome check fails, or scenario aborted with
agent_unreachable mid-way).

Consistency:
    Per-scenario consistency = passing_runs / N
    Flag INCONSISTENT if consistency < 80% AND runs > 1.

This module is the single place these rules live. Test it directly via
:mod:`tests.test_scoring` rather than testing scoring through the runner.
"""
from __future__ import annotations

from trainforge.schema import (
    OutcomeResult,
    OutcomeStatus,
    ScenarioResult,
    ScenarioRunResult,
    ScenarioStatus,
    ToolCallStatus,
    TurnResult,
    TurnStatus,
)


INCONSISTENCY_THRESHOLD = 0.80
_TERMINAL_TURN_FAILURES = {
    TurnStatus.AGENT_ERROR,
    TurnStatus.AGENT_TIMEOUT,
    TurnStatus.EVAL_ERROR,
    TurnStatus.EMPTY_RESPONSE,
}


def classify_scenario_run(
    turn_results: list[TurnResult], outcome: OutcomeResult
) -> ScenarioStatus:
    """Map one run's turn + outcome results to a :data:`ScenarioStatus`."""
    if outcome.status == OutcomeStatus.AGENT_UNREACHABLE:
        return ScenarioStatus.AGENT_UNREACHABLE

    outcome_ok = outcome.status == OutcomeStatus.EVALUATED and all(c.passed for c in outcome.checks)
    turns_clean = all(_turn_is_clean(t) for t in turn_results)

    if outcome_ok and turns_clean:
        return ScenarioStatus.PASS
    if outcome_ok and not turns_clean:
        return ScenarioStatus.PARTIAL_PASS
    return ScenarioStatus.FAIL


def aggregate_consistency(runs: list[ScenarioRunResult]) -> tuple[float, bool]:
    """Return ``(pass_rate, inconsistent?)`` across ``runs``.

    ``inconsistent`` only meaningful with multiple runs; single-run scenarios
    are never flagged inconsistent.
    """
    if not runs:
        return 0.0, False
    passed = sum(1 for r in runs if r.status == ScenarioStatus.PASS)
    rate = passed / len(runs)
    inconsistent = len(runs) > 1 and rate < INCONSISTENCY_THRESHOLD
    return rate, inconsistent


def summarize(scenarios: list[ScenarioResult]) -> dict[str, int | float]:
    """Build a dict suitable for :class:`trainforge.schema.RunSummary`."""
    total = len(scenarios)
    passed = 0
    partial = 0
    failed = 0
    unreachable = 0
    inconsistent = 0
    pass_rates: list[float] = []
    tool_failures = 0
    exact_match_failures = 0
    standard_check_failures = 0
    custom_check_failures = 0

    for sc in scenarios:
        pass_rates.append(sc.consistency)
        statuses = [r.status for r in sc.runs]
        if all(s == ScenarioStatus.AGENT_UNREACHABLE for s in statuses):
            unreachable += 1
        elif any(s == ScenarioStatus.PASS for s in statuses):
            passed += 1
        elif any(s == ScenarioStatus.PARTIAL_PASS for s in statuses):
            partial += 1
        else:
            failed += 1
        if sc.inconsistent:
            inconsistent += 1

        for run in sc.runs:
            for turn in run.turns:
                for tc in turn.tool_calls:
                    if tc.status != ToolCallStatus.PASS:
                        tool_failures += 1
                if turn.exact_match is False:
                    exact_match_failures += 1
                for sr in turn.standard_check_results:
                    if not sr.passed:
                        standard_check_failures += 1
                for c in turn.checks:
                    if not c.passed:
                        custom_check_failures += 1

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
        "tool_call_failures": tool_failures,
        "exact_match_failures": exact_match_failures,
        "standard_check_failures": standard_check_failures,
        "custom_check_failures": custom_check_failures,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _turn_is_clean(turn: TurnResult) -> bool:
    """A turn is clean iff every check on it passed.

    Failure conditions:
    - Turn-level error status (agent_error / agent_timeout / eval_error /
      empty_response) immediately disqualifies.
    - Any tool_call with status != 'pass'.
    - On may_diverge=False turns: exact_match is False.
    - On may_diverge=True turns: any standard_check_results entry not passed.
    - Any custom check entry not passed (regardless of may_diverge).
    """
    if turn.status in _TERMINAL_TURN_FAILURES:
        return False
    for record in turn.tool_calls:
        if record.status != ToolCallStatus.PASS:
            return False
    if turn.may_diverge:
        for sr in turn.standard_check_results:
            if not sr.passed:
                return False
    else:
        if turn.exact_match is not True:
            return False
    for c in turn.checks:
        if not c.passed:
            return False
    return True
