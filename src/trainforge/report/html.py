"""Render HTML reports from :class:`RunResults` and diff results.

Single Jinja2 environment serves both the main report (``report.html.j2``)
and the regression diff (``diff.html.j2``). The templates live next to this
module under ``templates/``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from trainforge.schema import RunResults, ScenarioRunResult, TurnResult

if TYPE_CHECKING:  # pragma: no cover
    from trainforge.diff import DiffReport


@lru_cache(maxsize=1)
def _env() -> Environment:
    env = Environment(
        loader=PackageLoader("trainforge.report", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = _pct
    env.filters["status_label"] = _status_label
    env.filters["plain"] = _plain
    return env


def _plain(obj):
    """Recursively turn pydantic models into plain JSON-serializable types."""
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _status_label(status: str) -> str:
    return {
        "pass": "PASS",
        "partial_pass": "PARTIAL",
        "fail": "FAIL",
        "agent_unreachable": "UNREACHABLE",
        "evaluated": "OK",
        "agent_error": "AGENT ERROR",
        "agent_timeout": "AGENT TIMEOUT",
        "eval_error": "EVAL ERROR",
        "empty_response": "EMPTY RESPONSE",
    }.get(status, status.upper())


def render_report_html(results: RunResults) -> str:
    """Render the primary run report to an HTML string."""
    template = _env().get_template("report.html.j2")
    failures = _collect_failures(results)
    standard_failures = _collect_standard_failures(results)
    return template.render(
        results=results,
        summary=results.summary,
        config=results.config,
        scenarios=results.scenarios,
        failures=failures,
        standard_failures=standard_failures,
    )


def render_diff_html(diff: "DiffReport") -> str:
    """Render a regression diff report to an HTML string."""
    template = _env().get_template("diff.html.j2")
    return template.render(diff=diff)


# ---------------------------------------------------------------------------
# Aggregations used by the template
# ---------------------------------------------------------------------------


def _collect_standard_failures(results: RunResults) -> list[dict]:
    """Flat list of every standard-check failure across the whole run.

    Useful for "which NLP-consistency axes most often broke?" view.
    """
    out: list[dict] = []
    for sc in results.scenarios:
        for run in sc.runs:
            for turn in run.turns:
                for sr in turn.standard_check_results:
                    if sr.passed:
                        continue
                    out.append(
                        {
                            "scenario_id": sc.scenario_id,
                            "scenario_name": sc.name,
                            "run_index": run.run_index,
                            "turn_index": turn.turn_index,
                            "id": sr.id,
                            "question": sr.question,
                            "explanation": sr.explanation,
                        }
                    )
    return out


def _collect_failures(results: RunResults) -> list[dict]:
    failures: list[dict] = []
    for sc in results.scenarios:
        for run in sc.runs:
            if run.status == "pass":
                continue
            failures.append(
                {
                    "scenario_id": sc.scenario_id,
                    "scenario_name": sc.name,
                    "run_index": run.run_index,
                    "status": run.status,
                    "first_failing": _first_failing_turn(run),
                    "outcome_failures": [c for c in run.outcome.checks if not c.passed],
                    "error": run.error,
                }
            )
    return failures


def _first_failing_turn(run: ScenarioRunResult) -> TurnResult | None:
    for t in run.turns:
        if t.status != "evaluated":
            return t
        if any(tc.status != "pass" for tc in t.tool_calls):
            return t
        if t.exact_match is False:
            return t
        if any(not sr.passed for sr in t.standard_check_results):
            return t
        if any(not c.passed for c in t.checks):
            return t
    return None
