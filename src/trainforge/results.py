"""Helpers to assemble a :class:`RunResults` from per-scenario pieces.

Thin on purpose: pydantic models live in :mod:`trainforge.schema`, scoring
lives in :mod:`trainforge.scoring`. This module just glues them together.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from trainforge.schema import (
    RunConfig,
    RunResults,
    RunSummary,
    ScenarioResult,
    dump_results as _dump_results,
)
from trainforge.scoring import summarize


def build_run_results(
    scenarios: Iterable[ScenarioResult],
    *,
    agent_url: str,
    llm_model: str,
    runs: int,
    timeout_seconds: float,
) -> RunResults:
    """Combine scenario results + config into a finished :class:`RunResults`."""
    scenario_list = list(scenarios)
    summary = summarize(scenario_list)
    return RunResults(
        config=RunConfig(
            agent_url=agent_url,
            llm_model=llm_model,
            runs=runs,
            timeout_seconds=timeout_seconds,
        ),
        summary=RunSummary(**summary),  # type: ignore[arg-type]
        scenarios=scenario_list,
    )


def write_results(results: RunResults, path: str | Path) -> None:
    """Persist ``results`` to ``path`` as pretty-printed JSON."""
    _dump_results(results, path)
