"""`trainforge rescore` deterministic re-aggregation.

Models the full friction-free flow:
  1. `trainforge run --no-judge`  → results.json with pending checks.
  2. Coding agent labels each pending check in results.json.
  3. `trainforge rescore --results results.json` → official summary.

This file tests step 3 directly. Steps 1 and 2 are covered in
test_run_no_judge.py and (for step 2) by manual edits to a fixture
results file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from trainforge.cli import cli
from trainforge.runner import ScenarioRunner
from trainforge.schema import parse_scenarios, dump_results
from trainforge.transport import InProcessTransport


async def _make_no_judge_results(scenario_raw, tmp_path: Path, fake_llm):
    """Helper: run a scenario in no-judge mode and write results.json."""
    scenario = parse_scenarios(scenario_raw).scenarios[0]

    async def agent(messages):
        return {"response": "hello"}

    runner = ScenarioRunner(
        agent=InProcessTransport(agent=agent), llm=fake_llm, no_judge=True
    )
    result = await runner.run_scenario(scenario, runs=1)

    # Wrap in a full RunResults-shaped doc.
    from trainforge.results import build_run_results

    results = build_run_results(
        [result],
        agent_url="in-process",
        llm_model="none",
        runs=1,
        timeout_seconds=30.0,
    )
    out = tmp_path / "results.json"
    dump_results(results, str(out))
    return out


_SCENARIO_WITH_CHECKS = {
    "version": "2.0",
    "scenarios": [
        {
            "id": "sc-rescore-1",
            "name": "rescore",
            "turns": [
                {"role": "user", "message": "hi"},
                {
                    "role": "agent",
                    "golden_response": "hello",
                    "may_diverge": True,
                    "checks": ["agent greeted"],
                },
            ],
            "expected_outcome": "greeted",
            "outcome_checks": ["greeting returned"],
        }
    ],
}


async def test_rescore_refuses_when_checks_still_pending(
    tmp_path: Path, fake_llm
) -> None:
    out = await _make_no_judge_results(_SCENARIO_WITH_CHECKS, tmp_path, fake_llm)

    runner = CliRunner()
    result = runner.invoke(cli, ["rescore", "--results", str(out)])

    assert result.exit_code != 0
    assert "pending" in result.output.lower()


async def test_rescore_succeeds_after_labels_filled_in(
    tmp_path: Path, fake_llm, monkeypatch
) -> None:
    out = await _make_no_judge_results(_SCENARIO_WITH_CHECKS, tmp_path, fake_llm)

    # Simulate the coding agent's labeling step: flip every pending
    # check to passed=True, pending=False.
    data = json.loads(out.read_text(encoding="utf-8"))
    for sc in data["scenarios"]:
        for run in sc["runs"]:
            for turn in run["turns"]:
                for c in turn.get("checks", []):
                    c["passed"] = True
                    c["pending"] = False
                    c["explanation"] = "labeled by coding agent"
                for c in turn.get("standard_check_results", []):
                    c["passed"] = True
                    c["pending"] = False
                    c["explanation"] = "labeled by coding agent"
            for c in run["outcome"].get("checks", []):
                c["passed"] = True
                c["pending"] = False
                c["explanation"] = "labeled by coding agent"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    cli_runner = CliRunner()
    result = cli_runner.invoke(cli, ["rescore", "--results", str(out)])

    assert result.exit_code == 0, result.output
    assert "1/1 passed" in result.output


async def test_rescore_labels_can_produce_failures(
    tmp_path: Path, fake_llm
) -> None:
    """If the labeler marks any LLM-dependent check as failed, the
    deterministic rescorer must reflect that in the final verdict."""
    out = await _make_no_judge_results(_SCENARIO_WITH_CHECKS, tmp_path, fake_llm)

    data = json.loads(out.read_text(encoding="utf-8"))
    for sc in data["scenarios"]:
        for run in sc["runs"]:
            # Pass everything in the turn so the turn is clean...
            for turn in run["turns"]:
                for c in turn.get("checks", []):
                    c["passed"] = True
                    c["pending"] = False
                for c in turn.get("standard_check_results", []):
                    c["passed"] = True
                    c["pending"] = False
            # ...but mark the OUTCOME check as failed.
            for c in run["outcome"].get("checks", []):
                c["passed"] = False
                c["pending"] = False
                c["explanation"] = "agent did not actually greet"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    cli_runner = CliRunner()
    result = cli_runner.invoke(cli, ["rescore", "--results", str(out)])
    # Outcome failed → scenario FAILS → CLI exits non-zero.
    assert result.exit_code != 0
    assert "0/1 passed" in result.output


async def test_rescore_writes_to_separate_output_when_specified(
    tmp_path: Path, fake_llm
) -> None:
    out = await _make_no_judge_results(_SCENARIO_WITH_CHECKS, tmp_path, fake_llm)

    data = json.loads(out.read_text(encoding="utf-8"))
    for sc in data["scenarios"]:
        for run in sc["runs"]:
            for turn in run["turns"]:
                for c in turn.get("checks", []):
                    c["passed"] = True
                    c["pending"] = False
                for c in turn.get("standard_check_results", []):
                    c["passed"] = True
                    c["pending"] = False
            for c in run["outcome"].get("checks", []):
                c["passed"] = True
                c["pending"] = False
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    rescored = tmp_path / "rescored.json"
    cli_runner = CliRunner()
    result = cli_runner.invoke(
        cli,
        ["rescore", "--results", str(out), "--output", str(rescored)],
    )
    assert result.exit_code == 0, result.output
    assert rescored.exists()
    # Original file should be unchanged (we only rewrote scoring; the
    # labels were already there, but the summary may differ).
    assert out.exists()
