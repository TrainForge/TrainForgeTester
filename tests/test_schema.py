"""Schema validation tests - the contract with the generator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainforge.errors import MalformedScenarioError, UnsupportedScenarioVersionError
from trainforge.schema import (
    AgentTurn,
    UserTurn,
    load_scenarios,
    parse_scenarios,
)


def test_example_scenario_from_spec_parses(example_scenarios_path: Path) -> None:
    """The example shipped in the repo is a valid v2.0 scenarios file."""
    file = load_scenarios(example_scenarios_path)
    assert file.version == "2.0"
    assert len(file.scenarios) == 1

    sc = file.scenarios[0]
    assert sc.id == "sc-001"
    assert len(sc.turns) == 6
    assert isinstance(sc.turns[0], UserTurn)
    assert isinstance(sc.turns[1], AgentTurn)
    # The weather turn (index 3) is the only may_diverge turn in the example.
    assert sc.turns[3].may_diverge is True  # type: ignore[union-attr]


def test_small_fixture_parses(small_scenarios_path: Path) -> None:
    file = load_scenarios(small_scenarios_path)
    assert len(file.scenarios) == 1
    assert file.scenarios[0].turns[0].role == "user"


def test_may_diverge_default_is_false() -> None:
    """v2.0 flips the default to deterministic exact-match."""
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "x",
                "name": "x",
                "turns": [
                    {"role": "user", "message": "hi"},
                    {"role": "agent", "golden_response": "hi back", "checks": []},
                ],
                "expected_outcome": "x",
            }
        ],
    }
    sc = parse_scenarios(raw).scenarios[0]
    agent_turn = sc.turns[1]
    assert isinstance(agent_turn, AgentTurn)
    assert agent_turn.may_diverge is False


def test_version_mismatch_raises() -> None:
    raw = {"version": "1.0", "scenarios": []}
    with pytest.raises(UnsupportedScenarioVersionError):
        parse_scenarios(raw)


def test_turns_must_start_with_user() -> None:
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "bad",
                "name": "bad",
                "turns": [
                    {"role": "agent", "golden_response": "hi", "checks": []},
                ],
                "expected_outcome": "x",
            }
        ],
    }
    with pytest.raises(MalformedScenarioError):
        parse_scenarios(raw)


def test_turns_must_alternate() -> None:
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "bad",
                "name": "bad",
                "turns": [
                    {"role": "user", "message": "hi"},
                    {"role": "user", "message": "hi again"},
                ],
                "expected_outcome": "x",
            }
        ],
    }
    with pytest.raises(MalformedScenarioError):
        parse_scenarios(raw)


def test_unknown_field_is_rejected() -> None:
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "bad",
                "name": "bad",
                "turns": [
                    {"role": "user", "message": "hi", "bogus": "x"},
                    {"role": "agent", "golden_response": "ok", "checks": []},
                ],
                "expected_outcome": "x",
            }
        ],
    }
    with pytest.raises(MalformedScenarioError):
        parse_scenarios(raw)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MalformedScenarioError):
        load_scenarios(tmp_path / "nope.json")


def test_load_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(MalformedScenarioError):
        load_scenarios(p)


def test_results_roundtrip(tmp_path: Path) -> None:
    from trainforge.schema import (
        OutcomeResult,
        OutcomeStatus,
        RunConfig,
        RunResults,
        RunSummary,
        ScenarioResult,
        ScenarioRunResult,
        ScenarioStatus,
        dump_results,
        load_results,
    )

    results = RunResults(
        config=RunConfig(
            agent_url="http://example", llm_model="fake-1", runs=1, timeout_seconds=30.0
        ),
        summary=RunSummary(
            total_scenarios=1,
            passed=1,
            partial=0,
            failed=0,
            unreachable=0,
            inconsistent=0,
            pass_rate=1.0,
            overall_consistency=1.0,
        ),
        scenarios=[
            ScenarioResult(
                scenario_id="a",
                name="a",
                runs=[
                    ScenarioRunResult(
                        run_index=0,
                        status=ScenarioStatus.PASS,
                        turns=[],
                        outcome=OutcomeResult(status=OutcomeStatus.EVALUATED, checks=[]),
                    )
                ],
                consistency=1.0,
                inconsistent=False,
            )
        ],
    )
    path = tmp_path / "r.json"
    dump_results(results, path)
    loaded = load_results(path)
    assert loaded.summary.passed == 1
    assert loaded.scenarios[0].scenario_id == "a"
    again = json.loads(path.read_text())
    assert again["summary"]["passed"] == 1
    assert again["version"] == "2.0"


def test_legacy_customer_role_is_accepted_and_normalized() -> None:
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "legacy",
                "name": "legacy",
                "turns": [
                    {"role": "customer", "message": "hi"},
                    {"role": "agent", "golden_response": "ok", "checks": []},
                ],
                "expected_outcome": "x",
            }
        ],
    }
    parsed = parse_scenarios(raw)
    assert parsed.scenarios[0].turns[0].role == "user"


def test_scenario_with_empty_turns_is_rejected() -> None:
    raw = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "bad-empty",
                "name": "bad-empty",
                "turns": [],
                "expected_outcome": "x",
            }
        ],
    }
    with pytest.raises(MalformedScenarioError):
        parse_scenarios(raw)


def test_scenarios_file_model_validator_rejects_wrong_version() -> None:
    from pydantic import ValidationError

    from trainforge.schema import ScenariosFile

    with pytest.raises(ValidationError):
        ScenariosFile.model_validate({"version": "1.0", "scenarios": []})


def test_run_results_model_validator_rejects_wrong_version() -> None:
    from pydantic import ValidationError

    from trainforge.schema import RunResults

    with pytest.raises(ValidationError):
        RunResults.model_validate(
            {
                "version": "1.0",
                "config": {
                    "agent_url": "http://example",
                    "llm_model": "m",
                    "runs": 1,
                    "timeout_seconds": 1.0,
                },
                "summary": {
                    "total_scenarios": 0,
                    "passed": 0,
                    "partial": 0,
                    "failed": 0,
                    "unreachable": 0,
                    "inconsistent": 0,
                    "pass_rate": 0.0,
                    "overall_consistency": 0.0,
                },
                "scenarios": [],
            }
        )
