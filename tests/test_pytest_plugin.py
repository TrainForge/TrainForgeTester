"""End-to-end coverage for the trainforge[pytest] plugin.

Uses pytest's built-in ``pytester`` fixture to spawn isolated pytest
sessions in a tmp directory. Each test boots a fresh interpreter view of
the plugin, drops a sample scenario file + a tiny agent module, and
asserts on the captured pytest output.
"""
from __future__ import annotations

import json
import textwrap

import pytest

pytest_plugins = ["pytester"]


_AGENT_SOURCE = textwrap.dedent(
    """
    async def run(messages):
        return {"response": "pong"}
    """
).strip()


_SCENARIO = {
    "version": "2.0",
    "scenarios": [
        {
            "id": "sc-plugin-1",
            "name": "plugin smoke",
            "turns": [
                {"role": "user", "message": "ping"},
                {"role": "agent", "golden_response": "pong"},
            ],
            "expected_outcome": "ack",
            "outcome_checks": [],
        }
    ],
}


def _write_layout(pytester, *, agent_src=_AGENT_SOURCE, scenarios=_SCENARIO):
    pytester.makepyfile(my_agent=agent_src)
    pytester.mkdir("tests")
    pytester.mkdir("tests/agent")
    pytester.mkdir("tests/agent/scenarios")
    (pytester.path / "tests/agent/scenarios/hello.json").write_text(
        json.dumps(scenarios), encoding="utf-8"
    )


def test_plugin_discovers_and_runs_scenarios(pytester, monkeypatch) -> None:
    """End-to-end: scenario file + agent module → one passing pytest case."""
    _write_layout(pytester)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)

    result = pytester.runpytest(
        "-q",
        "--trainforge-agent=my_agent:run",
        "--trainforge-scenarios-dir=tests/agent/scenarios",
    )
    result.assert_outcomes(passed=1)


def test_plugin_emits_one_case_per_scenario(pytester, monkeypatch) -> None:
    """A file with two scenarios produces two parametrized pytest cases."""
    two = {
        "version": "2.0",
        "scenarios": [
            {
                "id": "sc-a",
                "name": "a",
                "turns": [
                    {"role": "user", "message": "ping"},
                    {"role": "agent", "golden_response": "pong"},
                ],
                "expected_outcome": "ack",
                "outcome_checks": [],
            },
            {
                "id": "sc-b",
                "name": "b",
                "turns": [
                    {"role": "user", "message": "ping"},
                    {"role": "agent", "golden_response": "pong"},
                ],
                "expected_outcome": "ack",
                "outcome_checks": [],
            },
        ],
    }
    _write_layout(pytester, scenarios=two)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)

    result = pytester.runpytest(
        "-q",
        "--trainforge-agent=my_agent:run",
        "--trainforge-scenarios-dir=tests/agent/scenarios",
    )
    result.assert_outcomes(passed=2)


def test_plugin_lazy_llm_when_no_keys(pytester, monkeypatch) -> None:
    """No OPENAI_API_KEY/URL → exact-match scenarios still run."""
    _write_layout(pytester)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)

    result = pytester.runpytest(
        "-q",
        "--trainforge-agent=my_agent:run",
        "--trainforge-scenarios-dir=tests/agent/scenarios",
    )
    result.assert_outcomes(passed=1)


def test_plugin_skips_when_no_agent_configured(pytester) -> None:
    _write_layout(pytester)
    # Don't pass --trainforge-agent at all.
    result = pytester.runpytest(
        "-q",
        "--trainforge-scenarios-dir=tests/agent/scenarios",
    )
    # The scenario test should be SKIPPED (not failed), because the
    # agent fixture raises pytest.skip when no spec is configured.
    result.assert_outcomes(skipped=1)


def test_plugin_malformed_scenario_raises_usage_error(pytester) -> None:
    """A typo in a scenario file MUST fail collection, not silently drop."""
    pytester.makepyfile(my_agent=_AGENT_SOURCE)
    pytester.mkdir("tests")
    pytester.mkdir("tests/agent")
    pytester.mkdir("tests/agent/scenarios")
    # Wrong version → UnsupportedScenarioVersionError → UsageError.
    bad = {"version": "9.9", "scenarios": []}
    (pytester.path / "tests/agent/scenarios/bad.json").write_text(
        json.dumps(bad), encoding="utf-8"
    )

    result = pytester.runpytest(
        "-q",
        "--trainforge-agent=my_agent:run",
        "--trainforge-scenarios-dir=tests/agent/scenarios",
    )
    # UsageError raised inside a collector → pytest exit 2 (INTERRUPTED).
    # We assert on both the exit code AND the surfaced error message so
    # the test fails loudly if either changes.
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*invalid TrainForge scenario file*"])


def test_plugin_failure_message_includes_turn_diagnostic(
    pytester, monkeypatch
) -> None:
    """An agent that returns the wrong text → pytest failure with turn N diagnostic."""
    wrong_agent = textwrap.dedent(
        """
        async def run(messages):
            return {"response": "NOT PONG"}
        """
    ).strip()
    pytester.makepyfile(my_agent=wrong_agent)
    pytester.mkdir("tests")
    pytester.mkdir("tests/agent")
    pytester.mkdir("tests/agent/scenarios")
    (pytester.path / "tests/agent/scenarios/hello.json").write_text(
        json.dumps(_SCENARIO), encoding="utf-8"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)

    result = pytester.runpytest(
        "-q",
        "--trainforge-agent=my_agent:run",
        "--trainforge-scenarios-dir=tests/agent/scenarios",
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*exact-match failed*"])
