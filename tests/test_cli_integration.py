"""End-to-end CLI tests: runner + mock-agent.

Boots ``trainforge mock-agent`` in-process and exercises the full pipeline
- ``trainforge run`` against it with a stubbed LLM,
- ``trainforge report`` from the written results,
- ``trainforge diff`` across two runs.

No live LLM calls. The LLM is injected via the ``trainforge.cli._LLM_CLIENT_FACTORY``
test hook so we don't need an Anthropic key.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from click.testing import CliRunner

from trainforge import cli as cli_module
from trainforge.cli import cli
from trainforge.mock_agent import MockAgentServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def golden_server(small_scenarios_path: Path):
    server = MockAgentServer(str(small_scenarios_path), port=_free_port(), mode="golden")
    server.start()
    yield server
    server.stop()


@pytest.fixture
def diverge_server(small_scenarios_path: Path):
    server = MockAgentServer(str(small_scenarios_path), port=_free_port(), mode="diverge", seed=1)
    server.start()
    yield server
    server.stop()


# ---------------------------------------------------------------------------
# Injectable stub LLMs
# ---------------------------------------------------------------------------


def _count_questions(user_prompt: str) -> int:
    """Both turn-eval and outcome-eval prompts include a numbered list. Count
    the questions by counting the lines that match ``N. ``."""
    import re as _re

    matches = _re.findall(r"^\d+\. ", user_prompt, flags=_re.MULTILINE)
    return len(matches)


class _AutoPassLLM:
    """Stub that returns 'all pass' in the v0.2 compact format.

    Inspects the prompt to figure out how many questions to answer, then
    returns ``{"r": [1] * n, "f": {}}``.
    """

    model = "fake-pass"

    def complete(self, system: str, user: str) -> str:
        n = _count_questions(user) or 1
        return json.dumps({"r": [1] * n, "f": {}})


class _AutoFailLLM:
    """Stub that fails every question in the v0.2 compact format."""

    model = "fake-fail"

    def complete(self, system: str, user: str) -> str:
        n = _count_questions(user) or 1
        failures = {str(i): "stubbed failure" for i in range(1, n + 1)}
        return json.dumps({"r": [0] * n, "f": failures})


@pytest.fixture
def pass_all_llm(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "_LLM_CLIENT_FACTORY",
        lambda api_key, model: _AutoPassLLM(),
    )


@pytest.fixture
def fail_all_llm(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "_LLM_CLIENT_FACTORY",
        lambda api_key, model: _AutoFailLLM(),
    )


# ---------------------------------------------------------------------------
# trainforge run - happy path
# ---------------------------------------------------------------------------


def test_run_against_golden_mock_all_pass(
    golden_server, pass_all_llm, small_scenarios_path: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    out_path = tmp_path / "results.json"
    result = runner.invoke(
        cli,
        [
            "run",
            "--scenarios",
            str(small_scenarios_path),
            "--agent-url",
            golden_server.url,
            "--llm-api-key",
            "unused",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    assert data["version"] == "2.0"
    assert data["summary"]["total_scenarios"] == 1
    assert data["summary"]["passed"] == 1
    assert data["summary"]["failed"] == 0
    assert data["scenarios"][0]["runs"][0]["status"] == "pass"


def test_run_then_report_writes_html(
    golden_server, pass_all_llm, small_scenarios_path: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    results_path = tmp_path / "results.json"
    html_path = tmp_path / "report.html"

    run_result = runner.invoke(
        cli,
        [
            "run",
            "--scenarios",
            str(small_scenarios_path),
            "--agent-url",
            golden_server.url,
            "--llm-api-key",
            "x",
            "--output",
            str(results_path),
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    report_result = runner.invoke(
        cli,
        ["report", "--results", str(results_path), "--output", str(html_path)],
    )
    assert report_result.exit_code == 0, report_result.output

    html = html_path.read_text()
    assert "TrainForge Report" in html
    assert "Simple booking" in html
    assert "PASS" in html


def test_run_multi_run_consistency(
    golden_server, pass_all_llm, small_scenarios_path: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    out_path = tmp_path / "results.json"
    result = runner.invoke(
        cli,
        [
            "run",
            "--scenarios",
            str(small_scenarios_path),
            "--agent-url",
            golden_server.url,
            "--llm-api-key",
            "x",
            "--runs",
            "3",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    assert data["config"]["runs"] == 3
    assert len(data["scenarios"][0]["runs"]) == 3
    assert data["scenarios"][0]["consistency"] == 1.0
    assert data["scenarios"][0]["inconsistent"] is False


# ---------------------------------------------------------------------------
# trainforge run - divergence mode
# ---------------------------------------------------------------------------


def test_run_against_diverge_mock_with_failing_llm_reports_failure(
    diverge_server, fail_all_llm, small_scenarios_path: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    out_path = tmp_path / "results.json"
    result = runner.invoke(
        cli,
        [
            "run",
            "--scenarios",
            str(small_scenarios_path),
            "--agent-url",
            diverge_server.url,
            "--llm-api-key",
            "x",
            "--output",
            str(out_path),
        ],
    )
    # Runner exits 1 when there are failures.
    assert result.exit_code == 1, result.output
    data = json.loads(out_path.read_text())
    assert data["summary"]["failed"] == 1
    assert data["scenarios"][0]["runs"][0]["status"] == "fail"


# ---------------------------------------------------------------------------
# trainforge run - error mode
# ---------------------------------------------------------------------------


def test_run_against_error_mock_marks_turn_errors(
    pass_all_llm, small_scenarios_path: Path, tmp_path: Path
) -> None:
    """Spec Error Handling table: agent 500 -> mark turn agent_error, continue."""
    server = MockAgentServer(
        str(small_scenarios_path),
        port=_free_port(),
        mode="error",
        error_rate=1.0,
        timeout_rate=0.0,
        seed=0,
    )
    server.start()
    try:
        runner = CliRunner()
        out_path = tmp_path / "results.json"
        result = runner.invoke(
            cli,
            [
                "run",
                "--scenarios",
                str(small_scenarios_path),
                "--agent-url",
                server.url,
                "--llm-api-key",
                "x",
                "--timeout",
                "2",
                "--output",
                str(out_path),
            ],
        )
        # Exit code depends on the stubbed outcome evaluator; the invariant
        # we care about is that each turn was correctly marked agent_error.
        data = json.loads(out_path.read_text())
        turn_statuses = [
            t["status"] for t in data["scenarios"][0]["runs"][0]["turns"]
        ]
        assert turn_statuses == ["agent_error", "agent_error"]
        # Each errored turn should produce no successful checks.
        for turn in data["scenarios"][0]["runs"][0]["turns"]:
            assert turn["actual_response"] == ""
            assert turn["error"] is not None
        # And the run cannot be a full PASS (turn errors block that).
        assert data["scenarios"][0]["runs"][0]["status"] != "pass"
        assert result.exit_code in (0, 1)
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# trainforge diff
# ---------------------------------------------------------------------------


def test_diff_end_to_end(
    golden_server,
    diverge_server,
    small_scenarios_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    diff_html = tmp_path / "diff.html"

    monkeypatch.setattr(
        cli_module, "_LLM_CLIENT_FACTORY", lambda api_key, model: _AutoPassLLM()
    )
    r1 = runner.invoke(
        cli,
        [
            "run",
            "--scenarios",
            str(small_scenarios_path),
            "--agent-url",
            golden_server.url,
            "--llm-api-key",
            "x",
            "--output",
            str(before),
        ],
    )
    assert r1.exit_code == 0, r1.output

    monkeypatch.setattr(
        cli_module, "_LLM_CLIENT_FACTORY", lambda api_key, model: _AutoFailLLM()
    )
    r2 = runner.invoke(
        cli,
        [
            "run",
            "--scenarios",
            str(small_scenarios_path),
            "--agent-url",
            diverge_server.url,
            "--llm-api-key",
            "x",
            "--output",
            str(after),
        ],
    )
    assert r2.exit_code == 1, r2.output

    diff_result = runner.invoke(
        cli,
        [
            "diff",
            "--before",
            str(before),
            "--after",
            str(after),
            "--output",
            str(diff_html),
        ],
    )
    assert diff_result.exit_code == 1, diff_result.output  # regression present
    html = diff_html.read_text()
    assert "Regression" in html
    assert "Simple booking" in html


# ---------------------------------------------------------------------------
# BYO-key validation
# ---------------------------------------------------------------------------


def test_run_without_api_key_and_without_factory_errors(
    small_scenarios_path: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli_module, "_LLM_CLIENT_FACTORY", None)
    monkeypatch.setattr(cli_module, "find_dotenv", lambda usecwd=True: "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--scenarios",
            str(small_scenarios_path),
            "--agent-url",
            "http://unused",
            "--output",
            str(tmp_path / "r.json"),
        ],
    )
    assert result.exit_code != 0
    assert "LLM API key" in result.output


# ---------------------------------------------------------------------------
# v1.1 tool_loops end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def tools_golden_server(tools_scenarios_path: Path):
    server = MockAgentServer(str(tools_scenarios_path), port=_free_port(), mode="golden")
    server.start()
    yield server
    server.stop()


def test_run_with_tool_loops_all_pass(
    tools_golden_server, pass_all_llm, tools_scenarios_path: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    out_path = tmp_path / "results.json"
    result = runner.invoke(
        cli,
        [
            "run",
            "--scenarios",
            str(tools_scenarios_path),
            "--agent-url",
            tools_golden_server.url,
            "--llm-api-key",
            "x",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    assert data["summary"]["passed"] == 1
    assert data["summary"]["tool_call_failures"] == 0
    turn = data["scenarios"][0]["runs"][0]["turns"][0]
    statuses = [tc["status"] for tc in turn["tool_calls"]]
    assert statuses == ["pass", "pass", "pass"]


def test_run_with_tool_loops_diverge_records_tool_failures(
    tools_scenarios_path: Path, pass_all_llm, tmp_path: Path
) -> None:
    """Diverge-mode mock intentionally calls wrong tools / drops args; the
    runner must record the failures AND keep the conversation on-track via
    golden injection."""
    server = MockAgentServer(
        str(tools_scenarios_path),
        port=_free_port(),
        mode="diverge",
        seed=1,
    )
    server.start()
    try:
        runner = CliRunner()
        out_path = tmp_path / "results.json"
        result = runner.invoke(
            cli,
            [
                "run",
                "--scenarios",
                str(tools_scenarios_path),
                "--agent-url",
                server.url,
                "--llm-api-key",
                "x",
                "--output",
                str(out_path),
            ],
        )
        data = json.loads(out_path.read_text())
        assert data["summary"]["tool_call_failures"] > 0
        assert result.exit_code in (0, 1)
    finally:
        server.stop()
