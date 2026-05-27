from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from trainforge import cli as cli_module
from trainforge.cli import (
    _build_llm_client,
    _overall_status_label,
    _resolve_llm_api_key,
    _resolve_llm_api_url,
    _truncate,
    cli,
)


def test_truncate_and_status_labels() -> None:
    assert _truncate("abc", 10) == "abc"
    assert _truncate("abcdef", 4).endswith("…")
    assert _overall_status_label(["agent_unreachable"]) == "UNREACHABLE"
    assert _overall_status_label(["partial_pass"]) == "PARTIAL"
    assert _overall_status_label(["fail"]) == "FAIL"


def test_resolve_llm_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_URL", "https://api.example/v1")
    assert _resolve_llm_api_key(None) == "k"
    assert _resolve_llm_api_url(None) == "https://api.example/v1"


def test_build_llm_client_lazy_when_no_credentials(monkeypatch) -> None:
    """No keys → return a stub that lets deterministic scenarios run.
    The error is deferred to first ``.complete()`` so users can run
    scenarios that don't need the judge without configuring an LLM.
    """
    monkeypatch.setattr(cli_module, "_LLM_CLIENT_FACTORY", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)

    client = _build_llm_client(llm_api_url=None, llm_api_key=None, model="m")
    # Construction succeeds; calling complete() is what raises.
    assert client.model == "m"
    with pytest.raises(Exception):
        client.complete("sys", "user")


def test_build_llm_client_lazy_when_only_key_present(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "_LLM_CLIENT_FACTORY", None)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_API_URL", raising=False)

    client = _build_llm_client(llm_api_url=None, llm_api_key=None, model="m")
    with pytest.raises(Exception):
        client.complete("sys", "user")


def test_build_llm_client_uses_factory(monkeypatch) -> None:
    class _X:
        pass

    monkeypatch.setattr(cli_module, "_LLM_CLIENT_FACTORY", lambda api_key, model: _X())
    out = _build_llm_client(llm_api_url=None, llm_api_key=None, model="m")
    assert isinstance(out, _X)


def test_report_invalid_results_file_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["report", "--results", str(bad), "--output", str(tmp_path / "x.html")],
    )
    assert result.exit_code != 0
    assert "invalid results file" in result.output


def test_run_with_unsupported_scenario_version_errors(tmp_path: Path) -> None:
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        '{"version":"1.0","scenarios":[]}',
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--scenarios",
            str(scenarios),
            "--agent-url",
            "http://unused",
            "--output",
            str(tmp_path / "r.json"),
        ],
    )
    assert result.exit_code != 0
    assert "unsupported scenarios version" in result.output


def test_run_with_malformed_scenario_file_errors(tmp_path: Path) -> None:
    scenarios = tmp_path / "bad_scenarios.json"
    scenarios.write_text("not json", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--scenarios",
            str(scenarios),
            "--agent-url",
            "http://unused",
            "--llm-api-key",
            "x",
            "--llm-api-url",
            "https://api.example/v1",
            "--output",
            str(tmp_path / "r.json"),
        ],
    )
    assert result.exit_code != 0
    assert "malformed scenarios" in result.output


def test_build_llm_client_default_branch(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "_LLM_CLIENT_FACTORY", None)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_URL", "https://api.example/v1")

    class _Client:
        def __init__(self, api_key: str, base_url: str, model: str) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.model = model

    monkeypatch.setattr(
        "trainforge.llm.openai_compatible_client.OpenAICompatibleClient",
        _Client,
    )

    c = _build_llm_client(llm_api_url=None, llm_api_key=None, model="m")
    assert isinstance(c, _Client)
    assert c.api_key == "k"
    assert c.base_url == "https://api.example/v1"
    assert c.model == "m"


def test_mock_agent_cmd_handles_keyboard_interrupt(monkeypatch, small_scenarios_path: Path) -> None:
    class _FakeServer:
        def __init__(self, scenarios_path: str, port: int, host: str, mode: str) -> None:
            self.url = "http://127.0.0.1:9999/chat"

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "MockAgentServer", _FakeServer)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "mock-agent",
            "--scenarios",
            str(small_scenarios_path),
            "--port",
            "9999",
            "--host",
            "127.0.0.1",
            "--mode",
            "golden",
        ],
    )
    assert result.exit_code == 0
    assert "stopping" in result.output

