"""TrainForge pytest plugin.

Scenario JSON files become pytest test items directly via
``pytest_collect_file`` — no shim test file required. Each scenario in a
JSON file is one pytest case; failures show the TrainForge diagnostic
(which turn, which tool, which arg mismatched) in the assertion message.

Configuration:

    # pytest.ini or pyproject.toml
    [tool.pytest.ini_options]
    trainforge_agent = "my_pkg.my_module:my_agent"
    trainforge_scenarios_dir = "tests/agent/scenarios"

Or pass on the CLI:

    pytest --trainforge-agent my_pkg.my_module:my_agent \\
           --trainforge-scenarios-dir tests/agent/scenarios

Limitations:
    Each scenario runs once. Consistency scoring (``runs > 1``) is
    CLI-only by design — pytest's "one test = one assertion" model
    doesn't map cleanly to N-run aggregation.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from trainforge.agent_resolver import AgentResolutionError, resolve_in_process_agent
from trainforge.errors import MalformedScenarioError, UnsupportedScenarioVersionError
from trainforge.llm.base import (
    LazyMissingLLMClient,
    OPENAI_COMPAT_DEFAULT_MODEL,
)
from trainforge.runner import ScenarioRunner
from trainforge.schema import Scenario, ScenarioStatus, load_scenarios
from trainforge.transport import InProcessTransport

_AGENT_CACHE_KEY = "_trainforge_agent_callable"
_LLM_CACHE_KEY = "_trainforge_llm_client"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("trainforge", "TrainForge agent regression tests")
    group.addoption(
        "--trainforge-agent",
        dest="trainforge_agent",
        default=None,
        help=(
            "In-process agent spec 'module:callable' (uvicorn-style). "
            "Required for trainforge scenarios to run."
        ),
    )
    group.addoption(
        "--trainforge-scenarios-dir",
        dest="trainforge_scenarios_dir",
        default="tests/agent/scenarios",
        help="Directory to scan for *.json scenario files.",
    )
    parser.addini(
        "trainforge_agent",
        "In-process agent spec 'module:callable' for trainforge scenarios.",
        default=None,
    )
    parser.addini(
        "trainforge_scenarios_dir",
        "Directory to scan for *.json scenario files.",
        default="tests/agent/scenarios",
    )


def _resolve_config(config: pytest.Config) -> tuple[str | None, str]:
    agent_spec = config.getoption("trainforge_agent") or config.getini("trainforge_agent")
    scenarios_dir = (
        config.getoption("trainforge_scenarios_dir")
        or config.getini("trainforge_scenarios_dir")
        or "tests/agent/scenarios"
    )
    return agent_spec, scenarios_dir


def _resolve_agent_once(config: pytest.Config) -> Any | None:
    """Cache the resolved agent callable on the config so we don't
    re-import the user's module per-scenario."""
    cached = getattr(config, _AGENT_CACHE_KEY, None)
    if cached is not None:
        return cached
    agent_spec, _ = _resolve_config(config)
    if not agent_spec:
        return None
    try:
        callable_ = resolve_in_process_agent(agent_spec)
    except AgentResolutionError as exc:
        raise pytest.UsageError(
            f"could not resolve trainforge_agent={agent_spec!r}: {exc}"
        ) from exc
    setattr(config, _AGENT_CACHE_KEY, callable_)
    return callable_


def _resolve_llm_once(config: pytest.Config):
    """Lazy LLM: real client when credentials are present, stub
    otherwise. Deterministic scenarios run without configuring an LLM
    at all; LLM-dependent scenarios fail with a clear message when the
    judge is invoked."""
    cached = getattr(config, _LLM_CACHE_KEY, None)
    if cached is not None:
        return cached

    key = os.environ.get("OPENAI_API_KEY")
    url = os.environ.get("OPENAI_API_URL")
    if not key or not url:
        client = LazyMissingLLMClient()
    else:
        from trainforge.llm.openai_compatible_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            api_key=key, base_url=url, model=OPENAI_COMPAT_DEFAULT_MODEL
        )
    setattr(config, _LLM_CACHE_KEY, client)
    return client


def _scenarios_dir_abs(config: pytest.Config) -> Path:
    _, raw = _resolve_config(config)
    return (Path(config.rootpath) / raw).resolve()


def pytest_collect_file(
    file_path: Path, parent: pytest.Collector
) -> pytest.Collector | None:
    """Collect scenario JSON files inside the configured scenarios dir.

    Anything outside the configured directory (e.g. fixtures elsewhere
    in the project that happen to be JSON) is ignored.
    """
    if file_path.suffix != ".json":
        return None
    try:
        scenarios_dir = _scenarios_dir_abs(parent.config)
    except Exception:
        return None
    if not scenarios_dir.exists():
        return None
    resolved = file_path.resolve()
    try:
        resolved.relative_to(scenarios_dir)
    except ValueError:
        return None
    return TrainForgeScenarioFile.from_parent(parent, path=file_path)


class TrainForgeScenarioFile(pytest.File):
    """One scenario JSON file = one collected file = N pytest items."""

    def collect(self):
        try:
            file = load_scenarios(str(self.path))
        except (MalformedScenarioError, UnsupportedScenarioVersionError) as exc:
            # Loud collection failure rather than silently dropping a
            # broken scenario — a typo silently skipping a test is
            # exactly what regression testing is supposed to prevent.
            raise pytest.UsageError(
                f"invalid TrainForge scenario file {self.path}: {exc}"
            ) from exc
        for scenario in file.scenarios:
            yield TrainForgeScenarioItem.from_parent(
                parent=self, name=scenario.id, scenario=scenario
            )


class TrainForgeScenarioItem(pytest.Item):
    def __init__(self, *, scenario: Scenario, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scenario = scenario

    def runtest(self) -> None:
        agent = _resolve_agent_once(self.config)
        if agent is None:
            pytest.skip(
                "no --trainforge-agent configured; pass "
                "--trainforge-agent module:callable or set trainforge_agent in pytest config"
            )
        llm = _resolve_llm_once(self.config)

        transport = InProcessTransport(agent=agent)
        runner = ScenarioRunner(agent=transport, llm=llm)
        result = asyncio.run(runner.run_scenario(self.scenario, runs=1))

        statuses = [r.status for r in result.runs]
        if all(s == ScenarioStatus.PASS for s in statuses):
            return

        failures: list[str] = self._format_failures(result)
        detail = "\n  ".join(failures) if failures else "no per-turn diagnostics"
        pytest.fail(
            f"trainforge scenario {self.scenario.id} failed "
            f"(statuses={statuses}):\n  {detail}"
        )

    def _format_failures(self, result) -> list[str]:
        failures: list[str] = []
        for run in result.runs:
            for turn in run.turns:
                if turn.exact_match is False:
                    failures.append(
                        f"turn {turn.turn_index}: exact-match failed "
                        f"(got {turn.actual_response!r})"
                    )
                for tc in turn.tool_calls:
                    if tc.status != "pass":
                        failures.append(
                            f"turn {turn.turn_index} tool[{tc.position}]: "
                            f"{tc.status} expected={tc.expected_name!r} "
                            f"got={tc.actual_name!r}"
                        )
                for check in turn.checks:
                    if not check.passed:
                        failures.append(
                            f"turn {turn.turn_index} check: {check.check!r} failed"
                        )
                for na in turn.node_assertion_results:
                    if not na.passed:
                        failures.append(
                            f"turn {turn.turn_index} node: {na.explanation}"
                        )
            for check in run.outcome.checks:
                if not check.passed:
                    failures.append(f"outcome: {check.check!r} failed")
        return failures

    def reportinfo(self) -> tuple[Path, int, str]:
        return self.path, 0, f"trainforge scenario: {self.scenario.id}"
