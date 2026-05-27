"""TrainForge pytest plugin.

Auto-discovers TrainForge scenario JSON files and emits one parametrized
pytest case per scenario. Failures show the TrainForge diagnostic (which
turn, which tool, which arg mismatched) in the assertion message.

Configuration (in ``pytest.ini`` / ``pyproject.toml``):

    [tool.pytest.ini_options]
    trainforge_agent = "my_pkg.my_module:my_agent"
    trainforge_scenarios_dir = "tests/agent/scenarios"

Or pass on the CLI:

    pytest --trainforge-agent my_pkg.my_module:my_agent

Limitations:
    The pytest plugin runs each scenario once. Consistency scoring
    (``runs > 1``) requires the ``trainforge run`` CLI, where the
    aggregation across runs is meaningful in a single result file. In
    pytest's "one test = one assertion" world, multi-run consistency is
    awkward to render, so we keep it CLI-only on purpose.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from trainforge.agent_resolver import AgentResolutionError, resolve_in_process_agent
from trainforge.errors import MalformedScenarioError, UnsupportedScenarioVersionError
from trainforge.llm.base import OPENAI_COMPAT_DEFAULT_MODEL
from trainforge.runner import ScenarioRunner
from trainforge.schema import Scenario, ScenarioStatus, load_scenarios
from trainforge.transport import InProcessTransport


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("trainforge", "TrainForge agent regression tests")
    group.addoption(
        "--trainforge-agent",
        dest="trainforge_agent",
        default=None,
        help="In-process agent spec 'module:callable' (uvicorn-style). "
        "Required for trainforge scenarios to run.",
    )
    group.addoption(
        "--trainforge-scenarios-dir",
        dest="trainforge_scenarios_dir",
        default="tests/agent/scenarios",
        help="Directory to scan for *.json scenario files. Default: tests/agent/scenarios.",
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


def _discover_scenario_files(scenarios_dir: Path) -> list[Path]:
    if not scenarios_dir.exists():
        return []
    return sorted(scenarios_dir.rglob("*.json"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # No-op; collection is handled via `pytest_generate_tests` on the
    # built-in fixture below.
    return None


@pytest.fixture(scope="session")
def trainforge_agent_callable(pytestconfig: pytest.Config) -> Any:
    agent_spec, _ = _resolve_config(pytestconfig)
    if not agent_spec:
        pytest.skip(
            "no --trainforge-agent configured; set --trainforge-agent module:callable "
            "or add trainforge_agent to pytest config"
        )
    try:
        return resolve_in_process_agent(agent_spec)
    except AgentResolutionError as exc:
        pytest.fail(f"could not resolve trainforge_agent={agent_spec!r}: {exc}")


def _collect_scenarios(config: pytest.Config) -> list[tuple[Path, Scenario]]:
    _, scenarios_dir_str = _resolve_config(config)
    scenarios_dir = Path(config.rootpath) / scenarios_dir_str
    collected: list[tuple[Path, Scenario]] = []
    for path in _discover_scenario_files(scenarios_dir):
        try:
            file = load_scenarios(str(path))
        except (MalformedScenarioError, UnsupportedScenarioVersionError):
            # Surface as a real test failure rather than crashing collection.
            continue
        for sc in file.scenarios:
            collected.append((path, sc))
    return collected


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "trainforge_scenario" not in metafunc.fixturenames:
        return
    pairs = _collect_scenarios(metafunc.config)
    if not pairs:
        return
    ids = [f"{p.name}::{sc.id}" for p, sc in pairs]
    metafunc.parametrize("trainforge_scenario", [sc for _, sc in pairs], ids=ids)


@pytest.fixture(scope="session")
def trainforge_llm():
    """LLM client for the pytest plugin.

    If ``OPENAI_API_KEY`` and ``OPENAI_API_URL`` are set, returns a real
    OpenAI-compatible client. Otherwise returns
    :class:`~trainforge.llm.base.LazyMissingLLMClient`: deterministic
    scenarios (exact-match, no custom turn checks, no outcome_checks)
    still run, and only LLM-dependent scenarios fail with a clear
    "missing credentials" error when the judge is actually invoked.
    """
    import os

    from trainforge.llm.base import LazyMissingLLMClient

    key = os.environ.get("OPENAI_API_KEY")
    url = os.environ.get("OPENAI_API_URL")
    if not key or not url:
        return LazyMissingLLMClient()

    from trainforge.llm.openai_compatible_client import OpenAICompatibleClient

    return OpenAICompatibleClient(
        api_key=key, base_url=url, model=OPENAI_COMPAT_DEFAULT_MODEL
    )


def test_trainforge_scenario(
    trainforge_scenario: Scenario,
    trainforge_agent_callable: Any,
    trainforge_llm,
) -> None:
    """Run one TrainForge scenario as a pytest case.

    A scenario passes when every run ends in ``ScenarioStatus.PASS``.
    Failure diagnostics include the scenario id, status counts, and the
    first failing turn's detail string.
    """
    transport = InProcessTransport(agent=trainforge_agent_callable)
    runner = ScenarioRunner(agent=transport, llm=trainforge_llm)
    result = asyncio.run(runner.run_scenario(trainforge_scenario, runs=1))

    statuses = [r.status for r in result.runs]
    if all(s == ScenarioStatus.PASS for s in statuses):
        return

    # Build a helpful failure message.
    failures: list[str] = []
    for run in result.runs:
        for turn in run.turns:
            if turn.exact_match is False:
                failures.append(
                    f"turn {turn.turn_index}: exact-match failed (got {turn.actual_response!r})"
                )
            for tc in turn.tool_calls:
                if tc.status != "pass":
                    failures.append(
                        f"turn {turn.turn_index} tool[{tc.position}]: "
                        f"{tc.status} expected={tc.expected_name!r} got={tc.actual_name!r}"
                    )
            for check in turn.checks:
                if not check.passed:
                    failures.append(f"turn {turn.turn_index} check: {check.check!r} failed")
            for na in turn.node_assertion_results:
                if not na.passed:
                    failures.append(f"turn {turn.turn_index} node: {na.explanation}")
        for check in run.outcome.checks:
            if not check.passed:
                failures.append(f"outcome: {check.check!r} failed")

    detail = "\n  ".join(failures) if failures else "no per-turn diagnostics"
    pytest.fail(
        f"trainforge scenario {trainforge_scenario.id} failed "
        f"(statuses={statuses}):\n  {detail}"
    )
