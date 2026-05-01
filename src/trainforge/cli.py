"""``trainforge`` CLI entry points.

Four subcommands matching testing-spec-v1.md section "CLI Interface":

- ``trainforge run``        - execute scenarios against an agent API.
- ``trainforge report``     - render HTML from a results file.
- ``trainforge diff``       - compare two results files for regressions.
- ``trainforge mock-agent`` - serve a fake agent for development.
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from trainforge import __version__
from trainforge.agent_client import AgentClient
from trainforge.diff import compute_diff
from trainforge.errors import MalformedScenarioError, UnsupportedScenarioVersionError
from trainforge.llm.base import LLMClient, OPENAI_COMPAT_DEFAULT_MODEL
from trainforge.mock_agent import MODES as MOCK_MODES, MockAgentServer
from trainforge.report import render_diff_html, render_report_html
from trainforge.results import build_run_results, write_results
from trainforge.runner import ScenarioRunner
from trainforge.schema import load_results, load_scenarios

log = logging.getLogger("trainforge")


_LLM_CLIENT_FACTORY: Callable[..., LLMClient] | None = None
"""Module-level hook for tests: a zero-arg callable returning an ``LLMClient``.

Production code never sets this. Tests patch it via ``cli._LLM_CLIENT_FACTORY``
to avoid needing a live Anthropic key (see tests/test_cli_integration.py).
"""


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: max(n - 1, 0)].rstrip() + "\u2026"


def _overall_status_label(statuses: list[str]) -> str:
    """Compact per-scenario label for the post-bar summary."""
    if statuses and all(s == "agent_unreachable" for s in statuses):
        return "UNREACHABLE"
    if any(s == "pass" for s in statuses):
        return "PASS"
    if any(s == "partial_pass" for s in statuses):
        return "PARTIAL"
    return "FAIL"


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@click.group()
@click.version_option(__version__, prog_name="trainforge")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """TrainForge: open-source conversational-agent test runner."""
    _load_dotenv_from_tree()
    _configure_logging(verbose)


def _load_dotenv_from_tree() -> None:
    """Load the nearest ``.env`` upward from the current working directory."""
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


# ---------------------------------------------------------------------------
# trainforge run
# ---------------------------------------------------------------------------


@cli.command("run")
@click.option("--scenarios", "scenarios_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--agent-url", required=True, help="POST endpoint of the agent under test.")
@click.option(
    "--llm-api-url",
    default=None,
    envvar="OPENAI_API_URL",
    help=(
        "Base URL for an OpenAI-compatible API (for example, https://api.openai.com/v1). "
        "You can also set it via $OPENAI_API_URL."
    ),
)
@click.option(
    "--llm-api-key",
    default=None,
    envvar="OPENAI_API_KEY",
    help=(
        "API key for the OpenAI-compatible provider. You can also set it via $OPENAI_API_KEY."
    ),
)
@click.option(
    "--llm-model",
    default=None,
    help=f"Evaluator model ID. Default: {OPENAI_COMPAT_DEFAULT_MODEL!r}.",
)
@click.option("--runs", type=click.IntRange(min=1), default=1, show_default=True, help="Number of runs per scenario for consistency.")
@click.option("--timeout", "timeout_seconds", type=click.FloatRange(min=1.0), default=30.0, show_default=True, help="Per-request agent timeout in seconds.")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Where to write results.json.")
def run_cmd(
    scenarios_path: str,
    agent_url: str,
    llm_api_url: str | None,
    llm_api_key: str | None,
    llm_model: str | None,
    runs: int,
    timeout_seconds: float,
    output_path: str,
) -> None:
    """Run scenarios against an agent API (static mode)."""
    try:
        scenarios_file = load_scenarios(scenarios_path)
    except UnsupportedScenarioVersionError as exc:
        raise click.ClickException(str(exc)) from exc
    except MalformedScenarioError as exc:
        raise click.ClickException(f"malformed scenarios: {exc}") from exc

    resolved_model = llm_model or OPENAI_COMPAT_DEFAULT_MODEL
    llm = _build_llm_client(
        llm_api_url=llm_api_url,
        llm_api_key=llm_api_key,
        model=resolved_model,
    )
    agent = AgentClient(url=agent_url, timeout_seconds=timeout_seconds)
    runner = ScenarioRunner(agent=agent, llm=llm)

    scenarios = scenarios_file.scenarios
    total_turns = sum(
        sum(1 for t in sc.turns if t.role == "agent") * runs
        for sc in scenarios
    )

    scenario_results: list = []
    with click.progressbar(
        length=max(total_turns, 1),
        label=f"Running {len(scenarios)} scenario(s) x {runs} run(s)",
        show_eta=True,
        show_pos=True,
        bar_template="%(label)s  [%(bar)s]  %(info)s",
        width=30,
    ) as bar:
        def _tick() -> None:
            bar.update(1)

        for scenario in scenarios:
            bar.label = f"{scenario.id} {_truncate(scenario.name, 48)}"
            result = runner.run_scenario(
                scenario, runs=runs, on_turn_complete=_tick
            )
            scenario_results.append(result)

    click.echo("")
    for sc, result in zip(scenarios, scenario_results):
        statuses = [r.status for r in result.runs]
        overall = _overall_status_label(statuses)
        passed = sum(1 for s in statuses if s == "pass")
        click.echo(f"  {sc.id}: {overall} ({passed}/{len(statuses)} runs)")
        if runs > 1:
            tag = " INCONSISTENT" if result.inconsistent else ""
            click.echo(f"    consistency: {result.consistency * 100:.0f}%{tag}")

    results = build_run_results(
        scenario_results,
        agent_url=agent_url,
        llm_model=resolved_model,
        runs=runs,
        timeout_seconds=timeout_seconds,
    )
    write_results(results, output_path)

    s = results.summary
    click.echo("")
    click.echo(
        f"{s.passed}/{s.total_scenarios} passed"
        f" ({s.pass_rate * 100:.0f}%),"
        f" partial={s.partial}, failed={s.failed}, unreachable={s.unreachable}"
    )
    if runs > 1:
        click.echo(
            f"overall consistency: {s.overall_consistency * 100:.0f}%,"
            f" inconsistent scenarios: {s.inconsistent}"
        )
    click.echo(f"wrote {output_path}")

    if s.failed or s.unreachable:
        sys.exit(1)


# ---------------------------------------------------------------------------
# trainforge report
# ---------------------------------------------------------------------------


@cli.command("report")
@click.option("--results", "results_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True)
def report_cmd(results_path: str, output_path: str) -> None:
    """Generate an HTML report from a results file."""
    try:
        results = load_results(results_path)
    except MalformedScenarioError as exc:
        raise click.ClickException(f"invalid results file: {exc}") from exc

    html = render_report_html(results)
    Path(output_path).write_text(html, encoding="utf-8")
    click.echo(f"wrote {output_path}")


# ---------------------------------------------------------------------------
# trainforge diff
# ---------------------------------------------------------------------------


@cli.command("diff")
@click.option("--before", "before_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--after", "after_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True)
@click.option("--consistency-epsilon", type=click.FloatRange(min=0.0, max=1.0), default=0.1, show_default=True)
def diff_cmd(
    before_path: str, after_path: str, output_path: str, consistency_epsilon: float
) -> None:
    """Diff two results.json files and render a regression report."""
    before = load_results(before_path)
    after = load_results(after_path)
    report = compute_diff(
        before,
        after,
        before_path=before_path,
        after_path=after_path,
        consistency_epsilon=consistency_epsilon,
    )
    html = render_diff_html(report)
    Path(output_path).write_text(html, encoding="utf-8")
    click.echo(
        f"regressed={report.regressed_count} fixed={report.fixed_count}"
        f" wrote {output_path}"
    )
    if report.regressed_count:
        sys.exit(1)


# ---------------------------------------------------------------------------
# trainforge mock-agent
# ---------------------------------------------------------------------------


@cli.command("mock-agent")
@click.option("--scenarios", "scenarios_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--port", type=click.IntRange(min=1, max=65535), default=8080, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--mode", type=click.Choice([m.value for m in MOCK_MODES]), default="golden", show_default=True)
def mock_agent_cmd(scenarios_path: str, port: int, host: str, mode: str) -> None:
    """Serve a fake agent API for development / runner self-tests."""
    server = MockAgentServer(scenarios_path, port=port, host=host, mode=mode)
    click.echo(f"mock-agent ({mode}) listening on {server.url}")
    click.echo("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nstopping")


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------


def _build_llm_client(
    *,
    llm_api_url: str | None,
    llm_api_key: str | None,
    model: str,
) -> LLMClient:
    factory = _LLM_CLIENT_FACTORY
    if factory is not None:
        return factory(api_key=llm_api_key, model=model)

    key = _resolve_llm_api_key(llm_api_key)
    if not key:
        raise click.ClickException(
            "missing LLM API key; pass --llm-api-key or set $OPENAI_API_KEY"
        )

    base_url = _resolve_llm_api_url(llm_api_url)
    if not base_url:
        raise click.ClickException(
            "missing LLM API URL; pass --llm-api-url or set $OPENAI_API_URL"
        )

    from trainforge.llm.openai_compatible_client import OpenAICompatibleClient

    return OpenAICompatibleClient(
        api_key=key,
        base_url=base_url,
        model=model,
    )


def _resolve_llm_api_key(llm_api_key: str | None) -> str | None:
    return llm_api_key or os.environ.get("OPENAI_API_KEY")


def _resolve_llm_api_url(llm_api_url: str | None) -> str | None:
    return llm_api_url or os.environ.get("OPENAI_API_URL")


if __name__ == "__main__":  # pragma: no cover
    cli()
