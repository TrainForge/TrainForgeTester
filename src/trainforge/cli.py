"""``trainforge`` CLI entry points.

Five subcommands matching testing-spec-v1.md section "CLI Interface" plus
the in-process expansions:

- ``trainforge run``        - execute scenarios against an agent (HTTP or in-process).
- ``trainforge record``     - REPL capture mode: talk to your agent, write a scenario.
- ``trainforge report``     - render HTML from a results file.
- ``trainforge diff``       - compare two results files for regressions.
- ``trainforge mock-agent`` - serve a fake agent for development.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from trainforge import __version__
from trainforge.agent_resolver import AgentResolutionError, resolve_in_process_agent
from trainforge.config import override_scope
from trainforge.diff import compute_diff
from trainforge.errors import MalformedScenarioError, UnsupportedScenarioVersionError
from trainforge.llm.base import LLMClient, OPENAI_COMPAT_DEFAULT_MODEL
from trainforge.mock_agent import MODES as MOCK_MODES, MockAgentServer
from trainforge.report import render_diff_html, render_report_html
from trainforge.results import build_run_results, write_results
from trainforge.runner import ScenarioRunner
from trainforge.schema import load_results, load_scenarios
from trainforge.transport import HttpTransport, InProcessTransport, Transport

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
@click.option(
    "--agent-url",
    default=None,
    help="POST endpoint of an HTTP agent under test. Mutually exclusive with --agent.",
)
@click.option(
    "--agent",
    "agent_spec",
    default=None,
    help=(
        "In-process agent spec, uvicorn-style: 'module:callable' or "
        "'module:factory()'. Mutually exclusive with --agent-url."
    ),
)
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
@click.option(
    "--override-model",
    "override_model",
    default=None,
    help=(
        "Set TRAINFORGE_OVERRIDE_MODEL (env + ContextVar) for the duration of this run. "
        "Agents that read it can swap models without code changes."
    ),
)
@click.option(
    "--override-prompt",
    "override_prompt_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a prompt file. Its contents are set as TRAINFORGE_OVERRIDE_PROMPT "
        "(env + ContextVar) for the run."
    ),
)
@click.option("--runs", type=click.IntRange(min=1), default=1, show_default=True, help="Number of runs per scenario for consistency.")
@click.option("--timeout", "timeout_seconds", type=click.FloatRange(min=1.0), default=30.0, show_default=True, help="Per-request agent timeout in seconds.")
@click.option(
    "--parallel",
    "parallel",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help=(
        "Run up to N scenarios concurrently via asyncio.gather. Higher values "
        "shorten total wall time but may trip judge-LLM rate limits."
    ),
)
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Where to write results.json.")
def run_cmd(
    scenarios_path: str,
    agent_url: str | None,
    agent_spec: str | None,
    llm_api_url: str | None,
    llm_api_key: str | None,
    llm_model: str | None,
    override_model: str | None,
    override_prompt_path: str | None,
    runs: int,
    timeout_seconds: float,
    parallel: int,
    output_path: str,
) -> None:
    """Run scenarios against an agent (HTTP endpoint OR in-process callable)."""
    if agent_url and agent_spec:
        raise click.ClickException(
            "--agent and --agent-url are mutually exclusive; pass exactly one"
        )
    if not agent_url and not agent_spec:
        raise click.ClickException(
            "pass --agent <module:callable> or --agent-url <http endpoint>"
        )

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

    agent: Transport
    transport_label: str
    if agent_spec:
        try:
            callable_ = resolve_in_process_agent(agent_spec)
        except AgentResolutionError as exc:
            raise click.ClickException(str(exc)) from exc
        agent = InProcessTransport(
            agent=callable_,
            timeout_seconds=timeout_seconds,
            source=agent_spec,
        )
        transport_label = f"in-process {agent_spec}"
    else:
        assert agent_url is not None
        agent = HttpTransport(url=agent_url, timeout_seconds=timeout_seconds)
        transport_label = agent_url

    runner = ScenarioRunner(agent=agent, llm=llm)

    override_prompt = None
    if override_prompt_path is not None:
        try:
            from trainforge.config import load_prompt_file

            override_prompt = load_prompt_file(override_prompt_path)
        except OSError as exc:
            raise click.ClickException(
                f"could not read --override-prompt file: {exc}"
            ) from exc

    scenarios = scenarios_file.scenarios
    total_turns = sum(
        sum(1 for t in sc.turns if t.role == "agent") * runs
        for sc in scenarios
    )

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

        with override_scope(model=override_model, prompt=override_prompt):
            scenario_results = asyncio.run(
                _run_all_scenarios(
                    runner=runner,
                    scenarios=scenarios,
                    runs=runs,
                    parallel=parallel,
                    on_turn_complete=_tick,
                    bar=bar,
                )
            )

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
        agent_url=transport_label,
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

    _print_failure_highlights(scenario_results)

    click.echo(f"wrote {output_path}")

    if s.failed or s.unreachable:
        sys.exit(1)


def _print_failure_highlights(scenario_results) -> None:
    """Surface deterministic failure messages on stdout so users see them
    without opening the HTML report. Only runs when there is something to
    show (tool-call failures, exact-match mismatches, custom-check fails).
    """
    interesting: list[str] = []
    for sc in scenario_results:
        for run in sc.runs:
            for turn in run.turns:
                for tc in turn.tool_calls:
                    if tc.status == "pass":
                        continue
                    actual = tc.actual_name or "(none)"
                    if tc.actual_arguments:
                        actual += f"({_short_args(tc.actual_arguments)})"
                    interesting.append(
                        f"  ✗ {tc.status}: expected {tc.expected_name!r}, "
                        f"agent called {actual}"
                    )
                if turn.exact_match is False and turn.status == "evaluated":
                    interesting.append(
                        f"  ✗ exact_match: turn {turn.turn_index} "
                        f"actual reply did not match golden verbatim"
                    )
                for c in turn.checks:
                    if not c.passed:
                        interesting.append(f"  ✗ custom check failed: {c.check}")
                for sr in turn.standard_check_results:
                    if not sr.passed:
                        interesting.append(
                            f"  ✗ standard check {sr.id} failed"
                            + (f" -- {sr.explanation}" if sr.explanation else "")
                        )
            for c in run.outcome.checks:
                if not c.passed:
                    interesting.append(f"  ✗ outcome check failed: {c.check}")
    if interesting:
        click.echo("")
        click.echo("Failures:")
        # Cap at first 12 lines so a huge run doesn't drown the terminal.
        for line in interesting[:12]:
            click.echo(line)
        if len(interesting) > 12:
            click.echo(f"  ... (+{len(interesting) - 12} more in results.json)")


def _short_args(args: dict, max_chars: int = 60) -> str:
    """Compact args for the terminal: ``key=value, key=value``, truncated."""
    parts = []
    for k, v in args.items():
        rendered = repr(v) if isinstance(v, str) else str(v)
        parts.append(f"{k}={rendered}")
    s = ", ".join(parts)
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "…"
    return s


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
# Async orchestration of scenario runs
# ---------------------------------------------------------------------------


async def _run_all_scenarios(
    *,
    runner: ScenarioRunner,
    scenarios,
    runs: int,
    parallel: int,
    on_turn_complete: Callable[[], None],
    bar,
):
    """Run scenarios sequentially (parallel=1) or via gather+semaphore."""
    if parallel <= 1:
        results = []
        for scenario in scenarios:
            bar.label = f"{scenario.id} {_truncate(scenario.name, 48)}"
            result = await runner.run_scenario(
                scenario, runs=runs, on_turn_complete=on_turn_complete
            )
            results.append(result)
        return results

    sem = asyncio.Semaphore(parallel)

    async def _bounded(sc):
        async with sem:
            return await runner.run_scenario(
                sc, runs=runs, on_turn_complete=on_turn_complete
            )

    return await asyncio.gather(*(_bounded(sc) for sc in scenarios))


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
