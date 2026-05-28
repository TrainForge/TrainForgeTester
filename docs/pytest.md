# Pytest plugin

TrainForge scenarios can run as pytest cases. One scenario = one
parametrized pytest test. Failures show the TrainForge diagnostic
(turn N, which tool, which arg mismatched) in the assertion message.

## Install

```bash
pip install "trainforge[pytest]"
```

The `[pytest]` extra registers a pytest plugin via entry points;
`pytest` discovers it automatically next time you run.

## Use

```bash
pytest \
  --trainforge-agent my_module:my_agent \
  --trainforge-scenarios-dir tests/agent/scenarios
```

| CLI flag | `pytest.ini` key | Default | Description |
|---|---|---|---|
| `--trainforge-agent` | `trainforge_agent` | — | In-process agent spec. Same syntax as `trainforge run --agent`. |
| `--trainforge-scenarios-dir` | `trainforge_scenarios_dir` | `tests/agent/scenarios` | Directory scanned for `*.json` scenario files. |

Equivalent `pytest.ini` / `pyproject.toml`:

```toml
[tool.pytest.ini_options]
trainforge_agent = "my_module:my_agent"
trainforge_scenarios_dir = "tests/agent/scenarios"
```

## How discovery works

The plugin registers a custom file collector via
`pytest_collect_file`. Every `.json` file inside the configured
scenarios directory is parsed as a TrainForge scenario file; each
scenario in the file becomes one pytest item.

- Files outside the configured directory are ignored (so unrelated
  JSON fixtures elsewhere in the project don't get picked up).
- Files with a malformed schema or unsupported `version` cause a
  `pytest.UsageError` at collection time. The error names the file
  path and the underlying validation error.

## What you'll see

A passing scenario looks like a normal pytest pass:

```
tests/agent/scenarios/booking.json::sc-booking-1 PASSED
tests/agent/scenarios/booking.json::sc-booking-2 PASSED
```

A failing scenario shows the TF diagnostic in the assertion message:

```
trainforge scenario sc-booking-3 failed (statuses=['fail']):
  turn 2 tool[0]: wrong_tool expected='lookup_customer' got='refund_customer'
  turn 2: exact-match failed (got 'I cannot help with that.')
```

## Lazy LLM behavior

When `OPENAI_API_KEY` / `OPENAI_API_URL` aren't set, the plugin
constructs a stub LLM client that errors only when the judge is
actually invoked. Deterministic scenarios (no `may_diverge: true`,
no per-turn custom checks, no `outcome_checks`) run to completion
without any LLM key — the hackathon onboarding path.

Scenarios that DO need the judge fail with a clear "missing
credentials" message when the LLM is first called.

## Limitations

- **One run per scenario.** Consistency scoring (`runs > 1`) is
  CLI-only. Pytest's "one test = one assertion" model doesn't map
  cleanly to N-run aggregation; if you want consistency metrics, use
  `trainforge run --runs N`.
- **In-process only.** The plugin uses `InProcessTransport`; HTTP
  agents need the CLI (`trainforge run --agent-url ...`).
- **No parallel scenarios across pytest workers (`pytest-xdist`).**
  ContextVar isolation works inside a single process via
  `asyncio.gather`, but xdist's subprocess workers don't inherit
  ContextVars. Env-var-based hot-swap still works under xdist; the
  ContextVar fallback does not.

## Recommended layout

```
tests/
  agent/
    scenarios/
      onboarding.json
      refund.json
      escalation.json
  test_agent_logic.py        # your other pytest tests
  test_agent_utils.py
```

`pytest tests/` runs the agent scenarios alongside any other pytest
files you have. Failures show up in the same CI output.

## Related

- [Getting started](getting-started.md) — install + quickstart.
- [CLI reference](cli.md) — `trainforge run` for the same flow without
  pytest.
- [Agent contracts](agents.md) — what your `--trainforge-agent`
  callable looks like.
