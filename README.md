# TrainForge

**Deterministic-first regression testing for AI agents.**

Scenarios run against your agent (in-process Python callable or HTTP
endpoint). Tool calls and scripted text are checked by Python equality;
free-form text is judged by 20 binary NLP-consistency questions per turn.
No 0-to-1 quality scores. No judge drift. When CI says pass, it
actually passed.

![TrainForge CLI catching an unsafe tool call](docs/assets/cli-run.gif)

## Install

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev]"
```

## 60-second quickstart

```bash
trainforge run \
  --agent examples.quickstart.agent:run \
  --scenarios examples/quickstart/scenarios/hello.json \
  --output results.json
```

You should see `1/1 passed (100%)`. The scenario uses exact-match with
no custom checks, so no LLM key is needed.

Replace `examples.quickstart.agent:run` with your own
`your_module:your_function` to test a real agent. The callable shape is:

```python
async def run(messages: list[dict]) -> dict:
    return {"response": "...", "tool_calls": [...]}
```

See [Getting started](docs/getting-started.md) for the full
walkthrough, including capture mode, hot-swap A/B testing, and the
HTTP transport.

## Why deterministic-first

Most agent-testing tools score quality on a 0-1 scale using an LLM
judge that drifts between runs on the same input. TrainForge replaces
the score with binary contracts you wrote. Either every contract
passed, or you can point at exactly which line broke.

See [Concepts](docs/concepts.md) for the full positioning, golden
injection, and the 20 standard NLP-consistency checks.

## Documentation

| Page | What's in it |
|---|---|
| [Getting started](docs/getting-started.md) | Install + both quickstarts (in-process and HTTP), capture mode, hot-swap. |
| [Concepts](docs/concepts.md) | Why deterministic-first, golden injection, the 20 NLP checks, the compact LLM wire format. |
| [Scenarios](docs/scenarios.md) | Scenario JSON schema: turns, tool loops, node assertions, `may_diverge` modes. |
| [Agent contracts](docs/agents.md) | In-process callable shape, HTTP wire format, tool round-trips, error mapping. |
| [CLI reference](docs/cli.md) | Every subcommand (`run`, `record`, `report`, `diff`, `mock-agent`) and flag. |
| [Pytest plugin](docs/pytest.md) | `pip install "trainforge[pytest]"` — scenarios as pytest cases. |
| [Migration from v1.x](docs/migration.md) | What to change in v1.0 scenarios to load under 2.0. |

## Developing

```bash
pytest                                # unit + integration tests
pytest --cov=trainforge --cov-report=term-missing
```

Tests inject a stub LLM via `trainforge.cli._LLM_CLIENT_FACTORY`; no
external LLM is contacted in CI.

## License

Apache 2.0
