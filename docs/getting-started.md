# Getting started

Two paths, depending on what your agent looks like today.

- **In-process Python callable** (LangChain, CrewAI, LangGraph, OpenAI
  Agents SDK, plain async function in a notebook): start [here](#in-process-60-seconds).
  No HTTP server needed.
- **HTTP agent** (already deployed as a service): start [here](#http-against-the-mock-agent).

After either, see:

- [Capture mode](#capture-mode-talk-to-your-agent-write-a-scenario) —
  generate a scenario from a live conversation.
- [Hot-swap](#hot-swap-ab-prompt-and-model-changes) — re-run the same
  scenarios against a different model or prompt and diff the result.

## Install

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the `trainforge` console script.

## In-process: 60 seconds

The repo ships a toy agent and one scenario so you can confirm the
in-process transport works without an LLM key.

```bash
trainforge run \
  --agent examples.quickstart.agent:run \
  --scenarios examples/quickstart/scenarios/hello.json \
  --output results.json
```

You should see `1/1 passed (100%)`. The scenario uses
`may_diverge: false` with no custom checks, so TrainForge does an
exact-match comparison and never invokes the LLM judge.

To test your own agent, replace `examples.quickstart.agent:run` with
`your_module:your_function`. The callable signature is:

```python
async def run(messages: list[dict]) -> dict:
    return {"response": "...", "tool_calls": [...]}
```

Sync callables (`def run(...)`) are accepted; they're wrapped via
`asyncio.to_thread`. Factory pattern works too:
`--agent my_module:make_agent()` calls `make_agent()` and uses the
returned callable.

See [Agent contracts](agents.md) for the full shape.

## HTTP: against the mock agent

For agents already running as an HTTP service, point `--agent-url` at
the endpoint. The repo ships a built-in mock you can use as a target
in development:

```bash
# Terminal 1 — mock agent returns the golden responses verbatim
trainforge mock-agent \
  --scenarios scenarios/example_restaurant_booking.json \
  --port 8080

# Terminal 2 — run against it (judge LLM is needed because the example
# scenario has may_diverge=true turns and outcome_checks)
export OPENAI_API_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
trainforge run \
  --scenarios scenarios/example_restaurant_booking.json \
  --agent-url http://localhost:8080/chat \
  --output results.json

trainforge report --results results.json --output report.html
open report.html
```

See [Agent contracts](agents.md) for the HTTP wire format your real
agent must speak.

## Example failure: unsafe tool call

A deterministic contract failure is obvious in the terminal, not
buried behind an opaque judge score:

```text
Failures:
  ✗ wrong_tool: expected 'lookup_customer', agent called refund_customer(invoice_id='INV-7821', amount=950)
  ✗ wrong_tool: expected 'request_approval', agent called refund_customer(invoice_id='INV-7821', amount=950)
```

The same failure is rendered in the HTML report:

![HTML report showing deterministic tool-call failure](assets/report-failures.png)

## Capture mode: talk to your agent, write a scenario

Generate a fresh scenario from a live conversation. No JSON authoring
by hand.

```bash
trainforge record \
  --agent examples.quickstart.agent:run \
  --output examples/quickstart/scenarios/recorded.json
```

You type messages; the agent replies in-process; TrainForge captures
the transcript including tool round-trips. On `:save`, the REPL asks
you the few questions only you can answer (`may_diverge` per turn,
`expected_outcome`, optional `outcome_checks`) and writes a valid
scenario JSON.

The resulting file is identical in shape to a hand-authored scenario
and runs with `trainforge run` immediately.

## Hot-swap: A/B prompt and model changes

The killer feature: change one flag, re-run, see exactly which turns
regressed.

```bash
# Baseline.
trainforge run \
  --agent examples.quickstart.agent:run \
  --scenarios examples/quickstart/scenarios/hello.json \
  --output baseline.json

# Same scenarios, different model.
trainforge run \
  --agent examples.quickstart.agent:run \
  --scenarios examples/quickstart/scenarios/hello.json \
  --output candidate.json \
  --override-model claude-sonnet-4-7

# Diff: which turns degraded?
trainforge diff \
  --before baseline.json \
  --after candidate.json \
  --output regression.html

# regressed=1 — the agent's reply changed under the new model.
open regression.html
```

`--override-model X` sets `TRAINFORGE_OVERRIDE_MODEL=X` as an env var
AND in a `ContextVar` for the duration of the run. Your agent reads
whichever is convenient. Same flow for `--override-prompt PATH`
(contents are loaded and set as `TRAINFORGE_OVERRIDE_PROMPT`).

**Important:** agents that cache config at module-load time won't see
the override (the module is already imported). Read the env var inside
your agent function, not at the top of the module.

## Use in pytest

If you installed with `pip install "trainforge[pytest]"`:

```bash
pytest --trainforge-agent your_module:your_function \
       --trainforge-scenarios-dir tests/agent/scenarios
```

Each scenario becomes one parametrized pytest case. Failures show the
TrainForge diagnostic (turn N, which tool, which arg mismatched) in
the assertion message.

See [Pytest plugin](pytest.md) for the full details.

## Next steps

- [Scenario format](scenarios.md) — the schema you'll write or generate.
- [CLI reference](cli.md) — every subcommand and flag.
- [Concepts](concepts.md) — why deterministic-first, golden injection,
  the 20 NLP checks.
