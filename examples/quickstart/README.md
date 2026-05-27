# Quickstart: 60 seconds to first green test

This folder ships a toy agent and one scenario so you can run TrainForge
end-to-end against a Python callable with no HTTP server.

## What's here

- `agent.py` — a one-turn async agent. Returns `"[model=...] Hello! I heard: ... (...)"`.
- `scenarios/hello.json` — one scenario that asserts the agent echoes the user's message.

## Run it

From the repo root, with `pip install -e .` already done:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_API_URL=https://api.openai.com/v1

trainforge run \
  --agent examples.quickstart.agent:run \
  --scenarios examples/quickstart/scenarios/hello.json \
  --output results.json

trainforge report --results results.json --output report.html
open report.html
```

You should see `1/1 passed (100%)`. The scenario uses `may_diverge: false`,
so even though there are no custom checks, TrainForge does an exact-string
match between the agent's reply and the `golden_response` baked into the
scenario.

## Try the hot-swap

The scenario's `golden_response` assumes the default prompt and model. If
you override either, the exact match will fail — which is the point of
regression testing:

```bash
trainforge run \
  --agent examples.quickstart.agent:run \
  --scenarios examples/quickstart/scenarios/hello.json \
  --output results-changed.json \
  --override-model claude-sonnet-4-7

trainforge diff \
  --before results.json \
  --after results-changed.json \
  --output regression.html
open regression.html
```

The diff report shows turn 1 regressed: the agent now says `[model=claude-sonnet-4-7] ...`
instead of `[model=echo-1] ...`. In a real workflow, this is how you'd
catch a model swap silently changing your agent's output.

## Capture mode

Generate a fresh scenario from a live conversation:

```bash
trainforge record \
  --agent examples.quickstart.agent:run \
  --output examples/quickstart/scenarios/recorded.json
```

Type messages, watch the agent reply, type `:save` when you're done. The
record loop prompts you for `may_diverge` per turn and an `expected_outcome`
for the whole conversation, then writes a valid scenario file.

## Use in pytest

If you installed with `pip install "trainforge[pytest]"`:

```bash
pytest --trainforge-agent examples.quickstart.agent:run \
       --trainforge-scenarios-dir examples/quickstart/scenarios
```

Each scenario becomes one pytest case. Failures show the TrainForge
diagnostic (turn N, which tool, which arg mismatched) in the assertion
message.
