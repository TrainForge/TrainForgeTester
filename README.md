# TrainForge

Open-source test runner for conversational agents. Runs hand-written or generated scenarios against a live agent API, evaluates each turn against a golden transcript via LLM classification, and reports consistency, divergence, and regressions.

## What it does

- Executes multi-turn scenarios against your agent's HTTP API.
- Uses **golden injection**: after every agent turn the runner feeds the agent the golden reference response on subsequent turns, so a divergence at turn 2 does not corrupt evaluation at turns 4, 6, 8. Each turn is tested in a clean context.
- Tests **tool calls**: scenarios can declare `tool_loops` - ordered or unordered groups of tool calls the agent must make before the next text turn. The runner validates each call deterministically (name + type + exact `expected` argument values, no LLM involvement), injects the golden tool result, and reports per-tool pass / wrong_tool / invalid_arguments / missing.
- Evaluates each agent text turn with one batched LLM call (consistency score 1-5 + per-check pass/fail + divergence type).
- Evaluates the overall outcome of the actual conversation with a second LLM call.
- Scores scenarios as PASS / PARTIAL / FAIL and aggregates consistency across runs.
- Renders an HTML report and a regression-diff HTML report.

**BYO-key.** TrainForge never touches your LLM key. Static mode talks only to your agent and to the LLM you point it at.

## Install

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the `trainforge` console script.

## Quickstart: run the example scenario against the mock agent

The repo ships with the restaurant-booking scenario from the spec under [`scenarios/example_restaurant_booking.json`](scenarios/example_restaurant_booking.json), and a built-in mock agent server.

```bash
# Terminal 1 - mock agent returns the golden responses verbatim
trainforge mock-agent \
  --scenarios scenarios/example_restaurant_booking.json \
  --port 8080

# Terminal 2 - run the scenarios against it
export ANTHROPIC_API_KEY=sk-ant-...
trainforge run \
  --scenarios scenarios/example_restaurant_booking.json \
  --agent-url http://localhost:8080/chat \
  --output results.json

# Render the HTML report
trainforge report --results results.json --output report.html
open report.html
```

## Commands

### `trainforge run`

Execute scenarios against an agent API.

```bash
trainforge run \
  --scenarios scenarios.json \
  --agent-url https://agent.example.com/chat \
  --llm-api-key sk-... \
  --llm-model claude-sonnet-4-6 \
  --runs 5 \
  --timeout 30 \
  --output results.json
```

- `--runs N` runs each scenario N times sequentially for consistency measurement. Scenarios below the 80% spec threshold are flagged `inconsistent` in the output.
- Exits non-zero if any scenarios FAIL or the agent is unreachable.

### `trainforge report`

Render a static HTML report from a results file. Sections: Summary, per-scenario turn-by-turn (golden vs actual), Divergence Summary (expected vs unexpected, grouped by type), Failure Analysis.

```bash
trainforge report --results results.json --output report.html
```

### `trainforge diff`

Compare two results files (e.g. before / after an agent change) and render a regression report.

```bash
trainforge run --scenarios scenarios.json --agent-url ... --output before.json
# ... deploy agent change ...
trainforge run --scenarios scenarios.json --agent-url ... --output after.json

trainforge diff --before before.json --after after.json --output regression.html
```

Buckets every scenario into: `newly_passing`, `newly_failing`, `still_passing`, `still_failing`, `consistency_changed`, `only_in_before`, `only_in_after`. Exits non-zero if any scenarios regressed.

### `trainforge mock-agent`

Serve a fake agent API for development and runner self-tests.

```bash
trainforge mock-agent \
  --scenarios scenarios/example_restaurant_booking.json \
  --port 8080 \
  --mode golden       # or: diverge | error
```

- `golden` - returns the scenario's golden response for each customer message. Every scenario should PASS.
- `diverge` - perturbs the golden response deterministically so the evaluator sees divergences.
- `error` - returns HTTP 500 and delays randomly to exercise the runner's error handling.

## Scenario format

Hand-authored scenarios are welcome; `version: "1.0"` is required. The runner validates scenarios up front and refuses unknown versions.

Minimal text-only scenario:

```json
{
  "version": "1.0",
  "scenarios": [
    {
      "id": "sc-001",
      "name": "...",
      "turns": [
        {"role": "customer", "message": "...", "intent": "..."},
        {"role": "agent", "golden_response": "...", "checks": ["..."], "may_diverge": false}
      ],
      "expected_outcome": "...",
      "outcome_checks": ["..."]
    }
  ]
}
```

### Tool-call extension

Each `agent` turn may declare zero or more `tool_loops` that must complete *before* the text turn. Every tool listed in a loop must be called exactly once; declaring the same tool twice represents two separate calls.

```json
{
  "role": "agent",
  "tool_loops": [
    {
      "ordered": false,
      "tools": [
        {
          "name": "check_weather",
          "arguments_schema": {
            "when": {"type": "string", "expected": "tonight"}
          },
          "expected_response": "Tonight: cold and rainy."
        },
        {
          "name": "check_availability",
          "arguments_schema": {
            "party_size": {"type": "integer", "expected": 2},
            "time":       {"type": "string",  "expected": "7pm"}
          },
          "expected_response": "Tables: corner (indoor), window (indoor)."
        }
      ]
    },
    {
      "ordered": true,
      "tools": [
        {
          "name": "book_table",
          "arguments_schema": {
            "party_size": {"type": "integer", "expected": 2},
            "time":       {"type": "string",  "expected": "7pm"},
            "seating":    {"type": "string",  "expected": "indoor"},
            "table":      {"type": "string",  "expected": "corner"}
          },
          "expected_response": "Booking confirmed. Reference: A1234."
        }
      ]
    }
  ],
  "golden_response": "Booked! Corner table for 2 at 7pm, indoor (A1234).",
  "checks": ["Agent confirms with a reference code"],
  "may_diverge": false
}
```

- `ordered: false` (default) - tools may be called in any order within the loop.
- `ordered: true` - positions are fixed; position 0 must be called first, etc.
- `expected_response` is the string the runner injects back as the tool's result. Golden-injection applies to tools exactly like it applies to text: whether the agent called the right tool or the wrong one, subsequent rounds see the golden tool_call + golden response in the history.

#### Per-argument validation (`arguments_schema`)

Tool calls are structured API inputs, so their validation is fully **deterministic** - no LLM involvement. Every argument in `arguments_schema` is implicitly required; extra keys in the agent's arguments are permitted. Each entry supports two levels of checking:

1. **`type`** - structural type check. Valid values: `string`, `integer`, `number`, `boolean`, `array`, `object`, `any`.
2. **`expected`** - optional literal value. When set, the agent's value must equal it exactly (Python `==`). Pick a canonical form and require it.

If both are declared, both must pass. Any failure is recorded as `invalid_arguments`.

```jsonc
// Type only - accepts any string:
"note": {"type": "string"}

// Exact literal match (use this for anything the test cares about):
"party_size": {"type": "integer", "expected": 2}
"seating":    {"type": "string",  "expected": "indoor"}
"when":       {"type": "string",  "expected": "tonight"}
```

If your agent might legitimately phrase the same value multiple ways (`"tonight"` vs `"this evening"` vs `"7pm"`), pick the canonical one and make the agent normalise - or declare three separate scenarios covering each phrasing.

Per-call outcomes recorded in `results.json` / the HTML report:

| Status              | Meaning                                                                                              |
|---------------------|------------------------------------------------------------------------------------------------------|
| `pass`              | Agent called the expected tool with valid arguments (type + any `expected` equality all pass).       |
| `wrong_tool`        | Agent called a different tool than expected at this slot.                                             |
| `invalid_arguments` | Name matched but arguments failed: missing key, wrong type, or wrong `expected` literal.             |
| `unexpected_tool`   | Agent emitted an extra tool_call after the loop was done.                                             |
| `missing`           | Loop ended with this expected tool never invoked.                                                     |

Any non-`pass` tool_call status blocks a scenario from full PASS. `may_diverge` applies to text divergence only; tool failures always count because they are structural rather than semantic.

## Agent API contract

The runner sends the full conversation history with every request; the agent is stateless from the runner's perspective.

```
POST <agent-url>
Content-Type: application/json

Request:
{
  "messages": [
    {"role": "customer", "content": "..."},
    {"role": "agent",    "content": "..."},
    {"role": "agent",    "content": "", "tool_calls": [{"id": "call_1", "name": "check_weather", "arguments": {"when": "tonight"}}]},
    {"role": "tool",     "tool_call_id": "call_1", "name": "check_weather", "content": "Tonight: cold and rainy."},
    {"role": "customer", "content": "..."}
  ]
}

Response:
{
  "response":   "optional text",
  "tool_calls": [{"id": "...", "name": "...", "arguments": {...}}, ...]
}
```

At least one of `response` or `tool_calls` must be present. Agents without tool support can keep returning `{"response": "..."}` only - scenarios without `tool_loops` are fully backward-compatible.

Errors are mapped as follows:

| Situation                   | Runner behavior                                    |
|----------------------------|-----------------------------------------------------|
| Agent returns non-2xx       | Mark turn `agent_error`. Continue.                  |
| Agent times out             | Retry once. Second timeout -> mark `agent_timeout`. |
| Agent unreachable (conn)    | Mark scenario `agent_unreachable`. Skip.            |
| Agent returns empty body    | Treat as divergence. All checks fail. Continue.     |
| LLM returns unparseable JSON| Retry with stricter prompt. Mark `eval_error`.      |
| Malformed scenarios file    | Refuse to start.                                    |

## Developing

```bash
# Unit + integration tests
pytest

# With coverage
pytest --cov=trainforge --cov-report=term-missing
```

Tests inject a stub LLM via `trainforge.cli._LLM_CLIENT_FACTORY`; the Anthropic SDK is never contacted in CI.

## License

MIT
