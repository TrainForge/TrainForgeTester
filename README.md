# TrainForge

**Deterministic-first agent regression testing.** Hand-written or generated scenarios run against a live agent API; structural agent behavior is checked by Python equality (no LLM, no flakiness), and only natural-language consistency between the agent's actual reply and the golden reply is delegated to an LLM — and even then as a fixed list of binary yes/no questions, never as a fuzzy 0-1 score.

## What it does

- Executes multi-turn scenarios against your agent's HTTP API.
- Uses **golden injection**: after every agent turn the runner feeds the agent the golden reference response on subsequent turns, so a divergence at turn 2 doesn't corrupt evaluation at turns 4, 6, 8. Each turn is tested in a clean context.
- Tests **tool calls deterministically**: scenarios declare `tool_loops` — ordered or unordered groups of tool calls the agent must make before the next text turn. Tool name, types, and exact `expected` argument values are checked by Python equality, no LLM in the path. Reports per-tool `pass` / `wrong_tool` / `invalid_arguments` / `missing` / `unexpected_tool`.
- Tests **agent text two ways**, controlled per-turn by `may_diverge`:
  - **`may_diverge: false` (default)** — exact `==` match between actual and golden. Use for curated/scripted replies (legal copy, fixed FAQ, compliance disclosures). Zero LLM calls for the text equivalence check.
  - **`may_diverge: true`** — runs the **20 standard NLP-consistency checks** (same language, same intent, same speech act, no added/omitted facts, comparable register/tone/length, etc.) plus any per-scenario custom checks. ONE batched LLM call per turn returns binary `1`/`0` for each question in a compact JSON array. No 0-1 quality scores, ever.
- Evaluates the overall outcome of the actual conversation with one more batched LLM call (also binary).
- Scores scenarios as PASS / PARTIAL / FAIL and aggregates consistency across `--runs N`.
- Renders an HTML report and a regression-diff HTML report.

**BYO-key.** TrainForge never touches your LLM key. Static mode talks only to your agent and to the LLM you point it at (Anthropic / NVIDIA / Cerebras supported out of the box).

## Why deterministic-first

Most "LLM-as-judge" frameworks ask the model to score quality on a 0-1 scale. That score moves run to run on the exact same input — not because the agent changed but because the judge is non-deterministic. TrainForge sidesteps this for everything that doesn't actually need a judge:

| Failure surface | How TrainForge checks it |
|---|---|
| Wrong tool called | Python string equality on tool name. |
| Wrong tool arguments | Python `==` on the declared `expected` literal + type check. |
| Tool called in wrong slot | Position match in ordered loop, or set match in unordered loop. |
| Verbatim agent reply mismatch | Python `actual == golden` on the response text. |
| Required tool never called | Loop position never matched after the agent finished the round. |
| End-state of the conversation | Per-scenario `outcome_checks` (LLM-judged binary). |
| Free-form agent rephrasing of golden | 20 fixed binary NLP-consistency questions per turn (LLM-judged). |

Every LLM-judged claim in the system is a yes/no question with a stable id. No 0-1 scores, no judge "reasoning" appears in the verdict — only `1` or `0` plus a brief reason for failures.

## Documentation

- Getting started: [`GETTING_STARTED.md`](GETTING_STARTED.md)
- Contributing guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)

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
export OPENAI_API_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
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

- `golden` - returns the scenario's golden response for each user message. Every scenario should PASS.
- `diverge` - perturbs the golden response deterministically so the standard NLP-consistency checks see divergences.
- `error` - returns HTTP 500 and delays randomly to exercise the runner's error handling.

## Scenario format (v2.0)

Hand-authored scenarios are welcome; `version: "2.0"` is required. The runner validates scenarios up front and refuses unknown versions.

Minimal text-only scenario:

```json
{
  "version": "2.0",
  "scenarios": [
    {
      "id": "sc-001",
      "name": "...",
      "turns": [
        {"role": "user", "message": "...", "intent": "..."},
        {"role": "agent", "golden_response": "...", "checks": ["..."]}
      ],
      "expected_outcome": "...",
      "outcome_checks": ["..."]
    }
  ]
}
```

### Two text-evaluation modes per agent turn

Every `agent` turn carries `may_diverge` (default `false`):

| `may_diverge` | Behavior | When to use |
|---|---|---|
| `false` (default) | Python `==` between actual and golden text. Failure -> `exact_match: false`. **Zero LLM calls** for the equivalence check. | Curated/scripted replies: legal disclaimers, fixed FAQ answers, policy-mandated responses. |
| `true` | The 20 standard NLP-consistency checks (see below) plus any per-scenario custom `checks`, batched into ONE LLM call returning binary `1`/`0` per question. | Open-ended replies that may legitimately rephrase the golden but should preserve intent, content, register, etc. |

### The 20 standard NLP-consistency checks

Applied automatically to every `may_diverge: true` agent turn. All 20 are binary comparisons of the actual AI response against the golden AI response:

| id | what it checks |
|---|---|
| `same_language` | Same natural language. |
| `same_speech_act` | Same speech act (statement / question / confirmation / request / promise / apology / refusal). |
| `same_intent` | Same communicative intent. |
| `same_action_state` | Same action state (not started / pending / in progress / completed / failed). |
| `same_next_step` | Same next step prompted from the user (or both signal "no next step"). |
| `same_propositional_content` | Same set of factual claims. |
| `no_added_facts` | No factual claims in actual that aren't in golden. |
| `no_omitted_facts` | No factual claims from golden that are missing in actual. |
| `no_contradictions` | Doesn't contradict any claim in golden. |
| `same_named_entities` | Same people / places / products / orgs referenced. |
| `same_numerics` | Numbers, dates, times, codes, IDs match. |
| `same_call_to_action` | Both contain (or omit) the same CTA. |
| `same_disclosures` | Both include (or omit) the same disclosures / caveats / warnings. |
| `comparable_register` | Same register (formal / casual / technical / consumer). |
| `comparable_tone` | Same tone (polite / curt / empathetic / neutral / enthusiastic). |
| `comparable_specificity` | Same specificity (concrete details vs generic placeholders). |
| `comparable_hedging` | Same level of confidence/hedging (decisive vs tentative). |
| `comparable_length` | Length within ~0.5x to 2x of golden. |
| `same_persona` | Same persona/voice; neither breaks character. |
| `same_information_order` | Same ordering of major information units. |

Source of truth: [`src/trainforge/standard_checks.py`](src/trainforge/standard_checks.py). The list is hard-coded and stable; check `id` values are part of the public results contract.

### Compact LLM wire format

Every LLM call (per-turn standard+custom batch, custom-only on exact-match turns, outcome eval) uses the same compact JSON shape:

```json
{"r": [1, 1, 0, 1, 1, ...], "f": {"3": "different language"}}
```

- `r` is a positional array of `1` (pass) and `0` (fail), length must equal the number of asked questions.
- `f` maps the 1-based index of failed questions to a brief reason. Pass questions get no entry.

For 20 standard checks all passing the response is ~30 output tokens. Replaces the v1.x "1-5 score + divergence_type + per-check JSON" format which produced 150-500 tokens per call.

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
  "checks": ["Response confirms the booking with a reference code"]
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
    {"role": "user",  "content": "..."},
    {"role": "agent", "content": "..."},
    {"role": "agent", "content": "", "tool_calls": [{"id": "call_1", "name": "check_weather", "arguments": {"when": "tonight"}}]},
    {"role": "tool",  "tool_call_id": "call_1", "name": "check_weather", "content": "Tonight: cold and rainy."},
    {"role": "user",  "content": "..."}
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

## Migrating from v1.x scenarios

If you have v1.x scenario JSON files, two changes are required to load under v0.2:

1. Bump `"version": "1.0"` to `"version": "2.0"` at the top.
2. Rename every `"role": "customer"` to `"role": "user"`.

Behavioral defaults flipped:

- `may_diverge` now defaults to `false` instead of `true` (v1.x had it as a per-turn opt-in but the example scenario set it to `true` on the weather turn). Most scripted-reply scenarios will just *work* better under v0.2 — but if you depended on the LLM-evaluated behavior, set `may_diverge: true` explicitly per turn.
- `consistency_score` (1-5) and `divergence_type` are gone from `results.json`. They are replaced by `exact_match: true|false|null` and `standard_check_results: [...]`. The HTML report renders the new shape.

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
