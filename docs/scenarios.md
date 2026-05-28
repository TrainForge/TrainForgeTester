# Scenario format

TrainForge scenarios are JSON. The runner validates them up front and
refuses unknown versions (see [Migration](migration.md) if you have
v1.x files).

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

The schema is enforced by Pydantic with `extra="forbid"` — unknown
fields are rejected at load time, not silently ignored.

## Scenario fields

| Field | Required | Description |
|---|---|---|
| `id` | yes | Stable identifier (string). Appears in results and reports. |
| `name` | yes | Human-readable. Shown in CLI output and HTML report. |
| `description` | optional | Free-form. Use it to explain what the scenario tests. |
| `source_transcript_id` | optional | Pointer back to the original conversation if you captured this. |
| `tags` | optional | Array of strings. Filter scenarios in CI by tag. |
| `turns` | yes | Alternating user → agent → user → agent. Must start with `user`. |
| `expected_outcome` | yes | One-sentence summary of what success looks like. Passed to the LLM judge. |
| `outcome_checks` | optional | Binary "did the agent achieve X?" questions. Empty list = no judge call. |

## User turns

```json
{
  "role": "user",
  "message": "Hi, I'd like to book a table for tonight",
  "intent": "initiate_booking"
}
```

| Field | Required | Description |
|---|---|---|
| `role` | yes | Must be `"user"`. |
| `message` | yes | Verbatim user text. |
| `intent` | optional | Short snake_case label. Documentation only; not validated. |

## Agent turns

```json
{
  "role": "agent",
  "tool_loops": [],
  "golden_response": "Of course! How many guests will be joining?",
  "checks": [],
  "may_diverge": false,
  "divergence_note": null,
  "node_assertions": []
}
```

| Field | Required | Description |
|---|---|---|
| `role` | yes | Must be `"agent"`. |
| `golden_response` | yes | The text the agent is expected to produce. Empty string is allowed by the schema, but discouraged: an empty golden makes `may_diverge: false` exact-match trivially fail against any non-empty reply, and the LLM judge for `may_diverge: true` has nothing to compare against. |
| `tool_loops` | optional | Tool calls that must happen before the text reply. See below. |
| `checks` | optional | Per-turn natural-language binary checks (LLM-evaluated). |
| `may_diverge` | optional | `false` (default) = exact-match. `true` = 20 NLP checks + custom. |
| `divergence_note` | optional | Free-form note for humans reading the scenario. |
| `node_assertions` | optional | In-process sub-agent / node fire assertions. See below. |

### Two text-evaluation modes

`may_diverge` is the single most important per-turn flag.

| `may_diverge` | Behavior | When to use |
|---|---|---|
| `false` (default) | Python `==` between actual and golden text. **Zero LLM calls** for the equivalence check. | Curated/scripted replies: legal disclaimers, fixed FAQ answers, policy-mandated responses. |
| `true` | The 20 standard NLP-consistency checks + any per-turn custom `checks`, batched into ONE LLM call returning binary `1`/`0` per question. | Open-ended replies that may legitimately rephrase the golden but should preserve intent, content, register, etc. |

Custom `checks` run in **both** modes when present (the deterministic
exact-match check happens first when `may_diverge: false`; the
LLM-judged custom checks run on top).

See [Concepts](concepts.md#the-20-standard-nlp-consistency-checks) for
the full list of standard NLP checks.

## Tool loops

Each agent turn may declare zero or more `tool_loops` that must
complete *before* the text turn. Every tool listed must be called
exactly once; declaring the same tool twice means two separate calls.

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

- `ordered: false` (default) — tools may be called in any order
  within the loop.
- `ordered: true` — positions are fixed; position 0 must be called
  first, etc.
- `expected_response` is the string the runner injects back as the
  tool's result. Golden-injection applies to tools exactly like text:
  whether the agent called the right tool or the wrong one,
  subsequent rounds see the golden tool_call + golden response in
  history.

### Per-argument validation

Tool calls are structured API inputs, so validation is fully
deterministic — no LLM involvement.

Every argument in `arguments_schema` is implicitly required; extra
keys in the agent's actual arguments are permitted. Each entry
supports two levels of checking:

1. **`type`** — structural type check. Valid values: `string`,
   `integer`, `number`, `boolean`, `array`, `object`, `any`.
2. **`expected`** — optional literal value. When set, the agent's
   value must equal it exactly (Python `==`).

```jsonc
// Type only — accepts any string:
"note": {"type": "string"}

// Exact literal match (use this for anything the test cares about):
"party_size": {"type": "integer", "expected": 2}
"seating":    {"type": "string",  "expected": "indoor"}
"when":       {"type": "string",  "expected": "tonight"}
```

If your agent might legitimately phrase the same value multiple ways
(`"tonight"` vs `"this evening"` vs `"7pm"`), pick a canonical form
and require it, or write three separate scenarios.

### Tool-call outcomes

Per-call status recorded in `results.json` and the HTML report:

| Status | Meaning |
|---|---|
| `pass` | Agent called the expected tool with valid arguments (type + any `expected` equality all pass). |
| `wrong_tool` | Agent called a different tool than expected at this slot. |
| `invalid_arguments` | Name matched but arguments failed: missing key, wrong type, or wrong `expected` literal. |
| `unexpected_tool` | Agent emitted an extra tool_call after the loop was done. |
| `missing` | Loop ended with this expected tool never invoked. |

Any non-`pass` tool status blocks full PASS. `may_diverge` applies to
text divergence only; tool failures always count because they're
structural.

## Node assertions (in-process only)

For agents that use sub-agents or a graph (LangGraph, OpenAI Agents
SDK handoffs, custom orchestrators), scenarios can assert that a
specific node fired during a turn:

```json
{
  "role": "agent",
  "golden_response": "Approval requested for $950 refund.",
  "node_assertions": [
    {
      "node_name": "refund_handler",
      "must_fire": true,
      "args_match": {"invoice_id": "INV-7821", "amount": 950}
    },
    {
      "node_name": "auto_approve",
      "must_fire": false
    }
  ]
}
```

The agent must wrap sub-agent calls in `with observer.node(name, args=...):`:

```python
from trainforge import observer

async def run_agent(messages):
    with observer.node("intent_classifier"):
        intent = classify(messages)
    if intent == "refund":
        with observer.node("refund_handler", args={"invoice_id": ..., "amount": ...}):
            return await handle_refund(messages)
```

| Field | Required | Description |
|---|---|---|
| `node_name` | yes | Name passed to `observer.node(name=...)`. Exact-match. |
| `must_fire` | optional | `true` (default) = must fire. `false` = must NOT fire. |
| `args_match` | optional | Literal-equality on observed `args`. Only valid when `must_fire=true`. |

The `args_match` filter is permissive: any keys absent from the
assertion are allowed in the observed args. Each declared key/value
must equal the observed args via Python `==`.

**Note:** `node_assertions` only fire under `InProcessTransport`. Over
HTTP, the runner has no way to observe in-process state; assertions
will report "node never fired" because the observer scope wasn't
entered. Schema rejects `args_match` when `must_fire=false` (negative
assertion + args filter is ambiguous).

## Outcome checks

Run once per scenario at the end, over the full actual transcript.
Each check is a binary "did the agent achieve X?" question
LLM-evaluated.

```json
{
  "expected_outcome": "Booking confirmed for 2 people at 7pm, indoor seating, corner table",
  "outcome_checks": [
    "A booking was made (not just discussed)",
    "Party size is 2",
    "Time is around 7pm",
    "Seating is indoor",
    "Confirmation number or reference was provided"
  ]
}
```

`outcome_checks: []` is allowed and short-circuits the LLM call
entirely (the scenario then only relies on per-turn checks and tool
verifications).

## Examples in the repo

| Scenario | What it tests |
|---|---|
| [`scenarios/example_restaurant_booking.json`](../scenarios/example_restaurant_booking.json) | Multi-turn booking with weather check + tool loops + may_diverge mix. |
| [`scenarios/example_restaurant_booking_arguments.json`](../scenarios/example_restaurant_booking_arguments.json) | Same flow but stresses argument-level deterministic validation. |
| [`scenarios/example_refund_approval_policy.json`](../scenarios/example_refund_approval_policy.json) | Policy gate: large refund must request approval before any refund tool fires. |
| [`examples/quickstart/scenarios/hello.json`](../examples/quickstart/scenarios/hello.json) | Tiny in-process toy scenario for the 60-second walkthrough. |

## Related

- [Agent contracts](agents.md) — how the agent actually receives turns.
- [CLI reference](cli.md) — `trainforge run`, `record`, `diff`, etc.
- [Concepts](concepts.md) — why deterministic-first, golden injection.
