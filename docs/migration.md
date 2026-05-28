# Migrating from v1.x

If you have v1.x scenario JSON files, two changes are required to load
under TrainForge 0.1 (`version: "2.0"`):

1. Bump `"version": "1.0"` to `"version": "2.0"` at the top.
2. Rename every `"role": "customer"` to `"role": "user"`.

## Behavioral defaults flipped

- **`may_diverge` defaults to `false`** instead of `true`. The v1.x
  example scenario set `may_diverge: true` on the weather turn; most
  scripted-reply scenarios will *just work* better under 2.0, but if
  you depended on the LLM-evaluated behavior, set `may_diverge: true`
  explicitly per turn.
- **`consistency_score` (1-5) and `divergence_type` are gone** from
  `results.json`. They're replaced by `exact_match: true | false | null`
  and `standard_check_results: [...]`. The HTML report renders the new
  shape.

## Migration script

For a one-off port, this is enough:

```bash
# Bump versions
sed -i '' 's/"version": "1.0"/"version": "2.0"/g' scenarios/*.json

# Rename role
sed -i '' 's/"role": "customer"/"role": "user"/g' scenarios/*.json
```

Then load with `trainforge run --scenarios ...` and fix any
`may_diverge` you need to add back explicitly.

## What didn't change

- Scenario file shape (top-level `version` + `scenarios` array).
- Turn alternation rules (user first, then alternate).
- The agent HTTP contract (`POST` with `{"messages": [...]}`).
- `golden_response`, `tool_loops`, `outcome_checks` semantics.

## What was added in 2.0+

The 2.0 schema kept growing additively after the initial port; if
you're already on 2.0 there's nothing to migrate, but these are
features you can adopt as needed:

- `AgentTurn.node_assertions` — in-process sub-agent / node fire
  assertions. See [Scenario format](scenarios.md#node-assertions-in-process-only).
- `TurnResult.node_assertion_results` in results JSON.
- The `--agent module:callable` flag for in-process testing.
- The `--override-model` / `--override-prompt` hot-swap flags.
- The `trainforge record` REPL capture mode.
- The `trainforge[pytest]` extras.

None of these break a v2.0 scenario file that doesn't use them.
