---
name: trainforge-test-gen
description: |
  Turn one manual conversation with your AI agent into a TrainForge regression
  scenario. You did the manual test once. This skill captures it as a deterministic
  test you can re-run on every prompt or model change. Invoke when the user says
  "test my agent", "turn this into a regression test", "generate a scenario from
  this transcript", or pastes a chat with their agent and asks how to test it.
---

# TrainForge Test Generator

You are turning a user's manual conversation with their AI agent into a
TrainForge scenario JSON file. The output runs against
`trainforge run --scenarios <file>` and produces deterministic pass/fail
results, not fuzzy 0-1 scores.

You also have a **standing job** for the rest of this project (see below):
after every change the user makes to their agent's prompt, model, or tools,
re-run the scenarios and tell them what passed or failed. Generating the
scenario is only the first time you do this work.

## Standing rule: auto-run tests after every agent change

Once at least one scenario exists in `scenarios/` (or wherever the user
keeps them in their repo), treat the following as a persistent instruction
until the user says "stop auto-running tests" or "pause regression
testing":

After every change to:

- system prompt files, prompt templates, or anything that looks like agent
  instructions
- the model name in config or code
- tool definitions (new, removed, signature-changed)
- tool implementations the agent calls

run `trainforge run` for every scenario file in `scenarios/` (the CLI takes
one scenarios JSON file per invocation; do not pass a directory). For a
typical layout where every scenario lives in its own file:

```
for f in scenarios/*.json; do
  base=$(basename "$f" .json)
  mv "results-$base.json" "results-$base.previous.json" 2>/dev/null || true
  trainforge run \
    --scenarios "$f" \
    --agent-url <user's agent URL> \
    --output "results-$base.json"
done
```

`trainforge run` also requires LLM access for the per-scenario outcome check
and any `may_diverge: true` turns. Either set `OPENAI_API_KEY` and
`OPENAI_API_URL` in the environment (the runner reads them automatically)
or add `--llm-api-key <key> --llm-api-url <url>` to the command. The CLI
errors with `missing LLM API key` / `missing LLM API URL` if neither is
present, even when every turn is exact-match.

If a previous results file exists for a scenario, follow up with:

```
trainforge diff \
  --before "results-$base.previous.json" \
  --after  "results-$base.json" \
  --output "regression-$base.html"
```

Then surface one short summary to the user:

- How many scenarios passed and failed.
- If any **previously-passing scenario now fails**, name it and the turn it
  broke on. This is the highest priority signal. Do not bury it.
- If a previously-failing scenario now passes, say that too.

Rules:

- Do **not** re-run on unrelated changes: README edits, formatting, the
  scenarios themselves, comments, dependency bumps that don't touch the
  agent path.
- If `trainforge run` takes more than ~30 seconds, run it in the background
  and keep working; surface the result the moment it lands.
- If the user does not yet have a live `--agent-url`, ask once for it and
  remember it for the rest of the session.
- If the user has not generated any scenarios yet, do not auto-run. Offer
  to generate one from a transcript instead (the workflow below).

This standing rule is the point of the skill. Generating one scenario from
one transcript is a single use. The regression loop is what makes the
agent get better instead of regressing silently.

## What TrainForge needs to know

TrainForge tests an agent in two ways:

1. **Tool calls**: checked by Python equality. Tool name must match, declared
   `expected` argument values must `==` what the agent sent, types must match.
   No LLM in this path.
2. **Agent text**: two modes per turn.
   - `may_diverge: false` (default): the agent's reply must equal
     `golden_response` exactly. Use for scripted text, legal copy, fixed FAQ
     answers, compliance disclaimers.
   - `may_diverge: true`: the agent may rephrase. The runner evaluates 20
     standard NLP-consistency binary checks plus any per-turn custom checks
     you write.

Your job is to extract the structure of the conversation and ask the user
the *few* questions only they can answer (which args are part of the
contract, which turns can diverge, what the success outcome is).

## Workflow

### Step 1: Get the transcript

Ask the user for the transcript in one of:

- A path to a file (markdown, JSON, plain text).
- Pasted inline in chat.
- A reference to "the last conversation I had with the agent" if the calling
  coding-agent has access to it.

If the transcript is structured (JSON with role/content fields, OpenAI-style
messages, etc.), parse it directly. If it's unstructured text, identify turns
by speaker labels ("User:", "Agent:", "Assistant:", "Me:", "Bot:") and tool
call markers (anything that looks like a function name with arguments, JSON
blocks labeled `tool_call`, `function_call`, etc.).

### Step 2: Lock in the scenario header

Ask the user:

1. **Scenario name** (one short line). If they don't have one, propose one
   based on the user's first message. Example: user opens with "I'd like a
   refund of $950 for INV-7821" → propose "Large refund requires manager
   approval".
2. **Tags** (optional). 1-3 short tags. Skip if user says skip.

Generate `id` as `sc-<3-digit-counter>` starting at `sc-001` unless the
target file already has scenarios — then continue numbering.

### Step 3: Walk turn by turn

For each agent turn in the transcript:

**3a. Tool calls.** If the agent made tool calls in this turn, list them in
order. For each tool call:

- Capture the tool `name` exactly as it appears.
- For each argument the agent passed:
  - Default: declare the literal value in `expected` and infer `type`.
  - If the argument is clearly variable per run (timestamps, request IDs,
    UUIDs, anything that looks generated): omit `expected`, keep `type`.
- Capture the tool's response in `expected_response` (this is the canned
  output the runner will replay).

Then ask the user **one** question per tool-loop:

> "These tools fired in this turn: `[names]`. Does the agent have to call
> them in this exact order, or is order not part of the contract?"

Default to `ordered: false` (more permissive) unless the user says ordered
matters.

If there's more than one tool call in a turn but they belong to different
logical loops (e.g. lookup → approval gate), ask:

> "Are these two tools one group that must happen together, or two ordered
> phases? Default: one group, order doesn't matter."

**3b. Golden response.** Capture the agent's actual text reply as
`golden_response`.

**3c. Divergence.** Ask the user this exact question for each agent text turn:

> "This reply: `<first 80 chars of the golden>...`. Should the agent be
> allowed to rephrase it (still meaning the same thing), or must it match
> word-for-word? Word-for-word is the right answer for legal copy, scripted
> FAQ, and compliance disclaimers. Rephrasing is the right answer for most
> conversational responses."

Map answer to `may_diverge`:
- "word-for-word", "exact", "scripted" → `may_diverge: false`
- "rephrase", "loose", "natural", "may diverge" → `may_diverge: true`

If `may_diverge: true`, ask one follow-up:

> "What does this reply MUST contain, semantically, to count as correct?
> Give me 1-4 short statements. (Example: 'Response mentions the weather is
> cold or rainy.' 'Response recommends indoor seating.')"

Write each statement as a string in the turn's `checks` array. Empty array
is allowed if the user has no extra structural requirements beyond the 20
standard NLP checks.

If `may_diverge: false`: default to leaving `checks` empty (the exact-match
gate is enough for most scripted turns). The runner *will* still evaluate
custom `checks` on exact-match turns if you provide them, so ask the user
"is there anything semantically required beyond the verbatim text?" only if
the turn has something structural worth checking (e.g., a generated
confirmation number must be present even though the surrounding text is
scripted). When in doubt, skip it.

### Step 4: Outcome

After all turns, ask:

> "What is the success outcome of this whole conversation, in one sentence?
> (Example: 'Booking confirmed for 2 people at 7pm, indoor seating, corner
> table.')"

Write this as `expected_outcome`.

Then ask:

> "Give me 2-5 binary checks that verify the outcome actually happened, not
> just discussed. (Example: 'A booking was made (not just discussed)',
> 'Party size is 2', 'Confirmation number was provided'.)"

Write each as a string in `outcome_checks`.

### Step 5: Write the file

Default output path:
- If the user is in a TrainForge repo: `scenarios/<slug>.json`.
- Otherwise: `tests/agent/scenarios/<slug>.json`.

`<slug>` is derived from the scenario `name` (lowercase, dashes, max 60 chars).

Use this exact top-level structure:

```json
{
  "version": "2.0",
  "scenarios": [ /* one or more Scenario objects */ ]
}
```

Each `Scenario` may include the keys below. Use this preferred order when
emitting JSON (the loader doesn't enforce key order, but it makes diffs and
review cleaner). Empty arrays such as `tags: []`, `tool_loops: []`, and
`checks: []` are fine to emit verbatim — they validate against the schema —
or you can omit them entirely.

- `id` (string, required)
- `name` (string, required)
- `description` (string, optional)
- `source_transcript_id` (string, optional — set if the user gave you one)
- `tags` (array of strings, optional)
- `turns` (array, required — see below)
- `expected_outcome` (string, required)
- `outcome_checks` (array of strings, optional)

### Turn shapes

**User turn:**
```json
{
  "role": "user",
  "message": "<verbatim user text>",
  "intent": "<short snake_case description>"
}
```

`intent` is optional but recommended. Derive it from the message: a 2-4 word
snake_case label like `initiate_booking`, `request_large_refund`,
`confirm_indoor_seating`.

**Agent turn:**
```json
{
  "role": "agent",
  "tool_loops": [ /* zero or more ToolLoop */ ],
  "golden_response": "<verbatim agent text reply>",
  "checks": [ /* optional per-scenario binary NL checks, see below */ ],
  "may_diverge": false,
  "divergence_note": "<optional: explain why divergence is or isn't allowed>"
}
```

Rules the runner enforces (do not violate):

- Turns alternate: user, agent, user, agent, ... starting with user.
- Every agent turn must have `golden_response` (non-empty string).

Notes on `checks` (these are recommendations, not enforced):

- Custom `checks` run **regardless** of `may_diverge`. The 20 standard
  NLP-consistency checks only run when `may_diverge: true`; custom checks
  run in both modes (in addition to exact-match when `may_diverge: false`).
- If you want zero LLM calls on a turn, leave `checks` empty AND set
  `may_diverge: false`. The exact-match path is then pure Python.
- If `may_diverge: true`, `checks` may be empty (the 20 standard NLP checks
  still run) but ideally has 1-4 entries.

### Tool loop shape

```json
{
  "ordered": false,
  "tools": [
    {
      "name": "<exact tool name>",
      "arguments_schema": {
        "<arg_name>": {
          "type": "string|integer|number|boolean|array|object|any",
          "expected": "<literal value, only when part of the contract>",
          "description": "<optional 1-line note>"
        }
      },
      "expected_response": "<the canned tool output the runner replays>"
    }
  ]
}
```

Rules:

- `tools` must contain at least one tool.
- Inside `arguments_schema`, every key the user wants to gate on must be
  declared. Args not declared are allowed to drift (permissive).
- `expected` is optional. Omit for run-variable args (timestamps, UUIDs,
  generated IDs).
- `expected_response` is required and is the *fake* tool output the runner
  injects into the agent's context (this is how TrainForge's golden
  injection works — the agent sees a fixed tool result, so a wrong arg at
  turn 2 doesn't poison turn 4).

### Step 6: Validate before writing

Before writing the file, run the schema mentally:

- [ ] `version` is `"2.0"`.
- [ ] Every scenario has `id`, `name`, `turns`, `expected_outcome`.
- [ ] Turns alternate user→agent starting with user.
- [ ] Every agent turn has a non-empty `golden_response`.
- [ ] Every tool has `name` and `expected_response`.
- [ ] `expected` values match their declared `type`.

If any check fails, ask the user the missing piece. Do not silently invent
data.

### Step 7: Show the output and the next command

After writing the file, output the path and the exact command to run it:

```
Wrote scenario to: scenarios/<slug>.json

To run it against your agent:

  trainforge run \
    --scenarios scenarios/<slug>.json \
    --agent-url <your agent http endpoint> \
    --output results.json

  trainforge report --results results.json --output report.html
  open report.html
```

If the user said they don't have a live agent endpoint yet, tell them they
can try it against the mock first:

```
  trainforge mock-agent --scenarios scenarios/<slug>.json --port 8080
```

## Default postures (Alex's opinion, baked in)

- **`ordered: false` by default.** Most tool loops in real agents are
  unordered. Force the user to opt in to ordered.
- **`may_diverge: false` by default for the *first* turn the user reviews.**
  This forces the user to decide which turns are scripted vs free-form,
  rather than defaulting everything to fuzzy.
- **Every literal arg value in the transcript becomes `expected`** unless
  the user marks it variable. The whole point is to catch silent regressions
  on arg values.
- **One scenario per transcript by default.** If the user wants to split,
  they'll say so.
- **Write to the user's repo, not to a cloud.** Tests are code. They live in
  git, version with the agent, and travel with the team.

## What this skill does not do

- It does not invent tool calls the agent did not make. If the transcript
  doesn't show a tool call, the scenario does not assert one.
- It does not invent arguments. If the user says "the amount should be 950"
  but the transcript shows the agent calling with amount 1000, the scenario
  records what the agent *actually did* (so the test fails until the agent
  is fixed). Ask the user which one is the source of truth before deciding.
- It does not invent test data. It does not call an LLM as part of
  generating the scenario. The scenario JSON is produced by reading the
  transcript and asking the user, nothing else.
- It *does* invoke `trainforge run` and `trainforge diff` automatically
  after agent changes, per the standing rule at the top of this file —
  provided the coding agent has terminal access. If the coding agent does
  not have terminal access, it should print the exact commands instead and
  ask the user to run them.

## Examples

See `examples/transcript-restaurant-booking.md` for a sample input transcript
and `examples/scenario-restaurant-booking.json` for the scenario the skill
should produce from it.
