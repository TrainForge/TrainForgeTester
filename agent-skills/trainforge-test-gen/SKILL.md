---
name: trainforge-test-gen
description: |
  Turn one good conversation with your AI agent into a TrainForge regression
  test in 5 seconds. Auto-mode captures the conversation and infers sensible
  defaults; you (the coding agent) then label the pending LLM-dependent checks
  using the 20-NLP-consistency rubric so the user never needs to configure an
  Anthropic / OpenAI key. Invoke when the user says "test my agent", "turn
  this into a regression test", "generate a scenario from this transcript",
  pastes a chat with their agent and asks how to test it, or wants to verify
  the agent still produces the right result after a change.
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
one scenarios JSON file per invocation; do not pass a directory).
**Default to `--no-judge`** and label the pending checks yourself
(see "Coding-agent-as-judge" below). For a typical layout where every
scenario lives in its own file:

```
for f in scenarios/*.json; do
  base=$(basename "$f" .json)
  mv "results-$base.json" "results-$base.previous.json" 2>/dev/null || true
  trainforge run \
    --scenarios "$f" \
    --agent <user's module:callable OR fallback to --agent-url> \
    --output "results-$base.json" \
    --no-judge

  # YOU now read results-$base.json, label every pending check (per
  # the 20-NLP rubric below), save the file, then:
  trainforge rescore --results "results-$base.json"
done
```

Pick the transport:

- **Prefer `--agent module:callable`** when the user's agent is a Python
  callable (LangChain, CrewAI, LangGraph, OpenAI Agents SDK, plain async
  function in a notebook). The in-process transport is faster, supports
  `node_assertions`, and works without standing up an HTTP server.
- **Use `--agent-url <http endpoint>`** only when the user's agent is
  already running as an HTTP service in production and they want to test
  it through the real wire.

Mutually exclusive — pass exactly one.

### LLM judge: prefer YOU over a configured API key

`trainforge run` normally needs LLM access for the per-scenario
outcome check and any `may_diverge: true` turns. There are two paths:

**Default path (no LLM credentials configured): use the `--no-judge` flow.**

The user is already paying for you (Claude / Cursor / whatever coding
agent they're running). Reuse that. Don't ask them to configure an
Anthropic / OpenAI key just to grade tests.

Append `--no-judge` to every `trainforge run` call you make:

```
trainforge run \
  --scenarios "$f" \
  --agent <user's module:callable OR --agent-url ...> \
  --output "results-$base.json" \
  --no-judge
```

The runner will execute the agent, capture actual vs golden, and
evaluate tool calls deterministically. Every LLM-dependent check
(custom turn checks, the 20 standard NLP-consistency checks, outcome
checks) is emitted with `pending: true` and a placeholder
`passed: false`. **YOU then label those pending checks** — see
"Coding-agent-as-judge" below — and call `trainforge rescore` to get
the deterministic summary.

**Configured-key path (CI, batch testing, user opted in):** If
`OPENAI_API_KEY` and `OPENAI_API_URL` are set in the environment, OR
the user explicitly says "use the configured API key for judging,"
drop `--no-judge` and let the runner make LLM calls itself. Same
output shape, no labeling step needed. Use this for CI runs.

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

## Coding-agent-as-judge: how to label pending checks

When you run with `--no-judge`, every LLM-dependent check in the
output `results.json` carries `pending: true` plus a placeholder
`passed: false`. Your job:

1. Read `results.json`.
2. For each pending check, apply the rubric below directly. **You are
   the labeler, not the aggregator.** Set `passed: true | false`,
   write a short `explanation`, and flip `pending: false`. Save the
   file. Do not invent scenario-level verdicts; those come from
   `trainforge rescore` in the next step.
3. Call `trainforge rescore --results results.json`. This re-runs the
   deterministic scoring pipeline (`classify_scenario_run` +
   `summarize`) over your labels and produces the official per-scenario
   pass/fail and the run summary.
4. Surface the rescore summary to the user. If any scenarios failed,
   name them and which turn failed.

### The three kinds of pending check

**1. Per-turn custom checks (`turn.checks[*]`).** The scenario author
wrote a natural-language question about THIS turn. The check object
looks like:

```json
{"check": "Response confirms the booking with a reference code",
 "passed": false, "explanation": "", "pending": true}
```

Read the turn's `actual_response` (and `golden_response` for context).
Decide: did the actual response satisfy the question? Set `passed`
true/false and write a 1-line `explanation`.

**2. Standard NLP-consistency checks (`turn.standard_check_results[*]`).**
Only present on `may_diverge: true` turns. There are exactly 20 of
these per turn, each with a stable `id`. Apply the rubric below to
each one, comparing `actual_response` against `golden_response`.

**3. Outcome checks (`run.outcome.checks[*]`).** Run once per scenario
at the end. Look at the FULL conversation (the runner doesn't put it
into the results file directly — reconstruct from per-turn
`user_message` + `actual_response` + the original scenario's
`expected_outcome`). Decide whether the agent achieved each binary
outcome check.

### The 20 NLP-consistency rubric

For each `standard_check_results` entry, apply the rule below. Compare
`actual_response` against `golden_response`. Return `passed: true` if
the rule holds, `false` otherwise. Write a brief `explanation` only
on failure.

| `id` | Rule for `passed: true` |
|---|---|
| `same_language` | Both replies are in the same natural language. |
| `same_speech_act` | Same speech act: statement / question / confirmation / request / promise / apology / refusal. |
| `same_intent` | Same communicative intent. |
| `same_action_state` | Same action state: not_started / pending / in_progress / completed / failed. |
| `same_next_step` | Same next-step prompt for the user (or both omit one). |
| `same_propositional_content` | Same set of factual claims. |
| `no_added_facts` | Actual introduces no claims that aren't in golden. |
| `no_omitted_facts` | Actual preserves every factual claim that's in golden. |
| `no_contradictions` | Actual doesn't contradict any claim in golden. |
| `same_named_entities` | Same people / places / products / orgs referenced. |
| `same_numerics` | Numbers, dates, times, codes, IDs match. |
| `same_call_to_action` | Both contain (or both omit) the same CTA. |
| `same_disclosures` | Same disclosures / caveats / warnings. |
| `comparable_register` | Same register: formal / casual / technical / consumer. |
| `comparable_tone` | Same tone: polite / curt / empathetic / neutral / enthusiastic. |
| `comparable_specificity` | Same specificity (concrete vs generic placeholders). |
| `comparable_hedging` | Same confidence level (decisive vs tentative). |
| `comparable_length` | Length within ~0.5× to 2× of golden. |
| `same_persona` | Same voice / persona; neither breaks character. |
| `same_information_order` | Same ordering of major information units. |

Apply each rule independently. A single rule failing should produce
`passed: false` for that one check only; the others may still pass.

### Worked example

`results.json` contains a turn like:

```json
{
  "turn_index": 1,
  "user_message": "What time?",
  "golden_response": "How about 7pm?",
  "actual_response": "7pm works for me",
  "may_diverge": true,
  "standard_check_results": [
    {"id": "same_language", "question": "Same natural language.",
     "passed": false, "explanation": "", "pending": true},
    {"id": "same_speech_act", "question": "Same speech act...",
     "passed": false, "explanation": "", "pending": true},
    ...
  ]
}
```

You label:

```json
{"id": "same_language", "question": "Same natural language.",
 "passed": true, "explanation": "", "pending": false}
{"id": "same_speech_act", "question": "Same speech act...",
 "passed": false,
 "explanation": "golden asks a question; actual makes a confirmation",
 "pending": false}
...
```

After labeling, save the file and call `trainforge rescore`.

### When NOT to use the labeling flow

- The user explicitly set `OPENAI_API_KEY` / `OPENAI_API_URL` and said
  "use the API key for judging." Drop `--no-judge`; the runner judges.
- The user is running in CI without a coding agent in the loop. They
  need a configured key.
- The scenarios have **zero** LLM-dependent checks (all
  `may_diverge: false`, no per-turn custom checks, no `outcome_checks`).
  The runner produces a full verdict deterministically; no labeling
  needed.

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

### The two capture paths

Before walking the user through manual scenario generation, check
which capture path fits:

**Path A — user has a live in-process agent (Python callable):**
prefer `trainforge record --auto`. One command, no questions, scenario
file written in 5 seconds. Tell them:

```bash
trainforge record \
  --agent <their module:callable> \
  --output scenarios/<short-name>.json \
  --auto
```

The REPL opens. They chat with their agent. They type `:save`. Done.
The resulting JSON defaults to `may_diverge: true` on every turn (the
20 NLP-consistency rubric will judge each), no outcome checks (they
can add later by editing the JSON), auto-generated scenario id /
name.

This is the recommended default for any user who can `python -c
"from their_module import their_agent"`. Drops first-capture friction
from minutes to seconds.

**Path B — user has a transcript already (or no live agent):**
fall back to the manual generation workflow below (Steps 1-7). The
user pastes a conversation, you walk through the questions, and the
output is the same shape of scenario JSON.

Pick the path before asking any questions. If the user has shown you
their agent code or talked about a live callable, Path A. Otherwise
Path B.

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
