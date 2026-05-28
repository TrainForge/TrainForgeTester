# Concepts

The mental model for what TrainForge is and what makes it different.

## What TrainForge does

TrainForge runs multi-turn conversational scenarios against your AI agent
and produces a pass/fail report. Scenarios are JSON. The agent is either
a Python callable in your process or an HTTP endpoint somewhere.

Three things make the run useful:

1. **Tool calls are checked deterministically.** Each scenario declares
   which tools the agent should call and what their arguments should be.
   Tool name and declared argument values are compared by Python
   equality. No LLM in the path; results don't drift between runs.

2. **Agent text has two paths per turn.** A `may_diverge: false` turn
   demands exact-match between the agent's reply and the scenario's
   `golden_response`. A `may_diverge: true` turn delegates to 20 binary
   NLP-consistency questions plus any per-turn custom checks, all
   batched into one LLM call.

3. **Outcome eval runs once per scenario** over the actual transcript.
   The judge answers binary "did the agent achieve X?" questions you
   declared as `outcome_checks`.

Every LLM-judged claim in the system is a yes/no question with a stable
id. There are no 0-1 quality scores anywhere in the runner.

## Why deterministic-first

Most eval frameworks score agent quality with an LLM judge that returns
a 0-1 float. That score moves run to run on the exact same input —
not because the agent changed but because the judge is non-deterministic.
TrainForge sidesteps this for everything that doesn't actually need a
judge.

| Failure surface | How TrainForge checks it |
|---|---|
| Wrong tool called | Python string equality on tool name. |
| Wrong tool arguments | Python `==` on the declared `expected` literal + type check. |
| Tool called in wrong slot | Position match in ordered loop, or set match in unordered. |
| Verbatim agent reply mismatch | Python `actual == golden` on the response text. |
| Required tool never called | Loop position never matched after the agent finished the round. |
| End-state of the conversation | Per-scenario `outcome_checks` (LLM-judged binary). |
| Free-form agent rephrasing of golden | 20 fixed binary NLP-consistency questions per turn (LLM-judged). |

The result: when CI says "pass," it actually passed. When it says
"fail," you can point at exactly which line of the contract broke.

## Golden injection

After every agent turn, the runner replaces the agent's actual reply
with the scenario's `golden_response` in the history that gets sent on
the next round. The agent at turn 4 always sees the correct
turn-1-through-3 transcript, never the divergent one it actually
produced.

This is what makes per-turn evaluation honest: a wrong reply at turn 2
doesn't corrupt the evaluation of turns 4, 6, 8. Each turn is judged
against a clean context.

```
Real conversation:                       Agent sees on next round:

User: book a table                       User: book a table
Agent: how many?           (golden)      Agent: how many?           (golden)
User: 2                                  User: 2
Agent: WRONG REPLY HERE    (actual)      Agent: <golden turn 3>     (injected)
User: indoor please                      User: indoor please
Agent: ???                               Agent: ???
                                         (sees a clean golden history)
```

The same applies to tool calls: if the agent calls the wrong tool, the
runner records the failure, then injects the *expected* tool's
declared `expected_response` into history so the rest of the
conversation still has the right shape.

## The 20 standard NLP-consistency checks

Applied automatically to every `may_diverge: true` agent turn. All 20
are binary comparisons of the actual reply against the golden reply:

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
| `comparable_length` | Length within ~0.5× to 2× of golden. |
| `same_persona` | Same persona/voice; neither breaks character. |
| `same_information_order` | Same ordering of major information units. |

Source of truth: [`src/trainforge/standard_checks.py`](../src/trainforge/standard_checks.py).
The list is hard-coded and stable; check `id` values are part of the
public results contract.

## Compact LLM wire format

Every LLM call uses the same compact JSON shape so per-turn evaluation
stays cheap (~30 output tokens for a fully-passing turn):

```json
{"r": [1, 1, 0, 1, 1, ...], "f": {"3": "different language"}}
```

- `r` is a positional array of `1` (pass) and `0` (fail). Length equals
  the number of asked questions.
- `f` maps the 1-based index of failed questions to a brief reason.
  Pass questions get no entry.

This replaces the v1.x "1-5 score + divergence_type + per-check JSON"
format which used 150-500 tokens per call.

## Related

- [Scenario format](scenarios.md) — schema reference.
- [Agent contracts](agents.md) — HTTP wire + in-process callable shape.
- [CLI reference](cli.md) — every subcommand.
