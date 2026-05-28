# TrainForge (Claude Code skill)

Full TrainForge interface for the coding agent. One install; the coding
agent (Claude Code / Cursor / etc.) drives capture, run, label, score,
diff, debug, A/B testing, HTML reports, pytest, and CI from inside the
user's session.

The user describes what they want — "test my agent", "I changed the
prompt, did anything break?", "is this reliable?" — and the skill picks
the right TrainForge commands, runs them, surfaces results. **The skill
never modifies the user's agent code.** It's a test driver, not a code
editor.

## Install

### Claude Code

```bash
mkdir -p ~/.claude/skills/trainforge
cp -r agent-skills/trainforge/* ~/.claude/skills/trainforge/
```

Then in Claude Code, either type `/trainforge` explicitly, or say
something matching the description ("test my agent", "I changed the
prompt", etc.) and Claude Code will auto-invoke.

### Cursor / any coding agent

```bash
mkdir -p .cursor/rules
cp agent-skills/trainforge/SKILL.md .cursor/rules/trainforge.md
```

Or paste the contents of `SKILL.md` directly into your coding agent's
chat at the start of a session.

## The 5-second capture (default flow)

```
# 1. Capture a conversation as a scenario (no questions).
trainforge record --agent your_module:your_fn --auto --output scenarios/x.json

# 2. Run + capture (no LLM key needed).
trainforge run --scenarios scenarios/x.json --agent your_module:your_fn \
               --output results.json --no-judge

# 3. The coding agent labels pending checks using the 20-NLP rubric
#    in SKILL.md (no external API needed — your coding agent IS the
#    judge).

# 4. Deterministic scorer aggregates the labels.
trainforge rescore --results results.json
```

The full `SKILL.md` walks the coding agent through every TrainForge
workflow — A/B testing, failure debugging, consistency scoring, HTML
reports, multi-agent (`node_assertions`), pytest integration, CI
setup. The README you're reading is the elevator pitch; SKILL.md is
the manual.

## The HARD RULE

> The skill is a test driver. The agent is the user's product.
> The skill never touches the product.

The skill can write scenario JSON files and fill in pending labels in
results JSON files. It will NEVER edit the user's agent code, prompts,
model config, tool definitions, or tool implementations. On a failing
test, it diagnoses and recommends; the user decides whether to fix the
agent or update the scenario.

Full statement: see "HARD RULE" near the top of `SKILL.md`.

## Examples

- [`examples/transcript-restaurant-booking.md`](examples/transcript-restaurant-booking.md) — sample input transcript for manual scenario generation.
- [`examples/scenario-restaurant-booking.json`](examples/scenario-restaurant-booking.json) — the scenario JSON the skill produces from it.

## Pricing

Free and open source. The TrainForge runner is free and open source.
For the default `--no-judge` + coding-agent-as-judge flow, no API key
is needed — your coding agent does the labeling. Configured keys
(`OPENAI_API_KEY` + `OPENAI_API_URL`) are only needed for CI runs
where no coding agent is in the loop.
