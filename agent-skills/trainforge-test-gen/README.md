# TrainForge Test Generator (agent skill)

Turn one manual conversation with your AI agent into a TrainForge regression
scenario. You did the test once by hand. This skill captures it as a
deterministic test you can re-run on every prompt or model change.

## What you get

Paste a transcript (or point at a file). The skill walks you through a few
short questions, then writes a `scenarios/<name>.json` file you can run with:

```bash
export OPENAI_API_KEY=<your-openai-compatible-key>
export OPENAI_API_URL=<your-openai-compatible-base-url>   # e.g. https://api.openai.com/v1

trainforge run \
  --scenarios scenarios/<name>.json \
  --agent-url <your-agent-url> \
  --output results.json
```

`OPENAI_API_KEY` and `OPENAI_API_URL` are required even when every turn is
exact-match — the runner uses them for the per-scenario outcome check.
TrainForge supports any OpenAI-compatible endpoint (Anthropic, NVIDIA,
Cerebras, OpenAI itself).

Tool calls are checked by Python equality. Tool arguments are checked by
literal `==` on whatever values you mark as part of the contract. Text turns
are either exact-match or evaluated by 20 binary NLP-consistency checks, per
turn. No 0-1 quality scores. No judge drift.

## Install

### Option A — Claude Code

Copy the skill into your Claude Code skills directory:

```bash
mkdir -p ~/.claude/skills/trainforge-test-gen
cp -r agent-skills/trainforge-test-gen/* ~/.claude/skills/trainforge-test-gen/
```

Then in Claude Code, type `/trainforge-test-gen` or paste a transcript and
ask Claude to "turn this into a TrainForge scenario."

### Option B — Cursor / any coding agent

Copy the contents of `SKILL.md` into your project's `.cursor/rules/` directory
(or wherever your agent loads system prompts from):

```bash
mkdir -p .cursor/rules
cp agent-skills/trainforge-test-gen/SKILL.md .cursor/rules/trainforge-test-gen.md
```

### Option C — paste-into-chat (no install)

Open `SKILL.md`. Copy the whole file. Paste it into your coding agent's chat,
then add: "Now turn this transcript into a TrainForge scenario:" followed by
the transcript.

## How to use it

1. Have a working agent (HTTP endpoint that follows the agent contract in
   TrainForge's main README).
2. Do one real manual conversation with it. Capture the transcript — copy
   the chat, export the messages JSON, or screenshot and OCR if you must.
3. Invoke the skill and hand it the transcript.
4. Answer 4-8 short questions (one per turn, plus the outcome check).
5. The skill writes `scenarios/<name>.json`.
6. Run `trainforge run` against your agent.
7. Open the HTML report.

From there, every prompt change, model swap, or refactor you ship gets
regression-tested against this scenario. When something breaks, the report
tells you exactly which turn, which tool, which argument, and what the
agent did differently.

## Recommended workflow

- One scenario per real-world flow you care about. Start with the three
  flows that hurt most when they break.
- Keep scenarios in your repo, in `scenarios/`. They are code. They version
  with your agent.
- Run them in CI on every PR. The deterministic checks have zero LLM cost.
  The NLP-divergence checks cost one batched LLM call per turn.
- When you find a bug in production, capture the transcript, generate a
  scenario from it, fix the agent, re-run. The scenario then guards against
  regression forever.

## What this skill does NOT do

- It doesn't invent data. If the transcript doesn't show a tool call, the
  scenario doesn't assert one.
- It doesn't run TrainForge or hit any LLM. It only writes JSON.
- It doesn't talk to a cloud. Your transcripts and scenarios stay in your
  repo.

## Examples

- `examples/transcript-restaurant-booking.md` — sample input transcript.
- `examples/scenario-restaurant-booking.json` — the scenario the skill
  produces from it.

## Pricing

The skill is free and open source. The TrainForge runner is free and open
source. You bring your own LLM key for the divergence checks (Anthropic,
NVIDIA, or Cerebras supported out of the box).
