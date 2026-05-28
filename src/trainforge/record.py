"""Capture mode: REPL-driven scenario generation from a live agent.

Run via ``trainforge record --agent module:callable --output scenarios/draft.json``.

The user types messages; the agent replies in-process; TrainForge captures
the full transcript (text + tool calls). On ``:save``, the user is prompted
for the few things only they can answer (``may_diverge`` per turn,
``expected_outcome``, ``outcome_checks``) and a valid scenario JSON is
written to disk.

With ``--auto``, the post-session prompts are skipped entirely. Defaults:

- ``may_diverge: true`` on every agent turn (lenient — the 20 standard
  NLP-consistency checks judge whether the agent rephrased acceptably).
- ``outcome_checks: []`` (no outcome judge; user can add later).
- Auto-generated scenario id, name, and ``expected_outcome`` summary.

Designed for the 5-second hackathon path: chat, save, done. User can
sharpen the JSON later if they want stricter checks.

Implemented as a lightweight ``input()`` prompt loop so there's no
terminal-UX dependency. Single-line input is read after each prompt;
tool calls are displayed inline so the user sees exactly what the
agent did.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from trainforge.transport import InProcessTransport, Message, Role


@dataclass
class _CapturedTurn:
    user_message: str
    agent_text: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_responses: list[dict] = field(default_factory=list)


@dataclass
class _RecordSession:
    transport: InProcessTransport
    history: list[Message] = field(default_factory=list)
    turns: list[_CapturedTurn] = field(default_factory=list)

    async def take_turn(self, user_message: str) -> _CapturedTurn:
        """Send a user message, drive any tool round-trips, return the captured turn."""
        self.history.append({"role": Role.USER, "content": user_message})
        captured = _CapturedTurn(user_message=user_message, agent_text="")

        # Tool round-trip loop: while the agent emits tool_calls, we display
        # them, ask the user for a fake response (just like the runner's
        # expected_response), and feed it back.
        while True:
            reply = await self.transport.chat(list(self.history))
            if not reply.is_tool_round:
                captured.agent_text = reply.text or ""
                self.history.append({"role": Role.AGENT, "content": captured.agent_text})
                return captured

            # Display the tool calls; record them.
            calls_json: list[dict] = [
                {"id": tc.id or "", "name": tc.name, "arguments": tc.arguments}
                for tc in reply.tool_calls
            ]
            captured.tool_calls.extend(calls_json)
            self.history.append(
                {"role": Role.AGENT, "content": reply.text or "", "tool_calls": calls_json}
            )

            # Ask the user for a response per tool call (this becomes the
            # scenario's `expected_response`).
            for tc in reply.tool_calls:
                print(f"  [tool] {tc.name}({json.dumps(tc.arguments)})")
                response = input(f"  ↳ expected response for {tc.name!r} (string): ").strip()
                tool_msg: Message = {
                    "role": Role.TOOL,
                    "tool_call_id": tc.id or "",
                    "name": tc.name,
                    "content": response,
                }
                self.history.append(tool_msg)
                captured.tool_responses.append(
                    {
                        "tool_call_id": tc.id or "",
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "expected_response": response,
                    }
                )


def _slug(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "scenario"


def _prompt(question: str, *, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer or default


def _yesno(question: str, *, default: bool) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        answer = input(f"{question}{suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("  please answer y or n")


def _materialize_scenario(
    session: _RecordSession,
    *,
    scenario_id: str,
    name: str,
    description: str,
    tags: list[str],
    expected_outcome: str,
    outcome_checks: list[str],
    per_turn_may_diverge: list[bool],
    per_turn_checks: list[list[str]],
) -> dict:
    turns: list[dict] = []
    for i, captured in enumerate(session.turns):
        turns.append({"role": "user", "message": captured.user_message})

        agent_turn: dict[str, Any] = {
            "role": "agent",
            "golden_response": captured.agent_text,
            "may_diverge": per_turn_may_diverge[i],
        }
        if captured.tool_calls:
            agent_turn["tool_loops"] = [
                {
                    "ordered": False,
                    "tools": [
                        {
                            "name": resp["name"],
                            "arguments_schema": {
                                k: {"type": _infer_type(v), "expected": v}
                                for k, v in resp["arguments"].items()
                            },
                            "expected_response": resp["expected_response"],
                        }
                        for resp in captured.tool_responses
                    ],
                }
            ]
        if per_turn_checks[i]:
            agent_turn["checks"] = per_turn_checks[i]
        turns.append(agent_turn)

    scenario: dict[str, Any] = {
        "id": scenario_id,
        "name": name,
        "description": description,
        "tags": tags,
        "turns": turns,
        "expected_outcome": expected_outcome,
    }
    if outcome_checks:
        scenario["outcome_checks"] = outcome_checks
    return {"version": "2.0", "scenarios": [scenario]}


def _infer_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "any"


def run_record_loop(
    *,
    agent_callable: Any,
    agent_spec: str,
    output_path: str,
    timeout_seconds: float = 30.0,
    auto: bool = False,
) -> int:
    """Drive the REPL; write the scenario JSON to ``output_path`` on save.

    Return code is 0 on save, 1 on abort. The caller (CLI) maps this to
    the process exit code.

    When ``auto=True`` the post-session prompts are skipped on
    ``:save``: every agent turn defaults to ``may_diverge: true`` and
    the scenario header is auto-generated.
    """
    transport = InProcessTransport(
        agent=agent_callable, timeout_seconds=timeout_seconds, source=agent_spec
    )
    session = _RecordSession(transport=transport)

    print("trainforge record: chat with your agent. Commands:")
    print("  :save     finalize and write the scenario file")
    print("  :quit     exit without saving")
    print("  :help     show this message")
    if auto:
        print()
        print("  [auto mode: :save will skip the post-session prompts]")
    print()

    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            print("aborting (EOF)")
            return 1
        except KeyboardInterrupt:
            print()
            print("aborting (Ctrl-C)")
            return 1

        if not line:
            continue
        if line in {":quit", ":q"}:
            print("aborting without saving")
            return 1
        if line in {":help", ":h", ":?"}:
            print("  :save     finalize and write the scenario file")
            print("  :quit     exit without saving")
            continue
        if line == ":save":
            if not session.turns:
                print("nothing to save yet")
                continue
            if auto:
                return _auto_finalize_and_save(session, output_path)
            return _finalize_and_save(session, output_path)

        # Otherwise: treat as a user message.
        try:
            captured = asyncio.run(session.take_turn(line))
        except Exception as exc:  # noqa: BLE001
            print(f"agent error: {type(exc).__name__}: {exc}")
            print("  (turn discarded; you can keep going or :quit)")
            # Roll back history to before the failed user message.
            session.history = session.history[:-1]
            continue

        session.turns.append(captured)
        if captured.tool_calls:
            print(f"agent> [{len(captured.tool_calls)} tool call(s) executed]")
        print(f"agent> {captured.agent_text}")


def _auto_finalize_and_save(session: _RecordSession, output_path: str) -> int:
    """Zero-question save path. Used by ``trainforge record --auto``.

    Sensible defaults:

    - ``may_diverge: true`` on every agent turn. The 20 standard
      NLP-consistency checks judge each turn at run time; users can
      tighten any specific turn to ``may_diverge: false`` later by
      editing the JSON.
    - ``outcome_checks: []`` (no outcome judge by default; user can add).
    - Scenario id / name / ``expected_outcome`` auto-generated.

    The point: chat, ``:save``, run. No questions. Iterate later.
    """
    now = datetime.now()
    timestamp_id = now.strftime("%Y%m%d-%H%M%S")
    timestamp_display = now.strftime("%Y-%m-%d %H:%M")

    # Use the final agent reply (first 80 chars) as a human-readable
    # placeholder for expected_outcome. It's not a real outcome
    # description, but it gives downstream readers a hint of what
    # happened without an LLM in the loop.
    final_agent_text = (
        session.turns[-1].agent_text if session.turns else ""
    ).strip()
    expected_outcome = (
        f"Recorded session — final reply: {final_agent_text[:80]}..."
        if len(final_agent_text) > 80
        else f"Recorded session — final reply: {final_agent_text}"
    ) or f"Recorded session ({timestamp_display})"

    scenario_doc = _materialize_scenario(
        session,
        scenario_id=f"sc-auto-{timestamp_id}",
        name=f"Recorded session {timestamp_display}",
        description=(
            "Auto-captured by `trainforge record --auto`. "
            "Every agent turn defaults to may_diverge=true (the 20 "
            "standard NLP-consistency checks judge each reply). Edit "
            "this file to tighten specific turns to exact-match or to "
            "add outcome_checks."
        ),
        tags=["auto"],
        expected_outcome=expected_outcome,
        outcome_checks=[],
        per_turn_may_diverge=[True] * len(session.turns),
        per_turn_checks=[[] for _ in session.turns],
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scenario_doc, indent=2), encoding="utf-8")
    print()
    print(f"wrote {output_path}")
    print(f"  scenario id:   sc-auto-{timestamp_id}")
    print(f"  turns:         {len(session.turns)}")
    print(f"  may_diverge:   true on every agent turn (edit to tighten)")
    print(f"  outcome_checks: [] (edit to add)")
    print()
    print("Run it with:")
    print(
        f"  trainforge run --agent <your agent spec> --scenarios {output_path} "
        f"--output results.json"
    )
    return 0


def _finalize_and_save(session: _RecordSession, output_path: str) -> int:
    print()
    print("Finalize scenario.")
    scenario_id = _prompt("scenario id", default="sc-recorded-001")
    name = _prompt("scenario name", default="Recorded session")
    description = _prompt("description (optional)", default="")
    tags_raw = _prompt("tags (comma-separated, optional)", default="")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    per_turn_may_diverge: list[bool] = []
    per_turn_checks: list[list[str]] = []
    for i, captured in enumerate(session.turns):
        preview = captured.agent_text[:80]
        print()
        print(f"Turn {i + 1}: agent said {preview!r}")
        may_diverge = _yesno(
            "  allow rephrasing (may_diverge=True)? "
            "[N for scripted/legal/compliance, Y for free-form]",
            default=False,
        )
        per_turn_may_diverge.append(may_diverge)

        checks: list[str] = []
        if _yesno("  add custom semantic checks for this turn?", default=False):
            print("  enter one check per line; blank to finish")
            while True:
                check = input("    check: ").strip()
                if not check:
                    break
                checks.append(check)
        per_turn_checks.append(checks)

    print()
    expected_outcome = _prompt(
        "expected outcome of the whole conversation, in one sentence",
        default="Outcome achieved.",
    )
    outcome_checks: list[str] = []
    if _yesno("add binary outcome checks?", default=True):
        print("enter one outcome check per line; blank to finish")
        while True:
            check = input("  check: ").strip()
            if not check:
                break
            outcome_checks.append(check)

    scenario_doc = _materialize_scenario(
        session,
        scenario_id=scenario_id,
        name=name,
        description=description,
        tags=tags,
        expected_outcome=expected_outcome,
        outcome_checks=outcome_checks,
        per_turn_may_diverge=per_turn_may_diverge,
        per_turn_checks=per_turn_checks,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scenario_doc, indent=2), encoding="utf-8")
    print()
    print(f"wrote {output_path}")
    print("run it with:")
    print(
        f"  trainforge run --agent <your agent spec> --scenarios {output_path} "
        f"--output results.json"
    )
    return 0
