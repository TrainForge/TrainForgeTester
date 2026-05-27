"""Capture mode: REPL-driven scenario generation from a live agent.

Run via ``trainforge record --agent module:callable --output scenarios/draft.json``.

The user types messages; the agent replies in-process; TrainForge captures
the full transcript (text + tool calls). On ``:save``, the user is prompted
for the few things only they can answer (``may_diverge`` per turn,
``expected_outcome``, ``outcome_checks``) and a valid scenario JSON is
written to disk.

Built on stdlib ``cmd.Cmd`` so there's no terminal-UX dependency. Single
greedy multi-line input is read via ``input()`` after each prompt; tool
calls are displayed inline so the user sees exactly what the agent did.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trainforge.transport import AgentReply, InProcessTransport, Message, Role


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
) -> int:
    """Drive the REPL; write the scenario JSON to ``output_path`` on save.

    Return code is 0 on save, 1 on abort. The caller (CLI) maps this to
    the process exit code.
    """
    transport = InProcessTransport(
        agent=agent_callable, timeout_seconds=timeout_seconds, source=agent_spec
    )
    session = _RecordSession(transport=transport)

    print("trainforge record: chat with your agent. Commands:")
    print("  :save     finalize and write the scenario file")
    print("  :quit     exit without saving")
    print("  :help     show this message")
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
