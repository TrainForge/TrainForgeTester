"""Core scenario runner - the heart of TrainForge.

v0.2 deterministic-first model:

1. **Tool loops are checked deterministically.** Each declared
   :class:`ToolLoop` runs *before* the agent's text turn. Tool name,
   arguments, type, ordering: all Python-equality checks. No LLM in this
   path. See :mod:`trainforge.tool_validator`.

2. **Text turn evaluation has two paths**, gated by ``AgentTurn.may_diverge``:

   - ``may_diverge=False`` (default) - the runner does
     ``actual_response == golden_response`` and writes ``exact_match`` to
     the result. Zero LLM calls. Use this for curated/scripted replies
     (legal, FAQ, compliance copy).
   - ``may_diverge=True`` - the runner composes the 20 standard
     NLP-consistency checks (:mod:`trainforge.standard_checks`) plus any
     per-scenario custom checks into ONE batched LLM call. Output decoded
     positionally back into ``standard_check_results`` and ``checks``.

3. **Outcome checks** still run via LLM at the end of each scenario, also
   in the compact batched format.

4. **Golden injection** is unchanged: every agent turn the agent sees the
   golden text + golden tool_calls + expected_response in history,
   regardless of what it actually said. Keeps each turn evaluated against
   a clean context.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Callable

from trainforge.agent_client import AgentClient, AgentReply, Message
from trainforge.errors import (
    AgentError,
    AgentTimeoutError,
    AgentUnreachableError,
    EvaluationError,
)
from trainforge.evaluation import (
    OutcomeEval,
    TurnEval,
    evaluate_outcome,
    evaluate_turn,
)
from trainforge.llm.base import LLMClient
from trainforge.schema import (
    AgentTurn,
    CheckResult,
    ExpectedTool,
    OutcomeResult,
    Scenario,
    ScenarioResult,
    ScenarioRunResult,
    StandardCheckResult,
    ToolArgumentSchema,
    ToolCallRecord,
    ToolLoop,
    TurnResult,
    UserTurn,
)
from trainforge.scoring import aggregate_consistency, classify_scenario_run
from trainforge.standard_checks import STANDARD_CHECKS
from trainforge.tool_validator import AgentToolCall, LoopMatcher, MatchDecision

log = logging.getLogger(__name__)


MAX_TOOL_ROUNDS_PER_LOOP = 8
"""Safety net against an agent that keeps emitting novel tool_calls forever
instead of converging on the expected set. If a loop takes more than this
many HTTP rounds, remaining expected tools are marked ``missing`` and the
runner moves on."""


@dataclass
class ScenarioRunner:
    """Runs one scenario N times using golden injection."""

    agent: AgentClient
    llm: LLMClient

    def run_scenario(
        self,
        scenario: Scenario,
        runs: int,
        *,
        on_turn_complete: Callable[[], None] | None = None,
    ) -> ScenarioResult:
        """Execute ``scenario`` ``runs`` times and aggregate the results.

        ``on_turn_complete`` is called once per *agent* turn processed
        (success or failure), useful for driving a progress bar. It is not
        called for user turns (which do no work beyond appending a
        message to history).
        """
        run_results: list[ScenarioRunResult] = []
        for run_index in range(runs):
            run_results.append(
                self._single_run(
                    scenario, run_index, on_turn_complete=on_turn_complete
                )
            )

        consistency, inconsistent = aggregate_consistency(run_results)
        return ScenarioResult(
            scenario_id=scenario.id,
            name=scenario.name,
            description=scenario.description,
            tags=scenario.tags,
            runs=run_results,
            consistency=consistency,
            inconsistent=inconsistent,
        )

    # ------------------------------------------------------------------
    # Per-run execution
    # ------------------------------------------------------------------

    def _single_run(
        self,
        scenario: Scenario,
        run_index: int,
        *,
        on_turn_complete: Callable[[], None] | None = None,
    ) -> ScenarioRunResult:
        golden_history: list[Message] = []
        actual_history: list[Message] = []

        turn_results: list[TurnResult] = []
        pending_user: UserTurn | None = None

        for i, turn in enumerate(scenario.turns):
            if isinstance(turn, UserTurn):
                pending_user = turn
                golden_history.append({"role": "user", "content": turn.message})
                actual_history.append({"role": "user", "content": turn.message})
                continue

            assert isinstance(turn, AgentTurn)
            assert pending_user is not None, (
                "schema validation should have ensured user-first alternation"
            )

            turn_result = self._execute_agent_turn(
                scenario=scenario,
                turn_index=i,
                agent_turn=turn,
                user_turn=pending_user,
                golden_history=golden_history,
                actual_history=actual_history,
            )
            turn_results.append(turn_result)
            if on_turn_complete is not None:
                on_turn_complete()

            # Golden injection: future turns see the GOLDEN text response.
            golden_history.append(
                {"role": "agent", "content": turn.golden_response}
            )
            pending_user = None

            if turn_result.error == "agent_unreachable":
                return _unreachable_run(run_index, turn_results, turn_result.error)

        outcome = self._evaluate_outcome(scenario, actual_history)
        status = classify_scenario_run(turn_results, outcome)

        return ScenarioRunResult(
            run_index=run_index,
            status=status,
            turns=turn_results,
            outcome=outcome,
        )

    # ------------------------------------------------------------------
    # Agent turn execution
    # ------------------------------------------------------------------

    def _execute_agent_turn(
        self,
        *,
        scenario: Scenario,
        turn_index: int,
        agent_turn: AgentTurn,
        user_turn: UserTurn,
        golden_history: list[Message],
        actual_history: list[Message],
    ) -> TurnResult:
        # --- Tool loops (deterministic; may be empty) --------------------
        tool_records: list[ToolCallRecord] = []
        for loop_index, loop in enumerate(agent_turn.tool_loops):
            try:
                self._run_tool_loop(
                    scenario_id=scenario.id,
                    turn_index=turn_index,
                    loop_index=loop_index,
                    loop=loop,
                    golden_history=golden_history,
                    actual_history=actual_history,
                    out_records=tool_records,
                )
            except _AgentAbort as abort:
                return _tool_abort_turn_result(
                    turn_index=turn_index,
                    user_turn=user_turn,
                    agent_turn=agent_turn,
                    tool_records=tool_records,
                    abort=abort,
                )

        # --- Text turn ---------------------------------------------------
        base = _TurnBase(
            turn_index=turn_index,
            user_message=user_turn.message,
            golden_response=agent_turn.golden_response,
            may_diverge=agent_turn.may_diverge,
            divergence_note=agent_turn.divergence_note,
            tool_calls=tool_records,
        )

        try:
            reply = self.agent.chat(list(golden_history))
        except AgentTimeoutError as exc:
            log.warning("agent timeout on scenario=%s turn=%d", scenario.id, turn_index)
            actual_history.append({"role": "agent", "content": ""})
            return base.to_error_result("agent_timeout", str(exc))
        except AgentError as exc:
            log.warning("agent error on scenario=%s turn=%d: %s", scenario.id, turn_index, exc)
            actual_history.append({"role": "agent", "content": ""})
            return base.to_error_result("agent_error", str(exc))
        except AgentUnreachableError as exc:
            log.error("agent unreachable: %s", exc)
            actual_history.append({"role": "agent", "content": ""})
            tr = base.to_error_result("agent_error", str(exc))
            return TurnResult(**{**tr.model_dump(), "error": "agent_unreachable"})

        actual_text = reply.text or ""
        actual_history.append({"role": "agent", "content": actual_text})

        if not actual_text.strip():
            # Empty agent response: every check fails by definition.
            failed_customs = [
                CheckResult(check=c, passed=False, explanation="agent returned empty text response")
                for c in agent_turn.checks
            ]
            return TurnResult(
                turn_index=turn_index,
                user_message=user_turn.message,
                golden_response=agent_turn.golden_response,
                actual_response="",
                may_diverge=agent_turn.may_diverge,
                divergence_note=agent_turn.divergence_note,
                tool_calls=tool_records,
                status="empty_response",
                exact_match=False if not agent_turn.may_diverge else None,
                standard_check_results=[],
                checks=failed_customs,
            )

        # ---- DETERMINISTIC PATH: exact text match ----------------------
        if not agent_turn.may_diverge:
            return self._evaluate_exact_match(
                base=base,
                agent_turn=agent_turn,
                user_turn=user_turn,
                actual_text=actual_text,
            )

        # ---- LLM PATH: 20 standard NLP checks + custom checks ----------
        return self._evaluate_with_llm(
            base=base,
            agent_turn=agent_turn,
            user_turn=user_turn,
            actual_text=actual_text,
            scenario_id=scenario.id,
            turn_index=turn_index,
        )

    # ------------------------------------------------------------------
    # Text-turn evaluation paths
    # ------------------------------------------------------------------

    def _evaluate_exact_match(
        self,
        *,
        base: "_TurnBase",
        agent_turn: AgentTurn,
        user_turn: UserTurn,
        actual_text: str,
    ) -> TurnResult:
        """Deterministic path: actual must equal golden verbatim.

        Custom checks are still evaluated (they exist outside the
        equivalence question - they may probe specific structural facts a
        scenario author cares about even on a verbatim turn). If there are
        no custom checks, the LLM is never called for this turn.
        """
        passed = actual_text == agent_turn.golden_response
        custom_results = self._evaluate_custom_only(
            agent_turn=agent_turn,
            user_turn=user_turn,
            actual_text=actual_text,
        )
        return TurnResult(
            turn_index=base.turn_index,
            user_message=base.user_message,
            golden_response=base.golden_response,
            actual_response=actual_text,
            may_diverge=False,
            divergence_note=base.divergence_note,
            tool_calls=base.tool_calls,
            status="evaluated",
            exact_match=passed,
            standard_check_results=[],
            checks=custom_results,
        )

    def _evaluate_custom_only(
        self,
        *,
        agent_turn: AgentTurn,
        user_turn: UserTurn,
        actual_text: str,
    ) -> list[CheckResult]:
        if not agent_turn.checks:
            return []
        try:
            turn_eval = evaluate_turn(
                self.llm,
                golden_response=agent_turn.golden_response,
                actual_response=actual_text,
                user_message=user_turn.message,
                questions=list(agent_turn.checks),
            )
        except EvaluationError as exc:
            log.warning("custom-check eval error: %s", exc)
            return [
                CheckResult(check=c, passed=False, explanation="eval_error")
                for c in agent_turn.checks
            ]
        return [
            CheckResult(check=c.question, passed=c.passed, explanation=c.explanation)
            for c in turn_eval.checks
        ]

    def _evaluate_with_llm(
        self,
        *,
        base: "_TurnBase",
        agent_turn: AgentTurn,
        user_turn: UserTurn,
        actual_text: str,
        scenario_id: str,
        turn_index: int,
    ) -> TurnResult:
        """LLM path used when ``may_diverge=True``.

        Composes the 20 standard NLP-consistency questions + the per-scenario
        custom checks into ONE batched call. Splits the result back into
        the two buckets by index.
        """
        standard_questions = [c.question for c in STANDARD_CHECKS]
        custom_questions = list(agent_turn.checks)
        questions = standard_questions + custom_questions

        try:
            turn_eval: TurnEval = evaluate_turn(
                self.llm,
                golden_response=agent_turn.golden_response,
                actual_response=actual_text,
                user_message=user_turn.message,
                questions=questions,
            )
        except EvaluationError as exc:
            log.warning(
                "turn eval error on scenario=%s turn=%d: %s",
                scenario_id,
                turn_index,
                exc,
            )
            standard_fail = [
                StandardCheckResult(
                    id=c.id,
                    question=c.question,
                    passed=False,
                    explanation="eval_error",
                )
                for c in STANDARD_CHECKS
            ]
            custom_fail = [
                CheckResult(check=c, passed=False, explanation="eval_error")
                for c in agent_turn.checks
            ]
            return TurnResult(
                turn_index=turn_index,
                user_message=user_turn.message,
                golden_response=agent_turn.golden_response,
                actual_response=actual_text,
                may_diverge=True,
                divergence_note=agent_turn.divergence_note,
                tool_calls=base.tool_calls,
                status="eval_error",
                exact_match=None,
                standard_check_results=standard_fail,
                checks=custom_fail,
                error=str(exc),
            )

        # Split positionally: first len(STANDARD_CHECKS) are standard, rest custom.
        n_std = len(STANDARD_CHECKS)
        std_evals = turn_eval.checks[:n_std]
        custom_evals = turn_eval.checks[n_std:]

        standard_results = [
            StandardCheckResult(
                id=STANDARD_CHECKS[i].id,
                question=STANDARD_CHECKS[i].question,
                passed=ev.passed,
                explanation=ev.explanation,
            )
            for i, ev in enumerate(std_evals)
        ]
        custom_results = [
            CheckResult(check=c.question, passed=c.passed, explanation=c.explanation)
            for c in custom_evals
        ]
        return TurnResult(
            turn_index=turn_index,
            user_message=user_turn.message,
            golden_response=agent_turn.golden_response,
            actual_response=actual_text,
            may_diverge=True,
            divergence_note=agent_turn.divergence_note,
            tool_calls=base.tool_calls,
            status="evaluated",
            exact_match=None,
            standard_check_results=standard_results,
            checks=custom_results,
        )

    # ------------------------------------------------------------------
    # Tool loop execution (unchanged from v0.1)
    # ------------------------------------------------------------------

    def _run_tool_loop(
        self,
        *,
        scenario_id: str,
        turn_index: int,
        loop_index: int,
        loop: ToolLoop,
        golden_history: list[Message],
        actual_history: list[Message],
        out_records: list[ToolCallRecord],
    ) -> None:
        matcher = LoopMatcher(loop)
        rounds = 0

        while not matcher.done and rounds < MAX_TOOL_ROUNDS_PER_LOOP:
            try:
                reply = self.agent.chat(list(golden_history))
            except AgentTimeoutError as exc:
                raise _AgentAbort("agent_timeout", str(exc)) from exc
            except AgentError as exc:
                raise _AgentAbort("agent_error", str(exc)) from exc
            except AgentUnreachableError as exc:
                raise _AgentAbort("agent_unreachable", str(exc)) from exc

            if not reply.is_tool_round:
                text = reply.text or ""
                actual_history.append({"role": "agent", "content": text})
                break

            rounds += 1

            actual_history.append(
                {
                    "role": "agent",
                    "content": reply.text or "",
                    "tool_calls": [_call_to_dict(c) for c in reply.tool_calls],
                }
            )

            decisions = matcher.process(reply.tool_calls)
            for decision in decisions:
                out_records.append(
                    _decision_to_record(decision, loop_index=loop_index)
                )

            injected_calls: list[dict] = []
            injected_responses: list[Message] = []
            for decision in decisions:
                if decision.matched_tool is None:
                    continue
                call_id = _ensure_call_id(decision.agent_call.id)
                injected_calls.append(
                    _expected_call_to_dict(decision.matched_tool, call_id)
                )
                injected_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": decision.matched_tool.name,
                        "content": decision.matched_tool.expected_response,
                    }
                )

            if injected_calls:
                golden_history.append(
                    {"role": "agent", "content": None, "tool_calls": injected_calls}
                )
                golden_history.extend(injected_responses)
                actual_history.extend(injected_responses)

        for position in matcher.finalize():
            expected = loop.tools[position]
            out_records.append(
                ToolCallRecord(
                    loop_index=loop_index,
                    position=position,
                    expected_name=expected.name,
                    expected_arguments_schema=expected.arguments_schema,
                    actual_name=None,
                    status="missing",
                    explanation="agent stopped emitting tool_calls before this tool was invoked",
                )
            )
            call_id = _ensure_call_id(None)
            golden_history.append(
                {
                    "role": "agent",
                    "content": None,
                    "tool_calls": [_expected_call_to_dict(expected, call_id)],
                }
            )
            response_msg: Message = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": expected.name,
                "content": expected.expected_response,
            }
            golden_history.append(response_msg)
            actual_history.append(response_msg)

        if rounds >= MAX_TOOL_ROUNDS_PER_LOOP and not matcher.done:  # pragma: no cover
            log.warning(
                "scenario=%s turn=%d loop=%d hit MAX_TOOL_ROUNDS_PER_LOOP",
                scenario_id,
                turn_index,
                loop_index,
            )

    # ------------------------------------------------------------------
    # Outcome eval
    # ------------------------------------------------------------------

    def _evaluate_outcome(
        self, scenario: Scenario, actual_history: list[Message]
    ) -> OutcomeResult:
        if not scenario.outcome_checks:
            return OutcomeResult(status="evaluated", checks=[])

        try:
            outcome: OutcomeEval = evaluate_outcome(
                self.llm,
                conversation=[dict(m) for m in actual_history],
                expected_outcome=scenario.expected_outcome,
                outcome_checks=list(scenario.outcome_checks),
            )
        except EvaluationError as exc:
            log.warning("outcome eval error on scenario=%s: %s", scenario.id, exc)
            return OutcomeResult(
                status="eval_error",
                checks=[
                    CheckResult(check=c, passed=False, explanation="eval_error")
                    for c in scenario.outcome_checks
                ],
                error=str(exc),
            )

        return OutcomeResult(
            status="evaluated",
            checks=[
                CheckResult(check=c.question, passed=c.passed, explanation=c.explanation)
                for c in outcome.checks
            ],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TurnBase:
    turn_index: int
    user_message: str
    golden_response: str
    may_diverge: bool
    divergence_note: str | None
    tool_calls: list[ToolCallRecord]

    def to_error_result(self, status: str, error: str) -> TurnResult:
        return TurnResult(
            turn_index=self.turn_index,
            user_message=self.user_message,
            golden_response=self.golden_response,
            actual_response="",
            may_diverge=self.may_diverge,
            divergence_note=self.divergence_note,
            tool_calls=self.tool_calls,
            status=status,  # type: ignore[arg-type]
            exact_match=False if not self.may_diverge else None,
            standard_check_results=[],
            checks=[],
            error=error,
        )


class _AgentAbort(Exception):
    """Raised from ``_run_tool_loop`` when the agent HTTP layer gave up."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _tool_abort_turn_result(
    *,
    turn_index: int,
    user_turn: UserTurn,
    agent_turn: AgentTurn,
    tool_records: list[ToolCallRecord],
    abort: _AgentAbort,
) -> TurnResult:
    is_unreachable = abort.status == "agent_unreachable"
    base = _TurnBase(
        turn_index=turn_index,
        user_message=user_turn.message,
        golden_response=agent_turn.golden_response,
        may_diverge=agent_turn.may_diverge,
        divergence_note=agent_turn.divergence_note,
        tool_calls=tool_records,
    )
    status = "agent_error" if is_unreachable else abort.status
    tr = base.to_error_result(status, abort.message)
    if is_unreachable:
        return TurnResult(**{**tr.model_dump(), "error": "agent_unreachable"})
    return tr


def _unreachable_run(
    run_index: int, turn_results: list[TurnResult], error: str
) -> ScenarioRunResult:
    return ScenarioRunResult(
        run_index=run_index,
        status="agent_unreachable",
        turns=turn_results,
        outcome=OutcomeResult(status="agent_unreachable", error=error),
        error=error,
    )


def _ensure_call_id(existing: str | None) -> str:
    return existing or f"call_{uuid.uuid4().hex[:12]}"


def _call_to_dict(call: AgentToolCall) -> dict:
    return {
        "id": _ensure_call_id(call.id),
        "name": call.name,
        "arguments": call.arguments,
    }


def _expected_call_to_dict(expected: ExpectedTool, call_id: str) -> dict:
    return {
        "id": call_id,
        "name": expected.name,
        "arguments": _default_arguments(expected.arguments_schema),
    }


def _default_arguments(schema: dict[str, ToolArgumentSchema]) -> dict:
    defaults: dict[str, object] = {}
    for key, arg in schema.items():
        if arg.expected is not None:
            defaults[key] = arg.expected
        else:
            defaults[key] = _default_for(arg.type)
    return defaults


def _default_for(t: str) -> object:
    return {
        "string": "",
        "integer": 0,
        "number": 0.0,
        "boolean": False,
        "array": [],
        "object": {},
        "any": "",
    }.get(t, "")


def _decision_to_record(
    decision: MatchDecision, *, loop_index: int
) -> ToolCallRecord:
    expected = decision.matched_tool
    position = (
        decision.expected_position
        if decision.expected_position is not None
        else -1
    )
    return ToolCallRecord(
        loop_index=loop_index,
        position=position,
        expected_name=expected.name if expected else None,
        expected_arguments_schema=expected.arguments_schema if expected else {},
        actual_name=decision.agent_call.name,
        actual_arguments=dict(decision.agent_call.arguments),
        status=decision.status,
        explanation=decision.explanation,
    )
