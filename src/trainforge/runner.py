"""Core scenario runner - the heart of TrainForge.

Implements testing-spec-v1.md section "Execution Flow" and enforces
"The Golden Injection Mechanism": after each agent turn the GOLDEN response
(not the actual one) is appended to the conversation the agent will see on
subsequent turns. The ACTUAL transcript is kept separately and used only for
outcome evaluation and reporting.

v1.1 tool-call extension: each agent turn may declare ``tool_loops`` that
run *before* the text turn. Within each loop the runner:

- sends the current golden history to the agent,
- parses the agent's tool_calls against the loop's expected tools,
- records pass / wrong_tool / invalid_arguments / unexpected_tool per call,
- **always** injects the GOLDEN tool_call + expected_response into the
  history, regardless of what the agent emitted (same invariant as the text
  golden injection),
- continues until every expected tool is matched or the agent stops
  emitting tool_calls (remaining expected tools are recorded as ``missing``).

Then the text-turn path runs as before.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Callable

from trainforge.agent_client import AgentClient, Message, Role
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
    ArgumentType,
    CheckResult,
    CustomerTurn,
    ExpectedTool,
    OutcomeResult,
    OutcomeStatus,
    Scenario,
    ScenarioResult,
    ScenarioRunResult,
    ScenarioStatus,
    ToolArgumentSchema,
    ToolCallRecord,
    ToolCallStatus,
    ToolLoop,
    TurnResult,
    TurnStatus,
)
from trainforge.scoring import aggregate_consistency, classify_scenario_run
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
        called for customer turns (which do no work beyond appending a
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
        """What the agent sees next turn. Spec: append GOLDEN agent
        responses + GOLDEN tool_call/tool_response pairs."""
        actual_history: list[Message] = []
        """What actually happened. Used for outcome eval + reporting."""

        turn_results: list[TurnResult] = []
        pending_customer: CustomerTurn | None = None

        for i, turn in enumerate(scenario.turns):
            if isinstance(turn, CustomerTurn):
                pending_customer = turn
                golden_history.append({"role": Role.USER, "content": turn.message})
                actual_history.append({"role": Role.USER, "content": turn.message})
                continue

            assert isinstance(turn, AgentTurn)
            assert pending_customer is not None, (
                "schema validation should have ensured customer-first alternation"
            )

            turn_result = self._execute_agent_turn(
                scenario=scenario,
                turn_index=i,
                agent_turn=turn,
                customer_turn=pending_customer,
                golden_history=golden_history,
                actual_history=actual_history,
            )
            turn_results.append(turn_result)
            if on_turn_complete is not None:
                on_turn_complete()

            # Golden injection for the text turn: the GOLDEN text response is
            # what future turns see.
            golden_history.append(
                {"role": Role.AGENT, "content": turn.golden_response}
            )
            pending_customer = None

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
        customer_turn: CustomerTurn,
        golden_history: list[Message],
        actual_history: list[Message],
    ) -> TurnResult:
        # --- Tool loops (may be empty) -----------------------------------
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
                    customer_turn=customer_turn,
                    agent_turn=agent_turn,
                    tool_records=tool_records,
                    abort=abort,
                )

        # --- Final text turn --------------------------------------------
        base = _TurnBase(
            turn_index=turn_index,
            customer_message=customer_turn.message,
            golden_response=agent_turn.golden_response,
            may_diverge=agent_turn.may_diverge,
            divergence_note=agent_turn.divergence_note,
            tool_calls=tool_records,
        )

        try:
            reply = self.agent.chat(list(golden_history))
        except AgentTimeoutError as exc:
            log.warning("agent timeout on scenario=%s turn=%d", scenario.id, turn_index)
            actual_history.append({"role": Role.AGENT, "content": ""})
            return base.to_error_result(TurnStatus.AGENT_TIMEOUT, str(exc))
        except AgentError as exc:
            log.warning("agent error on scenario=%s turn=%d: %s", scenario.id, turn_index, exc)
            actual_history.append({"role": Role.AGENT, "content": ""})
            return base.to_error_result(TurnStatus.AGENT_ERROR, str(exc))
        except AgentUnreachableError as exc:
            log.error("agent unreachable: %s", exc)
            actual_history.append({"role": Role.AGENT, "content": ""})
            tr = base.to_error_result(TurnStatus.AGENT_ERROR, str(exc))
            return TurnResult(**{**tr.model_dump(), "error": "agent_unreachable"})

        actual_text = reply.text or ""
        actual_history.append({"role": Role.AGENT, "content": actual_text})

        if not actual_text.strip():
            fail_checks = [
                CheckResult(check=c, passed=False, explanation="agent returned empty text response")
                for c in agent_turn.checks
            ]
            return TurnResult(
                turn_index=turn_index,
                customer_message=customer_turn.message,
                golden_response=agent_turn.golden_response,
                actual_response="",
                may_diverge=agent_turn.may_diverge,
                divergence_note=agent_turn.divergence_note,
                tool_calls=tool_records,
                status=TurnStatus.EMPTY_RESPONSE,
                consistency_score=1,
                divergence_type="missing_information",
                checks=fail_checks,
                diverged=True,
            )

        try:
            turn_eval: TurnEval = evaluate_turn(
                self.llm,
                golden_response=agent_turn.golden_response,
                actual_response=actual_text,
                checks=agent_turn.checks,
            )
        except EvaluationError as exc:
            log.warning("eval error on scenario=%s turn=%d: %s", scenario.id, turn_index, exc)
            fail_checks = [
                CheckResult(check=c, passed=False, explanation="eval_error")
                for c in agent_turn.checks
            ]
            return TurnResult(
                turn_index=turn_index,
                customer_message=customer_turn.message,
                golden_response=agent_turn.golden_response,
                actual_response=actual_text,
                may_diverge=agent_turn.may_diverge,
                divergence_note=agent_turn.divergence_note,
                tool_calls=tool_records,
                status=TurnStatus.EVAL_ERROR,
                checks=fail_checks,
                error=str(exc),
            )

        check_results = [
            CheckResult(check=c.check, passed=c.passed, explanation=c.explanation)
            for c in turn_eval.checks
        ]
        return TurnResult(
            turn_index=turn_index,
            customer_message=customer_turn.message,
            golden_response=agent_turn.golden_response,
            actual_response=actual_text,
            may_diverge=agent_turn.may_diverge,
            divergence_note=agent_turn.divergence_note,
            tool_calls=tool_records,
            status=TurnStatus.EVALUATED,
            consistency_score=turn_eval.consistency_score,
            divergence_type=turn_eval.divergence_type,
            checks=check_results,
            diverged=turn_eval.consistency_score < 5,
        )

    # ------------------------------------------------------------------
    # Tool loop execution
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
        """Drive one tool_loop to completion, appending to ``out_records``.

        Raises :class:`_AgentAbort` when the underlying agent call fails; the
        outer handler converts that into a turn-level error result.
        """
        matcher = LoopMatcher(loop)
        rounds = 0

        while not matcher.done and rounds < MAX_TOOL_ROUNDS_PER_LOOP:
            try:
                reply = self.agent.chat(list(golden_history))
            except AgentTimeoutError as exc:
                raise _AgentAbort(TurnStatus.AGENT_TIMEOUT, str(exc)) from exc
            except AgentError as exc:
                raise _AgentAbort(TurnStatus.AGENT_ERROR, str(exc)) from exc
            except AgentUnreachableError as exc:
                raise _AgentAbort(OutcomeStatus.AGENT_UNREACHABLE, str(exc)) from exc

            if not reply.is_tool_round:
                # Agent returned text while we were still expecting tool
                # calls. Record it into actual_history for reporting, then
                # break so remaining tools are marked ``missing`` below.
                text = reply.text or ""
                actual_history.append({"role": Role.AGENT, "content": text})
                break

            rounds += 1

            # Record the agent's actual tool_calls in the ACTUAL history
            # (preserves what really happened for the outcome eval).
            actual_agent_tool_msg: Message = {
                "role": Role.AGENT,
                "content": reply.text or "",
                "tool_calls": [_call_to_dict(c) for c in reply.tool_calls],
            }
            actual_history.append(actual_agent_tool_msg)

            decisions = matcher.process(reply.tool_calls)
            for decision in decisions:
                out_records.append(
                    _decision_to_record(decision, loop_index=loop_index)
                )

            # GOLDEN INJECTION: write the GOLDEN tool_call + tool_response
            # into the history for every decision that consumed an expected
            # position, regardless of whether the agent got it right.
            injected_calls: list[dict] = []
            injected_responses: list[Message] = []
            for decision in decisions:
                if decision.matched_tool is None:
                    continue  # unexpected_tool: nothing to inject
                call_id = _ensure_call_id(decision.agent_call.id)
                injected_calls.append(
                    _expected_call_to_dict(decision.matched_tool, call_id)
                )
                tool_response_msg: Message = {
                    "role": Role.TOOL,
                    "tool_call_id": call_id,
                    "name": decision.matched_tool.name,
                    "content": decision.matched_tool.expected_response,
                }
                injected_responses.append(tool_response_msg)

            if injected_calls:
                golden_agent_tool_msg: Message = {
                    "role": Role.AGENT,
                    "content": None,
                    "tool_calls": injected_calls,
                }
                golden_history.append(golden_agent_tool_msg)
                golden_history.extend(injected_responses)
                actual_history.extend(injected_responses)

        # Any pending positions left unmatched -> "missing".
        for position in matcher.finalize():
            expected = loop.tools[position]
            out_records.append(
                ToolCallRecord(
                    loop_index=loop_index,
                    position=position,
                    expected_name=expected.name,
                    expected_arguments_schema=expected.arguments_schema,
                    actual_name=None,
                    status=ToolCallStatus.MISSING,
                    explanation="agent stopped emitting tool_calls before this tool was invoked",
                )
            )
            # Still inject the golden call + response so the downstream text
            # turn sees a coherent history.
            call_id = _ensure_call_id(None)
            golden_missing_tool_msg: Message = {
                "role": Role.AGENT,
                "content": None,
                "tool_calls": [_expected_call_to_dict(expected, call_id)],
            }
            golden_history.append(golden_missing_tool_msg)
            response_msg: Message = {
                "role": Role.TOOL,
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
            return OutcomeResult(status=OutcomeStatus.EVALUATED, checks=[])

        try:
            outcome: OutcomeEval = evaluate_outcome(
                self.llm,
                conversation=[dict(m) for m in actual_history],
                expected_outcome=scenario.expected_outcome,
                outcome_checks=scenario.outcome_checks,
            )
        except EvaluationError as exc:
            log.warning("outcome eval error on scenario=%s: %s", scenario.id, exc)
            return OutcomeResult(
                status=OutcomeStatus.EVAL_ERROR,
                checks=[
                    CheckResult(check=c, passed=False, explanation="eval_error")
                    for c in scenario.outcome_checks
                ],
                error=str(exc),
            )

        return OutcomeResult(
            status=OutcomeStatus.EVALUATED,
            checks=[
                CheckResult(check=c.check, passed=c.passed, explanation=c.explanation)
                for c in outcome.checks
            ],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TurnBase:
    turn_index: int
    customer_message: str
    golden_response: str
    may_diverge: bool
    divergence_note: str | None
    tool_calls: list[ToolCallRecord]

    def to_error_result(self, status: TurnStatus, error: str) -> TurnResult:
        return TurnResult(
            turn_index=self.turn_index,
            customer_message=self.customer_message,
            golden_response=self.golden_response,
            actual_response="",
            may_diverge=self.may_diverge,
            divergence_note=self.divergence_note,
            tool_calls=self.tool_calls,
            status=status,
            checks=[],
            diverged=True,
            error=error,
        )


class _AgentAbort(Exception):
    """Raised from ``_run_tool_loop`` when the agent HTTP layer gave up."""

    def __init__(self, status: TurnStatus | OutcomeStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _tool_abort_turn_result(
    *,
    turn_index: int,
    customer_turn: CustomerTurn,
    agent_turn: AgentTurn,
    tool_records: list[ToolCallRecord],
    abort: _AgentAbort,
) -> TurnResult:
    is_unreachable = abort.status == OutcomeStatus.AGENT_UNREACHABLE
    base = _TurnBase(
        turn_index=turn_index,
        customer_message=customer_turn.message,
        golden_response=agent_turn.golden_response,
        may_diverge=agent_turn.may_diverge,
        divergence_note=agent_turn.divergence_note,
        tool_calls=tool_records,
    )
    status = (
        TurnStatus.AGENT_ERROR
        if is_unreachable
        else (abort.status if isinstance(abort.status, TurnStatus) else TurnStatus.AGENT_ERROR)
    )
    tr = base.to_error_result(status, abort.message)
    if is_unreachable:
        return TurnResult(**{**tr.model_dump(), "error": "agent_unreachable"})
    return tr


def _unreachable_run(
    run_index: int, turn_results: list[TurnResult], error: str
) -> ScenarioRunResult:
    return ScenarioRunResult(
        run_index=run_index,
        status=ScenarioStatus.AGENT_UNREACHABLE,
        turns=turn_results,
        outcome=OutcomeResult(status=OutcomeStatus.AGENT_UNREACHABLE, error=error),
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
    """Build a tool_call dict for golden injection using the expected tool.

    Arguments default to a minimal example that satisfies each schema key so
    the injected call looks well-formed to the agent on subsequent rounds.
    """
    return {
        "id": call_id,
        "name": expected.name,
        "arguments": _default_arguments(expected.arguments_schema),
    }


def _default_arguments(schema: dict[str, ToolArgumentSchema]) -> dict:
    """Fabricate a minimal valid arguments dict for golden injection."""
    defaults: dict[str, object] = {}
    for key, arg in schema.items():
        defaults[key] = _default_for(arg.type)
    return defaults


def _default_for(t: ArgumentType) -> object:
    return {
        ArgumentType.STRING: "",
        ArgumentType.INTEGER: 0,
        ArgumentType.NUMBER: 0.0,
        ArgumentType.BOOLEAN: False,
        ArgumentType.ARRAY: [],
        ArgumentType.OBJECT: {},
        ArgumentType.ANY: "",
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
