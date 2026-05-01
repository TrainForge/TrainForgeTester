"""Structural validation + matching for agent tool_calls.

Pure functions only - no HTTP, no LLM. Given an :class:`ExpectedTool` and a
dict of ``arguments`` the agent passed, decide whether the call is
schema-valid. Given a :class:`ToolLoop` and the agent's ordered list of
tool_calls this round, match each call to an expected position.

The runner owns the golden-injection side effects (writing into history);
this module's job is to classify.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trainforge.schema import (
    ArgumentType,
    ExpectedTool,
    ToolArgumentSchema,
    ToolCallStatus,
    ToolLoop,
)


@dataclass(frozen=True)
class ArgsValidation:
    """Result of :func:`validate_arguments` - fully deterministic."""

    missing_keys: list[str] = field(default_factory=list)
    wrong_type_keys: list[str] = field(default_factory=list)
    wrong_value_keys: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing_keys or self.wrong_type_keys or self.wrong_value_keys)

    @property
    def explanation(self) -> str:
        parts: list[str] = []
        if self.missing_keys:
            parts.append(f"missing arguments: {', '.join(self.missing_keys)}")
        if self.wrong_type_keys:
            parts.append(f"wrong-typed arguments: {', '.join(self.wrong_type_keys)}")
        if self.wrong_value_keys:
            parts.append(
                f"arguments with wrong value vs expected: {', '.join(self.wrong_value_keys)}"
            )
        return "; ".join(parts)


def validate_arguments(
    expected: ExpectedTool, arguments: dict
) -> ArgsValidation:
    """Check ``arguments`` against the expected tool's arguments_schema.

    Deterministic checks performed here:

    - Every declared key must be present.
    - Each value must match the declared ``type`` (except ``type="any"``).
    - If the schema declared an ``expected`` literal, the agent's value must
      equal it (``==``).

    Extra keys in ``arguments`` are permitted.
    """
    missing: list[str] = []
    wrong_type: list[str] = []
    wrong_value: list[str] = []

    for key, schema in expected.arguments_schema.items():
        if key not in arguments:
            missing.append(key)
            continue
        value = arguments[key]
        if not _value_matches_type(value, schema):
            wrong_type.append(key)
            continue
        if schema.expected is not None and value != schema.expected:
            wrong_value.append(key)

    return ArgsValidation(
        missing_keys=missing,
        wrong_type_keys=wrong_type,
        wrong_value_keys=wrong_value,
    )


def _value_matches_type(value: object, schema: ToolArgumentSchema) -> bool:
    t = schema.type
    if t == ArgumentType.ANY:
        return True
    if t == ArgumentType.STRING:
        return isinstance(value, str)
    if t == ArgumentType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if t == ArgumentType.NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == ArgumentType.BOOLEAN:
        return isinstance(value, bool)
    if t == ArgumentType.ARRAY:
        return isinstance(value, list)
    if t == ArgumentType.OBJECT:
        return isinstance(value, dict)
    return False  # pragma: no cover - enum exhausts options


# ---------------------------------------------------------------------------
# Loop matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentToolCall:
    """What the agent emitted. ``id`` is optional (some SDKs elide it)."""

    name: str
    arguments: dict
    id: str | None = None


@dataclass(frozen=True)
class MatchDecision:
    """One decision in the matching pipeline.

    ``expected_position`` is which slot in the loop this call consumed, if
    any. ``matched_tool`` is the ExpectedTool it was credited against, if
    any. ``status`` follows :data:`trainforge.schema.ToolCallStatus`.
    """

    agent_call: AgentToolCall
    expected_position: int | None
    matched_tool: ExpectedTool | None
    status: ToolCallStatus
    explanation: str = ""


@dataclass
class LoopMatcher:
    """Stateful matcher for a single :class:`ToolLoop` across multiple agent
    HTTP rounds.

    Usage::

        matcher = LoopMatcher(loop)
        for round_tool_calls in agent_rounds:
            decisions = matcher.process(round_tool_calls)
            # ... inject golden into history based on decisions ...
        remaining = matcher.finalize()
        # remaining are the expected tools the agent never called.
    """

    loop: ToolLoop
    pending_positions: list[int] = field(init=False)
    """Positions in ``loop.tools`` that are still un-matched."""

    def __post_init__(self) -> None:
        self.pending_positions = list(range(len(self.loop.tools)))

    @property
    def done(self) -> bool:
        return not self.pending_positions

    # ------------------------------------------------------------------
    # Per-round processing
    # ------------------------------------------------------------------

    def process(
        self, agent_calls: list[AgentToolCall]
    ) -> list[MatchDecision]:
        """Match one round of agent tool_calls against the loop.

        Returns one :class:`MatchDecision` per agent call, in the order the
        agent emitted them. Side-effect: consumes positions from
        ``pending_positions`` for ``pass`` / ``invalid_arguments`` /
        ``wrong_tool``; does NOT consume a position for ``unexpected_tool``.
        """
        decisions: list[MatchDecision] = []
        for ac in agent_calls:
            if self.loop.ordered:
                decisions.append(self._ordered_step(ac))
            else:
                decisions.append(self._unordered_step(ac))
        return decisions

    def finalize(self) -> list[int]:
        """Return the list of expected positions that were never matched.

        The runner turns these into ``missing`` :class:`ToolCallRecord`s.
        """
        leftover = self.pending_positions
        self.pending_positions = []
        return leftover

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ordered_step(self, ac: AgentToolCall) -> MatchDecision:
        if not self.pending_positions:
            return MatchDecision(
                agent_call=ac,
                expected_position=None,
                matched_tool=None,
                status=ToolCallStatus.UNEXPECTED_TOOL,
                explanation="agent emitted tool_call after all expected tools were already matched",
            )
        position = self.pending_positions[0]
        expected = self.loop.tools[position]
        if ac.name != expected.name:
            # Consume this slot regardless -- the runner will inject the
            # GOLDEN call for this position so the agent's history stays
            # on-track. Report wrong_tool on the agent's attempt.
            self.pending_positions.pop(0)
            return MatchDecision(
                agent_call=ac,
                expected_position=position,
                matched_tool=expected,
                status=ToolCallStatus.WRONG_TOOL,
                explanation=f"ordered loop expected {expected.name!r} at position {position}, agent called {ac.name!r}",
            )
        validation = validate_arguments(expected, ac.arguments)
        self.pending_positions.pop(0)
        if not validation.ok:
            return MatchDecision(
                agent_call=ac,
                expected_position=position,
                matched_tool=expected,
                status=ToolCallStatus.INVALID_ARGUMENTS,
                explanation=validation.explanation,
            )
        return MatchDecision(
            agent_call=ac,
            expected_position=position,
            matched_tool=expected,
            status=ToolCallStatus.PASS,
        )

    def _unordered_step(self, ac: AgentToolCall) -> MatchDecision:
        # Prefer the first pending position whose tool's name matches.
        for i, position in enumerate(self.pending_positions):
            candidate = self.loop.tools[position]
            if candidate.name == ac.name:
                validation = validate_arguments(candidate, ac.arguments)
                self.pending_positions.pop(i)
                if not validation.ok:
                    return MatchDecision(
                        agent_call=ac,
                        expected_position=position,
                        matched_tool=candidate,
                        status=ToolCallStatus.INVALID_ARGUMENTS,
                        explanation=validation.explanation,
                    )
                return MatchDecision(
                    agent_call=ac,
                    expected_position=position,
                    matched_tool=candidate,
                    status=ToolCallStatus.PASS,
                )
        # No pending tool has this name. If the loop is fully consumed, it's
        # unexpected; otherwise it's wrong_tool -- but we still consume the
        # first remaining slot so the runner can advance with golden
        # injection.
        if not self.pending_positions:
            return MatchDecision(
                agent_call=ac,
                expected_position=None,
                matched_tool=None,
                status=ToolCallStatus.UNEXPECTED_TOOL,
                explanation="agent emitted tool_call after all expected tools were already matched",
            )
        pending_names = [self.loop.tools[p].name for p in self.pending_positions]
        position = self.pending_positions.pop(0)
        expected = self.loop.tools[position]
        return MatchDecision(
            agent_call=ac,
            expected_position=position,
            matched_tool=expected,
            status=ToolCallStatus.WRONG_TOOL,
            explanation=(
                f"unordered loop expected one of {pending_names!r}, "
                f"agent called {ac.name!r}"
            ),
        )
