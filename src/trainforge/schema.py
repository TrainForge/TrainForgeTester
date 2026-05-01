"""Pydantic models for TrainForge's two wire formats.

- ``ScenariosFile``: input to ``trainforge run``. Mirrors the schema in
  testing-spec-v1.md section "Test Scenario Format (JSON)" plus the v1.1
  tool-call extension (see :class:`ToolLoop` / :class:`ExpectedTool`).
- ``RunResults``: output of ``trainforge run``, input to ``trainforge report``
  and ``trainforge diff``.

Both carry a ``version`` field. Any mismatch raises
:class:`trainforge.errors.UnsupportedScenarioVersionError` so format drift is
explicit (spec open-question #3).

Message types in the conversation history (the union is enforced by
:mod:`trainforge.agent_client`, not pydantic, because the history is wire
shape rather than stored state):

- ``{"role": "user",  "content": str}``
- ``{"role": "agent", "content": str}``
- ``{"role": "agent", "content": str | None, "tool_calls": [ToolCall, ...]}``
- ``{"role": "tool",  "tool_call_id": str, "name": str, "content": str}``
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from trainforge import RESULTS_FORMAT_VERSION, SCENARIO_FORMAT_VERSION
from trainforge.errors import MalformedScenarioError, UnsupportedScenarioVersionError


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Tool-call extension
# ---------------------------------------------------------------------------


try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11
    class StrEnum(str, Enum):
        pass


class ArgumentType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


class ToolArgumentSchema(_StrictModel):
    """Declaration for one argument of an expected tool.

    Every declared key is implicitly required; extra keys in the agent's
    arguments are allowed (permissive), matching real LLM tool-call
    behavior. Validation is fully deterministic:

    - ``type``     : structural type check (free, always runs).
    - ``expected`` : optional literal equality check. When set, the agent's
      value must equal it exactly (``==``). Tool arguments are structured
      API inputs; matching is deterministic by design. If you want to
      accept several equivalent phrasings, pick one canonical value and
      rewrite your agent / test accordingly.
    """

    type: ArgumentType = ArgumentType.ANY
    description: str = ""
    expected: Any = None
    """Optional literal. If non-None, the agent's value must ``==`` it."""


class ExpectedTool(_StrictModel):
    """One tool call the scenario expects the agent to make inside a loop.

    The same tool invoked twice with different arguments is declared as two
    separate ``ExpectedTool`` entries; there is no implicit "call multiple
    times" semantic.
    """

    name: str
    arguments_schema: dict[str, ToolArgumentSchema] = Field(default_factory=dict)
    expected_response: str
    """Fake tool output the runner injects into history when the agent calls
    this tool (or when the runner substitutes it after a wrong_tool failure)."""


class ToolLoop(_StrictModel):
    """A group of tool calls that must happen before the next agent text turn.

    ``ordered=True`` requires calls to arrive in the declared order (across
    any number of agent HTTP rounds). ``ordered=False`` (default) lets the
    agent call them in any order.
    """

    ordered: bool = False
    tools: list[ExpectedTool]

    @field_validator("tools")
    @classmethod
    def _non_empty(cls, v: list[ExpectedTool]) -> list[ExpectedTool]:
        if not v:
            raise ValueError("tool_loop.tools must contain at least one tool")
        return v


# ---------------------------------------------------------------------------
# Scenarios (input)
# ---------------------------------------------------------------------------


class UserRole(StrEnum):
    USER = "user"


class AgentRole(StrEnum):
    AGENT = "agent"


class TurnRole(StrEnum):
    USER = "user"
    AGENT = "agent"


class UserTurn(_StrictModel):
    role: UserRole
    message: str
    intent: str = ""

    @field_validator("role", mode="before")
    @classmethod
    def _accept_legacy_customer(cls, v: object) -> object:
        if v == "customer":
            return UserRole.USER
        return v


class AgentTurn(_StrictModel):
    role: AgentRole
    tool_loops: list[ToolLoop] = Field(default_factory=list)
    """Zero or more tool loops that must complete before the text response."""
    golden_response: str
    checks: list[str] = Field(default_factory=list)
    """Per-scenario natural-language checks (LLM-evaluated as binary). Run
    in addition to the :mod:`trainforge.standard_checks` battery whenever
    ``may_diverge=True``. Run regardless when ``may_diverge=False`` (along
    with the deterministic exact-match check)."""
    may_diverge: bool = False
    """Default ``False``: the actual agent reply must equal ``golden_response``
    exactly (Python ``==``). Use this for curated/scripted replies (legal
    disclaimers, fixed FAQ answers). Set ``True`` to allow the agent to
    rephrase; the runner then evaluates the 20 standard NLP-consistency
    checks via LLM instead of doing string equality."""
    divergence_note: str | None = None


Turn = UserTurn | AgentTurn


class Scenario(_StrictModel):
    id: str
    name: str
    description: str = ""
    source_transcript_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    turns: list[Turn]
    expected_outcome: str
    outcome_checks: list[str] = Field(default_factory=list)

    @field_validator("turns")
    @classmethod
    def _turns_alternate_starting_with_user(
        cls, turns: list[Turn]
    ) -> list[Turn]:
        if not turns:
            raise ValueError("scenario must have at least one turn")
        expected: tuple[TurnRole, ...] = (TurnRole.USER, TurnRole.AGENT)
        for i, t in enumerate(turns):
            want = expected[i % 2]
            if t.role != want:
                raise ValueError(
                    f"turn {i} has role={t.role!r}, expected {want!r}"
                    " (turns must alternate user, agent, ... starting with user)"
                )
        return turns


class ScenariosFile(_StrictModel):
    version: str
    scenarios: list[Scenario]

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if v != SCENARIO_FORMAT_VERSION:
            raise ValueError(
                f"unsupported scenarios version {v!r}; runner supports {SCENARIO_FORMAT_VERSION!r}"
            )
        return v


# ---------------------------------------------------------------------------
# Results (output)
# ---------------------------------------------------------------------------


class CheckResult(_StrictModel):
    check: str
    passed: bool
    explanation: str = ""


class ToolCallStatus(StrEnum):
    PASS = "pass"
    WRONG_TOOL = "wrong_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNEXPECTED_TOOL = "unexpected_tool"
    MISSING = "missing"


class ToolCallRecord(_StrictModel):
    """One evaluated tool-call attempt.

    - ``pass``              : agent called the expected tool with valid args.
    - ``wrong_tool``        : agent called a known-but-different tool (ordered
      loop, position mismatch, or unordered loop with a tool not in this loop
      at all -- the runner can't always tell which).
    - ``invalid_arguments`` : name matched but arguments didn't satisfy
      ``arguments_schema`` (missing key, wrong type, or wrong ``expected``
      value).
    - ``unexpected_tool``   : agent emitted an extra tool_call beyond the
      loop's declared size.
    - ``missing``            : loop ended with this expected tool never called.
    """

    loop_index: int
    position: int
    """0-based position within the loop of the expected tool this record
    corresponds to. For ``unexpected_tool`` records position is len(loop.tools)
    plus offset."""
    expected_name: str | None
    expected_arguments_schema: dict[str, ToolArgumentSchema] = Field(default_factory=dict)
    actual_name: str | None = None
    actual_arguments: dict = Field(default_factory=dict)
    status: ToolCallStatus
    explanation: str = ""


class TurnStatus(StrEnum):
    EVALUATED = "evaluated"
    AGENT_ERROR = "agent_error"
    AGENT_TIMEOUT = "agent_timeout"
    EVAL_ERROR = "eval_error"
    EMPTY_RESPONSE = "empty_response"


class StandardCheckResult(_StrictModel):
    """One binary standard NLP-consistency-check result.

    Carries the stable ``id`` from :mod:`trainforge.standard_checks` plus
    the LLM's binary verdict and (for failures) a short explanation."""

    id: str
    """Stable identifier from ``STANDARD_CHECKS``."""
    question: str
    """Verbatim text of the check (denormalized so reports are self-contained)."""
    passed: bool
    explanation: str = ""


class TurnResult(_StrictModel):
    turn_index: int
    """Index of the agent turn within ``scenario.turns``."""
    user_message: str
    golden_response: str
    actual_response: str
    may_diverge: bool
    divergence_note: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    status: TurnStatus = TurnStatus.EVALUATED

    # Text-equivalence verdict.
    exact_match: bool | None = None
    """When ``may_diverge=False``: ``True`` if ``actual_response == golden_response``,
    else ``False``. ``None`` when ``may_diverge=True`` (no exact-match check is run).
    """
    standard_check_results: list[StandardCheckResult] = Field(default_factory=list)
    """The 20 standard NLP-consistency check verdicts. Populated only when
    ``may_diverge=True``; empty otherwise."""
    checks: list[CheckResult] = Field(default_factory=list)
    """Per-scenario custom check verdicts (always run when there are any)."""
    error: str | None = None


class OutcomeStatus(StrEnum):
    EVALUATED = "evaluated"
    EVAL_ERROR = "eval_error"
    AGENT_UNREACHABLE = "agent_unreachable"


class OutcomeResult(_StrictModel):
    status: OutcomeStatus = OutcomeStatus.EVALUATED
    checks: list[CheckResult] = Field(default_factory=list)
    error: str | None = None


class ScenarioStatus(StrEnum):
    PASS = "pass"
    PARTIAL_PASS = "partial_pass"
    FAIL = "fail"
    AGENT_UNREACHABLE = "agent_unreachable"


class ScenarioRunResult(_StrictModel):
    run_index: int
    status: ScenarioStatus
    turns: list[TurnResult] = Field(default_factory=list)
    outcome: OutcomeResult = Field(default_factory=OutcomeResult)
    error: str | None = None


class ScenarioResult(_StrictModel):
    scenario_id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    runs: list[ScenarioRunResult]
    consistency: float
    """Pass rate across runs, 0.0 - 1.0."""
    inconsistent: bool
    """True if ``consistency < 0.80`` and runs > 1."""


class RunConfig(_StrictModel):
    agent_url: str
    llm_model: str
    runs: int
    timeout_seconds: float


class RunSummary(_StrictModel):
    total_scenarios: int
    passed: int
    partial: int
    failed: int
    unreachable: int
    inconsistent: int
    pass_rate: float
    overall_consistency: float
    tool_call_failures: int = 0
    """Total tool_calls across all scenarios/runs with status != pass."""
    exact_match_failures: int = 0
    """Turns with ``may_diverge=False`` where ``actual_response != golden_response``."""
    standard_check_failures: int = 0
    """Total standard NLP-consistency-check verdicts that returned 0 across
    all may_diverge=True turns."""
    custom_check_failures: int = 0
    """Total per-scenario custom check verdicts that returned 0."""


class RunResults(_StrictModel):
    version: str = RESULTS_FORMAT_VERSION
    config: RunConfig
    summary: RunSummary
    scenarios: list[ScenarioResult]

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if v != RESULTS_FORMAT_VERSION:
            raise ValueError(
                f"unsupported results version {v!r}; runner supports {RESULTS_FORMAT_VERSION!r}"
            )
        return v


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_scenarios(path: str | Path) -> ScenariosFile:
    """Load and validate a scenarios JSON file.

    Raises :class:`MalformedScenarioError` for schema failures and
    :class:`UnsupportedScenarioVersionError` for version mismatches.
    """
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MalformedScenarioError(f"scenarios file not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise MalformedScenarioError(f"scenarios file is not valid JSON: {exc}") from exc

    return parse_scenarios(raw)


def parse_scenarios(raw: dict) -> ScenariosFile:
    """Validate an in-memory dict against the scenarios schema."""
    if isinstance(raw, dict) and raw.get("version") not in (None, SCENARIO_FORMAT_VERSION):
        raise UnsupportedScenarioVersionError(
            f"unsupported scenarios version {raw.get('version')!r};"
            f" runner supports {SCENARIO_FORMAT_VERSION!r}"
        )
    try:
        return ScenariosFile.model_validate(raw)
    except ValidationError as exc:
        raise MalformedScenarioError(str(exc)) from exc


def load_results(path: str | Path) -> RunResults:
    """Load and validate a results JSON file."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    try:
        return RunResults.model_validate(raw)
    except ValidationError as exc:
        raise MalformedScenarioError(f"invalid results file: {exc}") from exc


def dump_results(results: RunResults, path: str | Path) -> None:
    """Write results to disk as pretty-printed JSON."""
    Path(path).write_text(
        json.dumps(results.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
