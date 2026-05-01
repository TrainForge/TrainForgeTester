"""Per-turn and per-scenario evaluation wrappers.

Wraps an :class:`LLMClient` with:
- prompt construction (via :mod:`trainforge.llm.prompts`),
- JSON parsing,
- one strict-retry on parse failure (testing-spec-v1.md "Error Handling" row
  "LLM returns unparseable JSON"),
- coercion into typed dataclasses the runner consumes.

Outputs are intentionally typed dataclasses, not pydantic models, to keep
this layer dependency-free of the wire-format types in
:mod:`trainforge.schema`; the runner is the single place where the two meet.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from trainforge.errors import EvaluationError
from trainforge.llm.base import LLMClient
from trainforge.llm.prompts import (
    STRICT_RETRY_SUFFIX,
    build_outcome_eval_prompt,
    build_turn_eval_prompt,
)


try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11
    class StrEnum(str, Enum):
        pass


class DivergenceType(StrEnum):
    NONE = "none"
    FACTUAL_DIFFERENCE = "factual_difference"
    STYLE_DIFFERENCE = "style_difference"
    MISSING_INFORMATION = "missing_information"
    EXTRA_INFORMATION = "extra_information"
    WRONG_ACTION = "wrong_action"


_VALID_DIVERGENCE_TYPES = {member.value for member in DivergenceType}


@dataclass(frozen=True)
class CheckEval:
    check: str
    passed: bool
    explanation: str = ""


@dataclass(frozen=True)
class TurnEval:
    consistency_score: int  # 1-5
    divergence_type: DivergenceType
    checks: list[CheckEval] = field(default_factory=list)


@dataclass(frozen=True)
class OutcomeEval:
    checks: list[CheckEval] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def evaluate_turn(
    llm: LLMClient,
    *,
    golden_response: str,
    actual_response: str,
    checks: list[str],
) -> TurnEval:
    """Batched per-turn evaluation. Raises :class:`EvaluationError` on failure."""
    system, user = build_turn_eval_prompt(
        golden_response=golden_response,
        actual_response=actual_response,
        checks=checks,
    )
    raw = _call_with_strict_retry(llm, system, user)
    return _parse_turn_eval(raw, checks)


def evaluate_outcome(
    llm: LLMClient,
    *,
    conversation: list[dict[str, str]],
    expected_outcome: str,
    outcome_checks: list[str],
) -> OutcomeEval:
    """Outcome evaluation over the full actual transcript."""
    system, user = build_outcome_eval_prompt(
        conversation=conversation,
        expected_outcome=expected_outcome,
        outcome_checks=outcome_checks,
    )
    raw = _call_with_strict_retry(llm, system, user)
    return _parse_outcome_eval(raw, outcome_checks)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _call_with_strict_retry(llm: LLMClient, system: str, user: str) -> dict:
    """Call ``llm.complete`` and parse a JSON object out of the response.

    If the first response fails to parse, retry once with a stricter system
    prompt suffix. If both fail, raise :class:`EvaluationError`.
    """
    try:
        return _extract_json(llm.complete(system, user))
    except _ParseFail as first_err:
        try:
            return _extract_json(llm.complete(system + STRICT_RETRY_SUFFIX, user))
        except _ParseFail as second_err:
            raise EvaluationError(
                f"LLM returned unparseable JSON twice: {first_err.msg!s} | {second_err.msg!s}"
            ) from second_err


class _ParseFail(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(raw: str) -> dict:
    """Permissive JSON extractor: strips code fences, falls back to the first
    balanced object. Returns a dict or raises :class:`_ParseFail`.
    """
    if not raw or not raw.strip():
        raise _ParseFail("empty response")

    candidates: list[str] = []
    fence = _FENCE_RE.search(raw)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(raw.strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for text in candidates:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise _ParseFail(f"no JSON object found in: {raw[:200]!r}")


def _parse_turn_eval(obj: dict, requested_checks: list[str]) -> TurnEval:
    raw_score = obj.get("consistency_score")
    if not isinstance(raw_score, int) or not 1 <= raw_score <= 5:
        raise EvaluationError(f"invalid consistency_score: {raw_score!r}")

    div = obj.get("divergence_type", DivergenceType.NONE.value)
    if not isinstance(div, str):
        raise EvaluationError(f"invalid divergence_type: {div!r}")
    if div not in _VALID_DIVERGENCE_TYPES:
        div = (
            DivergenceType.NONE.value
            if raw_score == 5
            else DivergenceType.FACTUAL_DIFFERENCE.value
        )

    raw_checks = obj.get("checks", [])
    if not isinstance(raw_checks, list):
        raise EvaluationError(f"'checks' is not a list: {raw_checks!r}")

    parsed_checks = _parse_checks(raw_checks, requested_checks)
    return TurnEval(
        consistency_score=raw_score,
        divergence_type=DivergenceType(div),
        checks=parsed_checks,
    )


def _parse_outcome_eval(obj: dict, requested_checks: list[str]) -> OutcomeEval:
    raw_checks = obj.get("checks", [])
    if not isinstance(raw_checks, list):
        raise EvaluationError(f"'checks' is not a list: {raw_checks!r}")
    return OutcomeEval(checks=_parse_checks(raw_checks, requested_checks))


def _parse_checks(raw_checks: list, requested: list[str]) -> list[CheckEval]:
    """Align LLM-returned checks with the requested check strings by position.

    LLMs occasionally reword the check text. We trust ``requested`` as the
    canonical label (spec: checks are natural-language strings from the
    scenario), and map results by index when possible.
    """
    by_index: list[CheckEval] = []
    for i, want in enumerate(requested):
        entry = raw_checks[i] if i < len(raw_checks) else None
        if not isinstance(entry, dict):
            by_index.append(CheckEval(check=want, passed=False, explanation="missing from LLM output"))
            continue
        passed = bool(entry.get("pass", entry.get("passed", False)))
        explanation = str(entry.get("explanation", "") or "")
        by_index.append(CheckEval(check=want, passed=passed, explanation=explanation))
    return by_index
