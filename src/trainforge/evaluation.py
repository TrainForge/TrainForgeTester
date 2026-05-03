"""LLM evaluation: turn checks (binary, batched) + outcome checks.

The runner only invokes evaluators when it cannot answer deterministically:

- Per-turn evaluation runs only when ``AgentTurn.may_diverge=True``.
  Otherwise the runner does Python ``==`` text equality and skips this
  module entirely. That is the TrainForge 0.1 deterministic-first contract.
- Outcome evaluation runs once per scenario over the full ACTUAL transcript.

Both evaluators speak the compact ``{"r": [...], "f": {...}}`` wire format
defined in :mod:`trainforge.llm.prompts`. This module owns:

- the prompt orchestration,
- one strict-retry on parse failure (matching the spec's error-handling
  table),
- positional decoding back into typed dataclasses the runner consumes.
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
    """One binary verdict from the LLM."""

    question: str
    passed: bool
    explanation: str = ""


@dataclass(frozen=True)
class TurnEval:
    """All binary verdicts for a single turn-eval call.

    The runner is responsible for splitting ``checks`` back into
    standard-check results and per-scenario custom-check results based on
    the order the runner originally composed the question list.
    """

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
    user_message: str,
    questions: list[str],
) -> TurnEval:
    """Batched per-turn evaluation. Raises :class:`EvaluationError` on failure."""
    system, user = build_turn_eval_prompt(
        golden_response=golden_response,
        actual_response=actual_response,
        user_message=user_message,
        questions=questions,
    )
    raw = _call_with_strict_retry(llm, system, user)
    return TurnEval(checks=_decode_compact(raw, questions))


def evaluate_outcome(
    llm: LLMClient,
    *,
    conversation: list[dict],
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
    return OutcomeEval(checks=_decode_compact(raw, outcome_checks))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _call_with_strict_retry(llm: LLMClient, system: str, user: str) -> dict:
    """Call ``llm.complete`` and parse a JSON object out of the response.

    On first parse failure, retry once with a stricter system-prompt suffix.
    On second failure, raise :class:`EvaluationError`.
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


def _decode_compact(obj: dict, questions: list[str]) -> list[CheckEval]:
    """Decode a ``{"r": [...], "f": {...}}`` payload into typed CheckEvals.

    Strict on length: ``r`` must contain exactly ``len(questions)`` entries.
    Permissive on the failure-explanations object: missing keys for failed
    indices get a placeholder explanation; extra keys are ignored.
    """
    raw_r = obj.get("r")
    if not isinstance(raw_r, list):
        raise EvaluationError(
            f"compact payload missing 'r' list: keys={sorted(obj.keys())!r}"
        )
    if len(raw_r) != len(questions):
        raise EvaluationError(
            f"compact payload 'r' has length {len(raw_r)}, expected {len(questions)}"
        )

    raw_f = obj.get("f", {}) or {}
    if not isinstance(raw_f, dict):
        raise EvaluationError(
            f"compact payload 'f' must be an object, got {type(raw_f).__name__}"
        )

    out: list[CheckEval] = []
    for i, value in enumerate(raw_r):
        passed = _coerce_binary(value)
        if passed is None:
            raise EvaluationError(
                f"compact payload r[{i}] is not 0/1/true/false: {value!r}"
            )
        explanation = ""
        if not passed:
            # Look up the explanation by 1-based index (string key).
            raw_expl = raw_f.get(str(i + 1)) or raw_f.get(i + 1)
            if isinstance(raw_expl, str):
                explanation = raw_expl
        out.append(
            CheckEval(question=questions[i], passed=passed, explanation=explanation)
        )
    return out


def _coerce_binary(v: object) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if v == 1:
            return True
        if v == 0:
            return False
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "pass", "y"}:
            return True
        if s in {"0", "false", "no", "fail", "n"}:
            return False
    return None
