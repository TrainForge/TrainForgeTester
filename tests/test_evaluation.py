"""Evaluator tests - compact-output parsing, strict-retry, length validation."""
from __future__ import annotations

import json

import pytest

from trainforge.errors import EvaluationError
from trainforge.evaluation import (
    _VALID_DIVERGENCE_TYPES,
    evaluate_outcome,
    evaluate_turn,
)


def _turn_payload(score: int, divergence: str, checks: list[tuple[str, bool]]) -> str:
    return json.dumps(
        {
            "consistency_score": score,
            "divergence_type": divergence,
            "checks": [
                {"check": c[0], "pass": c[1], "explanation": ""} for c in checks
            ],
        }
    )


# ---------------------------------------------------------------------------
# evaluate_turn (compact format)
# ---------------------------------------------------------------------------


def _compact(r: list[int], f: dict[str, str] | None = None) -> str:
    return json.dumps({"r": r, "f": f or {}})


def test_valid_divergence_set_matches_spec() -> None:
    expected = {
        "none",
        "factual_difference",
        "style_difference",
        "missing_information",
        "extra_information",
        "wrong_action",
    }
    assert _VALID_DIVERGENCE_TYPES == expected


def test_turn_eval_happy_path(fake_llm) -> None:
    fake_llm.queue_raw(_compact([1, 1]))
    result = evaluate_turn(
        fake_llm,
        golden_response="g",
        actual_response="a",
        user_message="hi",
        questions=["q1", "q2"],
    )
    assert [c.passed for c in result.checks] == [True, True]
    assert [c.question for c in result.checks] == ["q1", "q2"]


def test_turn_eval_failures_with_explanations(fake_llm) -> None:
    fake_llm.queue_raw(_compact([1, 0, 1], {"2": "different language"}))
    result = evaluate_turn(
        fake_llm,
        golden_response="g",
        actual_response="a",
        user_message="hi",
        questions=["q1", "q2", "q3"],
    )
    assert [c.passed for c in result.checks] == [True, False, True]
    assert result.checks[1].explanation == "different language"
    # Non-failed entries don't get explanations.
    assert result.checks[0].explanation == ""
    assert result.checks[2].explanation == ""


def test_turn_eval_strips_code_fences(fake_llm) -> None:
    fake_llm.queue_raw("```json\n" + _compact([1, 0]) + "\n```")
    result = evaluate_turn(
        fake_llm,
        golden_response="g",
        actual_response="a",
        user_message="hi",
        questions=["q1", "q2"],
    )
    assert [c.passed for c in result.checks] == [True, False]


def test_turn_eval_accepts_true_false_too(fake_llm) -> None:
    fake_llm.queue_raw(_compact([True, False]))  # type: ignore[list-item]
    result = evaluate_turn(
        fake_llm,
        golden_response="g",
        actual_response="a",
        user_message="hi",
        questions=["q1", "q2"],
    )
    assert [c.passed for c in result.checks] == [True, False]


def test_turn_eval_rejects_wrong_length(fake_llm) -> None:
    fake_llm.queue_raw(_compact([1, 0]))
    fake_llm.queue_raw(_compact([1, 0]))  # strict retry returns same wrong shape
    with pytest.raises(EvaluationError):
        evaluate_turn(
            fake_llm,
            golden_response="g",
            actual_response="a",
            user_message="hi",
            questions=["q1", "q2", "q3"],  # 3 questions but r has length 2
        )


def test_turn_eval_rejects_non_binary_value(fake_llm) -> None:
    fake_llm.queue_raw(json.dumps({"r": [1, 0.5], "f": {}}))
    fake_llm.queue_raw(json.dumps({"r": [1, 0.5], "f": {}}))
    with pytest.raises(EvaluationError):
        evaluate_turn(
            fake_llm,
            golden_response="g",
            actual_response="a",
            user_message="hi",
            questions=["q1", "q2"],
        )


def test_turn_eval_retries_once_then_succeeds(fake_llm) -> None:
    fake_llm.queue_raw("not json at all")
    fake_llm.queue_raw(_compact([0, 1], {"1": "no"}))
    result = evaluate_turn(
        fake_llm,
        golden_response="g",
        actual_response="a",
        user_message="hi",
        questions=["q1", "q2"],
    )
    assert [c.passed for c in result.checks] == [False, True]
    assert result.checks[0].explanation == "no"
    assert len(fake_llm.calls) == 2
    # The strict-retry system prompt has a suffix appended.
    first_system, _ = fake_llm.calls[0]
    second_system, _ = fake_llm.calls[1]
    assert second_system != first_system


def test_turn_eval_double_failure_raises(fake_llm) -> None:
    fake_llm.queue_raw("still garbage")
    fake_llm.queue_raw("also garbage")
    with pytest.raises(EvaluationError):
        evaluate_turn(
            fake_llm,
            golden_response="g",
            actual_response="a",
            user_message="hi",
            questions=["q1"],
        )


def test_turn_eval_empty_questions_raises() -> None:
    """Building a turn-eval prompt with no questions is a programmer error."""
    from trainforge.llm.prompts import build_turn_eval_prompt

    with pytest.raises(ValueError):
        build_turn_eval_prompt(
            golden_response="g", actual_response="a", user_message="hi", questions=[]
        )


# ---------------------------------------------------------------------------
# evaluate_outcome (also compact format)
# ---------------------------------------------------------------------------


def test_outcome_eval_happy_path(fake_llm) -> None:
    fake_llm.queue_raw(_compact([1, 0], {"2": "not found"}))
    result = evaluate_outcome(
        fake_llm,
        conversation=[{"role": "user", "content": "hi"}],
        expected_outcome="booking confirmed",
        outcome_checks=["o1", "o2"],
    )
    assert [c.passed for c in result.checks] == [True, False]
    assert result.checks[1].explanation == "not found"
