"""Evaluator tests - parsing, strict-retry, protocol shape."""
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
    fake_llm.queue_raw(_turn_payload(5, "none", [("c1", True), ("c2", True)]))
    result = evaluate_turn(
        fake_llm,
        golden_response="golden",
        actual_response="actual",
        checks=["c1", "c2"],
    )
    assert result.consistency_score == 5
    assert result.divergence_type == "none"
    assert [c.passed for c in result.checks] == [True, True]


def test_turn_eval_strips_code_fences(fake_llm) -> None:
    fake_llm.queue_raw("```json\n" + _turn_payload(4, "style_difference", [("c", True)]) + "\n```")
    result = evaluate_turn(
        fake_llm,
        golden_response="g",
        actual_response="a",
        checks=["c"],
    )
    assert result.consistency_score == 4


def test_turn_eval_retries_once_then_succeeds(fake_llm) -> None:
    fake_llm.queue_raw("not json at all")
    fake_llm.queue_raw(_turn_payload(3, "factual_difference", [("c", False)]))
    result = evaluate_turn(
        fake_llm, golden_response="g", actual_response="a", checks=["c"]
    )
    assert result.consistency_score == 3
    assert result.checks[0].passed is False
    assert len(fake_llm.calls) == 2
    first_system, _ = fake_llm.calls[0]
    second_system, _ = fake_llm.calls[1]
    assert first_system != second_system  # strict retry suffix appended


def test_turn_eval_double_failure_raises(fake_llm) -> None:
    fake_llm.queue_raw("still garbage")
    fake_llm.queue_raw("also garbage")
    with pytest.raises(EvaluationError):
        evaluate_turn(
            fake_llm, golden_response="g", actual_response="a", checks=["c"]
        )


def test_turn_eval_rejects_invalid_score(fake_llm) -> None:
    fake_llm.queue_raw(
        json.dumps(
            {"consistency_score": 9, "divergence_type": "none", "checks": []}
        )
    )
    with pytest.raises(EvaluationError):
        evaluate_turn(
            fake_llm, golden_response="g", actual_response="a", checks=[]
        )


def test_turn_eval_fills_missing_checks_as_failures(fake_llm) -> None:
    fake_llm.queue_raw(_turn_payload(5, "none", [("c1", True)]))
    result = evaluate_turn(
        fake_llm, golden_response="g", actual_response="a", checks=["c1", "c2"]
    )
    assert result.checks[0].passed is True
    assert result.checks[1].passed is False
    assert "missing" in result.checks[1].explanation.lower()


def test_outcome_eval_happy_path(fake_llm) -> None:
    fake_llm.queue_outcome([("o1", True, ""), ("o2", False, "not found")])
    result = evaluate_outcome(
        fake_llm,
        conversation=[{"role": "customer", "content": "hi"}],
        expected_outcome="booking confirmed",
        outcome_checks=["o1", "o2"],
    )
    assert [c.passed for c in result.checks] == [True, False]
    assert result.checks[1].explanation == "not found"


