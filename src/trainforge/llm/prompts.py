"""Evaluator prompts.

Two prompt builders:

- :func:`build_turn_eval_prompt` - batched per-turn evaluation: consistency
  score + divergence_type + pass/fail per check in a single call
  (spec section "Batching").
- :func:`build_outcome_eval_prompt` - outcome checks over the full ACTUAL
  conversation (spec section "LLM Usage in the Runner").
"""
from __future__ import annotations

import json


TURN_EVAL_SYSTEM = (
    "You are evaluating an AI agent's response inside a conversational test "
    "harness. Your job is strict, boring NLP classification - not taste.\n\n"
    "You are given:\n"
    "  - a GOLDEN response (the expected answer from a reference transcript)\n"
    "  - an ACTUAL response (what the agent under test produced)\n"
    "  - a list of CHECKS (natural-language criteria the response should satisfy)\n\n"
    "You must:\n"
    "1. Rate how semantically consistent ACTUAL is with GOLDEN on a 1-5 scale:\n"
    "   5 = Semantically identical (same meaning, wording may differ)\n"
    "   4 = Mostly consistent (same core information, minor differences)\n"
    "   3 = Partially consistent (some information matches, some differs)\n"
    "   2 = Mostly different (different information or approach)\n"
    "   1 = Completely different (contradicts or ignores GOLDEN)\n"
    "2. If score < 5, classify the divergence as ONE of:\n"
    "   factual_difference | style_difference | missing_information |"
    " extra_information | wrong_action\n"
    "   If score == 5, use divergence_type = \"none\".\n"
    "3. For each CHECK, decide pass (true) or fail (false) and give a brief "
    "explanation.\n\n"
    "Respond with ONLY a JSON object, no prose before or after, matching this "
    "exact shape:\n"
    "{\n"
    '  "consistency_score": <int 1-5>,\n'
    '  "divergence_type": "<one of the labels above>",\n'
    '  "checks": [\n'
    '    {"check": "<check text>", "pass": <bool>, "explanation": "<brief>"}\n'
    "  ]\n"
    "}"
)


OUTCOME_EVAL_SYSTEM = (
    "You are evaluating whether a conversation between a customer and an "
    "agent achieved its expected outcome. This is classification over the "
    "full ACTUAL transcript (what the agent really said), not the golden.\n\n"
    "For each outcome check, decide pass (true) or fail (false) based on the "
    "conversation. Pass means the check's claim is supported by the "
    "transcript.\n\n"
    "Respond with ONLY a JSON object, no prose before or after, matching this "
    "exact shape:\n"
    "{\n"
    '  "checks": [\n'
    '    {"check": "<check text>", "pass": <bool>, "explanation": "<brief>"}\n'
    "  ]\n"
    "}"
)


STRICT_RETRY_SUFFIX = (
    "\n\nIMPORTANT: your previous response was not valid JSON. Respond with "
    "ONLY the JSON object described above - no code fences, no commentary."
)


def build_turn_eval_prompt(
    *,
    golden_response: str,
    actual_response: str,
    checks: list[str],
) -> tuple[str, str]:
    """Return ``(system, user)`` for a batched per-turn evaluation call."""
    check_lines = "\n".join(f"- {c}" for c in checks) if checks else "(none)"
    user = (
        f"GOLDEN response:\n{golden_response}\n\n"
        f"ACTUAL response:\n{actual_response}\n\n"
        f"CHECKS:\n{check_lines}\n\n"
        "Evaluate now."
    )
    return TURN_EVAL_SYSTEM, user


def build_outcome_eval_prompt(
    *,
    conversation: list[dict[str, str]],
    expected_outcome: str,
    outcome_checks: list[str],
) -> tuple[str, str]:
    """Return ``(system, user)`` for the per-scenario outcome evaluation call.

    ``conversation`` is the ACTUAL transcript, a list of
    ``{"role": "user"|"agent", "content": "..."}`` dicts.
    """
    transcript = json.dumps(conversation, indent=2)
    check_lines = "\n".join(f"- {c}" for c in outcome_checks) if outcome_checks else "(none)"
    user = (
        f"FULL CONVERSATION (actual responses):\n{transcript}\n\n"
        f"EXPECTED OUTCOME: {expected_outcome}\n\n"
        f"OUTCOME CHECKS:\n{check_lines}\n\n"
        "Evaluate now."
    )
    return OUTCOME_EVAL_SYSTEM, user
