"""Evaluator prompts.

Two prompt builders, both producing the **compact batched output format**:

    {"r": [1, 0, 1, 1, 0, ...], "f": {"3": "different language", "5": "..."}}

- ``r`` is a positional array of 1 (pass) and 0 (fail), length must equal
  the number of asked questions, in the same order.
- ``f`` is an object mapping the 1-based index of FAILED questions to a
  short reason. Pass questions have no entry here.

Token economics: 20 questions all passing = ~30 output tokens. Versus the
old "score 1-5 + divergence_type + per-check JSON object" format which
typically produced 150-500 output tokens. ~5-10x cheaper per turn.

Two prompts:

- :func:`build_turn_eval_prompt` - per-agent-turn evaluation. Used only
  when ``AgentTurn.may_diverge=True``; otherwise the runner does a Python
  ``==`` text match and never calls the LLM. Composes the 20 standard
  NLP-consistency checks with any per-scenario custom checks into one
  batched call.
- :func:`build_outcome_eval_prompt` - end-of-conversation outcome
  evaluation over the full ACTUAL transcript.
"""
from __future__ import annotations

import json


TURN_EVAL_SYSTEM = (
    "You are an NLP comparison oracle for an AI-agent test harness. Your "
    "only job is to compare an ACTUAL agent reply against a GOLDEN agent "
    "reply along a fixed list of binary questions. You do not score "
    "quality; you only answer each question with 1 (yes) or 0 (no).\n\n"
    "You are given:\n"
    "  - the GOLDEN response (the expected agent reply)\n"
    "  - the ACTUAL response (what the agent under test produced)\n"
    "  - the most recent USER message (so you can judge language and topic)\n"
    "  - a numbered list of QUESTIONS, each phrased so that yes=1 and no=0\n\n"
    "Rules:\n"
    " 1. Answer every question independently. Do NOT let one answer bias another.\n"
    " 2. Each answer is 1 or 0. No 'partly', no fractions, no nulls.\n"
    " 3. If a question doesn't apply (e.g. asks about numbers when neither "
    "    response mentions any), answer 1.\n"
    " 4. Keep failure explanations under 120 characters.\n\n"
    "Respond with ONLY a JSON object matching exactly this shape:\n"
    '{"r": [1, 0, 1, ...], "f": {"3": "<brief reason>", "5": "<brief reason>"}}\n\n'
    "- r is an array of 1s and 0s, length exactly N (the number of questions),\n"
    "  in the same order as the questions.\n"
    "- f is an object whose keys are 1-based indices of FAILED questions only,\n"
    "  values are short reason strings. Omit keys for passed questions.\n"
    "Output nothing else - no prose, no code fences."
)


OUTCOME_EVAL_SYSTEM = (
    "You are evaluating whether a conversation between a user and an AI "
    "agent achieved its expected outcome. Read the full ACTUAL transcript "
    "(what the agent really did) and answer each outcome check with 1 "
    "(yes) or 0 (no).\n\n"
    "Rules:\n"
    " 1. Pass (1) means the check's claim is supported by the transcript.\n"
    " 2. Fail (0) means the claim is not supported.\n"
    " 3. Keep failure explanations under 120 characters.\n\n"
    "Respond with ONLY a JSON object matching exactly this shape:\n"
    '{"r": [1, 0, 1, ...], "f": {"2": "<brief reason>"}}\n\n'
    "- r length must equal the number of outcome checks, in order.\n"
    "- f maps 1-based indices of failed checks to short reasons.\n"
    "Output nothing else."
)


STRICT_RETRY_SUFFIX = (
    "\n\nIMPORTANT: your previous response was not parseable. Respond with "
    'ONLY the JSON object {"r": [...], "f": {...}} described above. No code '
    "fences, no commentary, no extra fields. r length must match the number "
    "of questions exactly."
)


def build_turn_eval_prompt(
    *,
    golden_response: str,
    actual_response: str,
    user_message: str,
    questions: list[str],
) -> tuple[str, str]:
    """Build the (system, user) pair for one batched turn-eval call.

    ``questions`` should already be the concatenation of standard checks
    (from :mod:`trainforge.standard_checks`) plus any per-scenario custom
    checks. The runner is responsible for splitting the response back into
    the two buckets by index.
    """
    if not questions:
        raise ValueError("build_turn_eval_prompt: questions must be non-empty")
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    user = (
        f"USER message (most recent):\n{user_message}\n\n"
        f"GOLDEN agent response:\n{golden_response}\n\n"
        f"ACTUAL agent response:\n{actual_response}\n\n"
        f"QUESTIONS ({len(questions)} total):\n{numbered}\n\n"
        "Evaluate each question now."
    )
    return TURN_EVAL_SYSTEM, user


def build_outcome_eval_prompt(
    *,
    conversation: list[dict],
    expected_outcome: str,
    outcome_checks: list[str],
) -> tuple[str, str]:
    """Build the (system, user) pair for the per-scenario outcome eval call.

    ``conversation`` is the ACTUAL transcript (a list of role/content/tool
    message dicts; the LLM gets the JSON dump verbatim).
    """
    if not outcome_checks:
        raise ValueError(
            "build_outcome_eval_prompt: outcome_checks must be non-empty"
        )
    transcript = json.dumps(conversation, indent=2)
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(outcome_checks))
    user = (
        f"FULL CONVERSATION (actual transcript):\n{transcript}\n\n"
        f"EXPECTED OUTCOME:\n{expected_outcome}\n\n"
        f"OUTCOME CHECKS ({len(outcome_checks)} total):\n{numbered}\n\n"
        "Evaluate each check now."
    )
    return OUTCOME_EVAL_SYSTEM, user
