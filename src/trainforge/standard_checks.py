"""The 20 standard NLP-consistency checks applied to every may_diverge agent turn.

These are the only LLM-judged claims TrainForge makes about an agent's text
response. They are deliberately:

- **Binary** - each check returns 1 (pass) or 0 (fail). No 0-1 quality
  scores, no 1-5 vibes; the LLM never has to "rate" anything.
- **Comparative** - every check is "does the actual AI response have
  property X *relative to the golden AI response*?" not "is the response
  good?". This keeps the judge's job small and well-scoped.
- **Hard-coded and stable** - the list ships with the runner. The ``id``
  field is part of the public results contract; renaming an id is a
  breaking change for downstream report diffs.

When ``AgentTurn.may_diverge=False`` (the default), this whole battery is
*skipped*. The runner does a Python ``==`` exact-match check between
``actual_response`` and ``golden_response``; that is the deterministic
default for curated / verbatim agent replies (legal disclaimers, scripted
responses, etc.).

When ``AgentTurn.may_diverge=True``, the runner composes
``STANDARD_CHECKS + scenario.custom_checks`` into a single batched LLM
call. See :mod:`trainforge.llm.prompts` for the wire format.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StandardCheck:
    """One binary NLP comparison between actual and golden agent responses."""

    id: str
    """Stable short identifier. Used in results.json and the HTML report.
    NEVER change an id without bumping ``RESULTS_FORMAT_VERSION`` - downstream
    consumers diff scenarios by check id."""

    question: str
    """The exact natural-language question fed to the LLM judge. Phrased so
    that pass=1 means "the comparison holds" and fail=0 means "it doesn't".
    """


STANDARD_CHECKS: tuple[StandardCheck, ...] = (
    StandardCheck(
        id="same_language",
        question="Both responses are written in the same natural language.",
    ),
    StandardCheck(
        id="same_speech_act",
        question=(
            "Both responses perform the same primary speech act "
            "(statement, question, confirmation, request, promise, apology, refusal)."
        ),
    ),
    StandardCheck(
        id="same_intent",
        question=(
            "Both responses share the same communicative intent "
            "(e.g., both are clarifying questions; both are booking confirmations; "
            "both are escalations to a human)."
        ),
    ),
    StandardCheck(
        id="same_action_state",
        question=(
            "Both responses indicate the same action state for the user-facing task "
            "(not started / pending / in progress / completed / failed)."
        ),
    ),
    StandardCheck(
        id="same_next_step",
        question=(
            "Both responses prompt the user for the same next step "
            "(or both signal that no next step from the user is needed)."
        ),
    ),
    StandardCheck(
        id="same_propositional_content",
        question="Both responses express the same set of factual claims.",
    ),
    StandardCheck(
        id="no_added_facts",
        question=(
            "The actual response contains no factual claims that are absent from "
            "the golden response."
        ),
    ),
    StandardCheck(
        id="no_omitted_facts",
        question=(
            "The actual response omits no factual claims that are present in "
            "the golden response."
        ),
    ),
    StandardCheck(
        id="no_contradictions",
        question=(
            "The actual response does not contradict any claim made in the golden "
            "response."
        ),
    ),
    StandardCheck(
        id="same_named_entities",
        question=(
            "Both responses reference the same named entities "
            "(people, places, products, organizations)."
        ),
    ),
    StandardCheck(
        id="same_numerics",
        question=(
            "Numbers, dates, times, codes, and identifiers match wherever both "
            "responses mention them."
        ),
    ),
    StandardCheck(
        id="same_call_to_action",
        question=(
            "Both responses contain (or both omit) the same call-to-action "
            "(e.g., 'is there anything else I can help with?')."
        ),
    ),
    StandardCheck(
        id="same_disclosures",
        question=(
            "Both responses include (or both omit) the same disclosures, caveats, "
            "warnings, or required notices."
        ),
    ),
    StandardCheck(
        id="comparable_register",
        question=(
            "Both responses are written in a comparable register "
            "(formal / casual / technical / consumer-friendly)."
        ),
    ),
    StandardCheck(
        id="comparable_tone",
        question=(
            "Both responses are written in a comparable tone "
            "(polite / curt / empathetic / neutral / enthusiastic)."
        ),
    ),
    StandardCheck(
        id="comparable_specificity",
        question=(
            "Both responses are at a comparable level of specificity "
            "(concrete details vs generic placeholders)."
        ),
    ),
    StandardCheck(
        id="comparable_hedging",
        question=(
            "Both responses are at a comparable level of confidence and hedging "
            "(decisive vs tentative)."
        ),
    ),
    StandardCheck(
        id="comparable_length",
        question=(
            "The actual response is at a comparable length to the golden response "
            "(not dramatically compressed or padded; roughly within 0.5x to 2x)."
        ),
    ),
    StandardCheck(
        id="same_persona",
        question=(
            "Both responses maintain the same persona / voice; neither breaks "
            "character (e.g., neither slips into developer-mode or third-person)."
        ),
    ),
    StandardCheck(
        id="same_information_order",
        question=(
            "Both responses present major information units in the same order "
            "(e.g., greeting -> confirmation -> reference; not reference -> greeting)."
        ),
    ),
)


def standard_check_ids() -> list[str]:
    """Convenience accessor for the canonical list of standard check ids."""
    return [c.id for c in STANDARD_CHECKS]


def standard_check_questions() -> list[str]:
    """Convenience accessor for the canonical list of standard check questions."""
    return [c.question for c in STANDARD_CHECKS]
