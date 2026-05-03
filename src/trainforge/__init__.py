"""TrainForge: open-source conversational-agent test runner."""

__version__ = "0.1.0"
SCENARIO_FORMAT_VERSION = "2.0"
"""Scenario format 2.0 is the first public TrainForge 0.1.0 format:
- Role ``customer`` renamed to ``user`` (aligns with OpenAI conventions).
- ``AgentTurn.may_diverge`` default flipped to ``False``: by default the
  runner does an exact ``==`` text match between the agent's actual reply
  and ``golden_response`` and skips the LLM-judged standard-checks battery.
  Set ``may_diverge: true`` per turn to opt in to the NLP comparison.
- Removed ``consistency_score`` and ``divergence_type`` from results in
  favour of the 20 binary :mod:`trainforge.standard_checks`."""
RESULTS_FORMAT_VERSION = "2.0"
