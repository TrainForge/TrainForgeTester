"""Custom exceptions for TrainForge.

The runner maps these to the error categories in the spec's
"Error Handling" table and propagates them into results.json.
"""
from __future__ import annotations


class TrainForgeError(Exception):
    """Base class for all TrainForge errors."""


class MalformedScenarioError(TrainForgeError):
    """Scenario JSON failed validation against the format spec."""


class UnsupportedScenarioVersionError(MalformedScenarioError):
    """Scenario file has a `version` the runner does not support."""


class AgentError(TrainForgeError):
    """Agent API returned a non-2xx response."""


class AgentTimeoutError(TrainForgeError):
    """Agent API exceeded the configured timeout (after one retry)."""


class AgentUnreachableError(TrainForgeError):
    """Agent API is unreachable (connection refused, DNS, etc)."""


class EvaluationError(TrainForgeError):
    """LLM evaluator failed even after a strict-retry."""
