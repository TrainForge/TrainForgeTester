"""Shared fixtures for the TrainForge test suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainforge.llm.base import LLMClient


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


@pytest.fixture(scope="session")
def example_scenarios_path() -> Path:
    """Path to the canonical example scenario shipped with TrainForge."""
    return SCENARIOS_DIR / "example_restaurant_booking.json"


@pytest.fixture(scope="session")
def small_scenarios_path() -> Path:
    return FIXTURES_DIR / "scenarios_small.json"


@pytest.fixture(scope="session")
def small_scenarios(small_scenarios_path: Path) -> dict:
    return json.loads(small_scenarios_path.read_text())


@pytest.fixture(scope="session")
def tools_scenarios_path() -> Path:
    return FIXTURES_DIR / "scenarios_tools.json"


class FakeLLM:
    """In-memory ``LLMClient`` that returns whatever responses the test queues.

    Use :meth:`queue_turn_pass_all` / :meth:`queue_turn_fail_all` / etc. to
    avoid constructing raw JSON strings in every test.
    """

    model = "fake-llm-1.0"

    def __init__(self) -> None:
        self._responses: list[str] = []
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self._responses:
            raise AssertionError(
                f"FakeLLM received an unexpected call (no responses queued).\n"
                f"System: {system[:80]}...\nUser: {user[:120]}..."
            )
        return self._responses.pop(0)

    def queue_raw(self, raw: str) -> "FakeLLM":
        self._responses.append(raw)
        return self

    def queue_turn(
        self,
        *,
        consistency_score: int,
        divergence_type: str = "none",
        checks: list[tuple[str, bool, str]] | None = None,
    ) -> "FakeLLM":
        payload = {
            "consistency_score": consistency_score,
            "divergence_type": divergence_type,
            "checks": [
                {"check": c[0], "pass": c[1], "explanation": c[2]}
                for c in (checks or [])
            ],
        }
        self._responses.append(json.dumps(payload))
        return self

    def queue_outcome(
        self, checks: list[tuple[str, bool, str]]
    ) -> "FakeLLM":
        payload = {
            "checks": [
                {"check": c[0], "pass": c[1], "explanation": c[2]}
                for c in checks
            ],
        }
        self._responses.append(json.dumps(payload))
        return self


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


def _assert_protocol(_llm: LLMClient) -> None:  # pragma: no cover - typing aid
    """Dummy reference so mypy/pyright flag protocol drift."""
    ...
