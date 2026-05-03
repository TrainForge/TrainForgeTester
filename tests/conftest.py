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
    """In-memory ``LLMClient`` returning queued responses in order.

    All evaluators in TrainForge 0.1 speak the compact ``{"r": [...], "f": {...}}``
    format, so test helpers center on that shape.
    """

    model = "fake-llm-2.0"

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

    def queue_compact(
        self,
        results: list[int | bool],
        failures: dict[str, str] | None = None,
    ) -> "FakeLLM":
        """Queue one compact-format response.

        ``results`` is the positional 1/0 (or True/False) array. ``failures``
        is a dict from 1-based index strings to brief reasons; pass
        explanations only for the indices set to 0/False.
        """
        self._responses.append(
            json.dumps({"r": results, "f": failures or {}})
        )
        return self

    def queue_all_pass(self, n: int) -> "FakeLLM":
        """Shortcut: ``n`` ones, no failures."""
        return self.queue_compact([1] * n)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


def _assert_protocol(_llm: LLMClient) -> None:  # pragma: no cover - typing aid
    """Dummy reference so mypy/pyright flag protocol drift."""
    ...
