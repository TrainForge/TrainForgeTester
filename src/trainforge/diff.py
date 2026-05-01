"""Regression comparison for two :class:`RunResults` files.

``trainforge diff`` loads two result files and bucketizes every scenario into
one of:

- ``newly_passing``     - failed in ``before``, passes in ``after``.
- ``newly_failing``     - passed in ``before``, fails in ``after``.
- ``still_passing``     - passed in both.
- ``still_failing``     - failed in both.
- ``consistency_changed`` - passing status unchanged but consistency moved.
- ``only_in_before``    - scenario removed after the change.
- ``only_in_after``     - scenario added after the change.

Scenario identity is ``scenario_id``; names are used only for display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from trainforge.schema import RunResults, ScenarioResult

Bucket = Literal[
    "newly_passing",
    "newly_failing",
    "still_passing",
    "still_failing",
    "consistency_changed",
    "only_in_before",
    "only_in_after",
]


@dataclass(frozen=True)
class ScenarioDiff:
    scenario_id: str
    name: str
    bucket: Bucket
    before_pass: bool | None
    after_pass: bool | None
    before_consistency: float | None
    after_consistency: float | None


@dataclass
class DiffReport:
    before_path: str
    after_path: str
    before: RunResults
    after: RunResults
    newly_passing: list[ScenarioDiff] = field(default_factory=list)
    newly_failing: list[ScenarioDiff] = field(default_factory=list)
    still_passing: list[ScenarioDiff] = field(default_factory=list)
    still_failing: list[ScenarioDiff] = field(default_factory=list)
    consistency_changed: list[ScenarioDiff] = field(default_factory=list)
    only_in_before: list[ScenarioDiff] = field(default_factory=list)
    only_in_after: list[ScenarioDiff] = field(default_factory=list)

    @property
    def all_entries(self) -> list[ScenarioDiff]:
        return (
            self.newly_failing
            + self.newly_passing
            + self.consistency_changed
            + self.only_in_before
            + self.only_in_after
            + self.still_failing
            + self.still_passing
        )

    @property
    def regressed_count(self) -> int:
        return len(self.newly_failing)

    @property
    def fixed_count(self) -> int:
        return len(self.newly_passing)


def compute_diff(
    before: RunResults,
    after: RunResults,
    *,
    before_path: str = "",
    after_path: str = "",
    consistency_epsilon: float = 0.1,
) -> DiffReport:
    """Bucketize scenarios into a :class:`DiffReport`.

    ``consistency_epsilon`` controls the threshold for flagging
    ``consistency_changed`` when the pass/fail status is unchanged.
    """
    report = DiffReport(
        before_path=before_path,
        after_path=after_path,
        before=before,
        after=after,
    )

    before_by_id = {s.scenario_id: s for s in before.scenarios}
    after_by_id = {s.scenario_id: s for s in after.scenarios}
    all_ids = set(before_by_id) | set(after_by_id)

    for scenario_id in sorted(all_ids):
        b = before_by_id.get(scenario_id)
        a = after_by_id.get(scenario_id)

        if b is None and a is not None:
            report.only_in_after.append(_diff_entry(a, bucket="only_in_after", side="after"))
            continue
        if a is None and b is not None:
            report.only_in_before.append(_diff_entry(b, bucket="only_in_before", side="before"))
            continue
        assert a is not None and b is not None

        b_pass = _pass_flag(b)
        a_pass = _pass_flag(a)

        if b_pass and not a_pass:
            bucket: Bucket = "newly_failing"
        elif not b_pass and a_pass:
            bucket = "newly_passing"
        else:
            moved = abs(a.consistency - b.consistency) >= consistency_epsilon
            if moved:
                bucket = "consistency_changed"
            elif b_pass and a_pass:
                bucket = "still_passing"
            else:
                bucket = "still_failing"

        entry = ScenarioDiff(
            scenario_id=scenario_id,
            name=a.name or b.name,
            bucket=bucket,
            before_pass=b_pass,
            after_pass=a_pass,
            before_consistency=b.consistency,
            after_consistency=a.consistency,
        )
        getattr(report, bucket).append(entry)

    return report


def _pass_flag(scenario: ScenarioResult) -> bool:
    """A scenario "passes" a run if any run came back with status ``pass``."""
    return any(r.status == "pass" for r in scenario.runs)


def _diff_entry(scenario: ScenarioResult, *, bucket: Bucket, side: str) -> ScenarioDiff:
    passed = _pass_flag(scenario)
    return ScenarioDiff(
        scenario_id=scenario.scenario_id,
        name=scenario.name,
        bucket=bucket,
        before_pass=passed if side == "before" else None,
        after_pass=passed if side == "after" else None,
        before_consistency=scenario.consistency if side == "before" else None,
        after_consistency=scenario.consistency if side == "after" else None,
    )
