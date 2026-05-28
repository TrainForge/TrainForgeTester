"""`trainforge record --auto` behavior.

The interactive REPL itself isn't easy to drive in tests, but the
post-session save path (``_auto_finalize_and_save``) is a pure
function that takes a captured session and writes a scenario file.
That's where the defaults live, and it's what we cover here.
"""
from __future__ import annotations

import json
from pathlib import Path

from trainforge.record import (
    _CapturedTurn,
    _RecordSession,
    _auto_finalize_and_save,
)
from trainforge.schema import ScenariosFile, parse_scenarios
from trainforge.transport import InProcessTransport


def _make_session_with_turns():
    session = _RecordSession(
        transport=InProcessTransport(agent=lambda m: {"response": ""})
    )
    session.turns.append(
        _CapturedTurn(user_message="hi", agent_text="hello")
    )
    session.turns.append(
        _CapturedTurn(
            user_message="book a table at 7pm",
            agent_text="booked the corner table for 2 at 7pm",
        )
    )
    return session


def test_auto_save_produces_valid_scenario_json(tmp_path: Path) -> None:
    session = _make_session_with_turns()
    out = tmp_path / "auto.json"

    exit_code = _auto_finalize_and_save(session, str(out))
    assert exit_code == 0
    assert out.exists()

    data = json.loads(out.read_text(encoding="utf-8"))
    # Must validate against the schema.
    parsed = parse_scenarios(data)
    assert isinstance(parsed, ScenariosFile)
    assert len(parsed.scenarios) == 1


def test_auto_save_defaults_may_diverge_true_on_every_turn(tmp_path: Path) -> None:
    session = _make_session_with_turns()
    out = tmp_path / "auto.json"
    _auto_finalize_and_save(session, str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    agent_turns = [t for t in data["scenarios"][0]["turns"] if t["role"] == "agent"]
    assert len(agent_turns) == 2
    for t in agent_turns:
        assert t["may_diverge"] is True


def test_auto_save_emits_empty_outcome_checks(tmp_path: Path) -> None:
    """Auto-mode never asks the user for outcome checks. The scenario
    has no outcome_checks (or an empty list); rescoring later is
    purely deterministic until the user adds them."""
    session = _make_session_with_turns()
    out = tmp_path / "auto.json"
    _auto_finalize_and_save(session, str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    sc = data["scenarios"][0]
    assert sc.get("outcome_checks", []) == []


def test_auto_save_scenario_id_starts_with_sc_auto_prefix(tmp_path: Path) -> None:
    session = _make_session_with_turns()
    out = tmp_path / "auto.json"
    _auto_finalize_and_save(session, str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scenarios"][0]["id"].startswith("sc-auto-")


def test_auto_save_creates_parent_dirs(tmp_path: Path) -> None:
    """The user shouldn't have to mkdir scenarios/ before saving."""
    session = _make_session_with_turns()
    out = tmp_path / "nested" / "deeper" / "scenarios" / "auto.json"
    _auto_finalize_and_save(session, str(out))
    assert out.exists()


def test_auto_save_tags_include_auto_marker(tmp_path: Path) -> None:
    """`tags: ["auto"]` lets users filter auto-generated scenarios out
    of CI / reports if they want."""
    session = _make_session_with_turns()
    out = tmp_path / "auto.json"
    _auto_finalize_and_save(session, str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "auto" in data["scenarios"][0]["tags"]
