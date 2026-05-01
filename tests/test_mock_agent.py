"""Mock agent HTTP server tests."""
from __future__ import annotations

import socket
from pathlib import Path

import pytest
import requests

from trainforge.mock_agent import MockAgentServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def golden_server(small_scenarios_path: Path):
    server = MockAgentServer(str(small_scenarios_path), port=_free_port(), mode="golden")
    server.start()
    yield server
    server.stop()


def test_golden_mode_returns_exact_golden(golden_server, small_scenarios) -> None:
    first_customer = small_scenarios["scenarios"][0]["turns"][0]["message"]
    first_golden = small_scenarios["scenarios"][0]["turns"][1]["golden_response"]

    resp = requests.post(
        golden_server.url,
        json={"messages": [{"role": "user", "content": first_customer}]},
        timeout=5.0,
    )
    assert resp.status_code == 200
    assert resp.json() == {"response": first_golden}


def test_unknown_customer_message_returns_404(golden_server) -> None:
    resp = requests.post(
        golden_server.url,
        json={"messages": [{"role": "user", "content": "totally unknown"}]},
        timeout=5.0,
    )
    assert resp.status_code == 404


def test_bad_payload_returns_400(golden_server) -> None:
    resp = requests.post(
        golden_server.url, data="not json", timeout=5.0
    )
    assert resp.status_code == 400


def test_diverge_mode_returns_perturbed(small_scenarios_path: Path, small_scenarios) -> None:
    server = MockAgentServer(str(small_scenarios_path), port=_free_port(), mode="diverge", seed=1)
    server.start()
    try:
        first_customer = small_scenarios["scenarios"][0]["turns"][0]["message"]
        first_golden = small_scenarios["scenarios"][0]["turns"][1]["golden_response"]
        resp = requests.post(
            server.url,
            json={"messages": [{"role": "user", "content": first_customer}]},
            timeout=5.0,
        )
        assert resp.status_code == 200
        assert resp.json()["response"] != first_golden
    finally:
        server.stop()


def test_error_mode_eventually_returns_error_status(small_scenarios_path: Path, small_scenarios) -> None:
    """With seed=0 and high error_rate the mock returns a non-200 response."""
    server = MockAgentServer(
        str(small_scenarios_path),
        port=_free_port(),
        mode="error",
        error_rate=1.0,
        timeout_rate=0.0,
        seed=0,
    )
    server.start()
    try:
        first_customer = small_scenarios["scenarios"][0]["turns"][0]["message"]
        resp = requests.post(
            server.url,
            json={"messages": [{"role": "user", "content": first_customer}]},
            timeout=5.0,
        )
        assert resp.status_code == 500
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# v1.1 tool-call support
# ---------------------------------------------------------------------------


@pytest.fixture
def tools_server(tools_scenarios_path: Path):
    server = MockAgentServer(str(tools_scenarios_path), port=_free_port(), mode="golden")
    server.start()
    yield server
    server.stop()


def test_golden_mock_emits_tool_calls_for_tool_loops(tools_server) -> None:
    customer_msg = "Book me a table for 2 at 7pm, indoor if it's raining."

    # Round 1: empty history -> mock should emit a tool_call for loop 0.
    resp = requests.post(
        tools_server.url,
        json={"messages": [{"role": "user", "content": customer_msg}]},
        timeout=5.0,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "tool_calls" in body
    assert len(body["tool_calls"]) == 1
    first_tool = body["tool_calls"][0]
    assert first_tool["name"] in {"check_weather", "check_availability"}


def test_golden_mock_advances_after_tool_responses(tools_server) -> None:
    customer_msg = "Book me a table for 2 at 7pm, indoor if it's raining."
    history = [
        {"role": "user", "content": customer_msg},
        {
            "role": "agent",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "check_weather", "arguments": {"when": "tonight"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "check_weather", "content": "Tonight: cold and rainy."},
        {
            "role": "agent",
            "content": "",
            "tool_calls": [{"id": "c2", "name": "check_availability", "arguments": {"party_size": 2, "time": "7pm"}}],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "check_availability", "content": "Tables: corner (indoor), window (indoor)."},
    ]
    resp = requests.post(tools_server.url, json={"messages": history}, timeout=5.0)
    assert resp.status_code == 200
    body = resp.json()
    # Loop 0 has 2 tools, both consumed -> next is loop 1 (book_table).
    assert body.get("tool_calls"), body
    assert body["tool_calls"][0]["name"] == "book_table"


def test_golden_mock_emits_final_text_after_all_tools(tools_server) -> None:
    customer_msg = "Book me a table for 2 at 7pm, indoor if it's raining."
    history = [
        {"role": "user", "content": customer_msg},
        {
            "role": "agent",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "check_weather", "arguments": {"when": "tonight"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "check_weather", "content": "Tonight: cold and rainy."},
        {
            "role": "agent",
            "content": "",
            "tool_calls": [{"id": "c2", "name": "check_availability", "arguments": {"party_size": 2, "time": "7pm"}}],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "check_availability", "content": "Tables: corner (indoor), window (indoor)."},
        {
            "role": "agent",
            "content": "",
            "tool_calls": [{"id": "c3", "name": "book_table", "arguments": {"party_size": 2, "time": "7pm", "seating": "indoor"}}],
        },
        {"role": "tool", "tool_call_id": "c3", "name": "book_table", "content": "Booking confirmed. Reference: X-7."},
    ]
    resp = requests.post(tools_server.url, json={"messages": history}, timeout=5.0)
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("response")
    assert not body.get("tool_calls")
