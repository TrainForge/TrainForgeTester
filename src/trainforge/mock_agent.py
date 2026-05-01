"""Development mock agent server.

Used to self-test the runner without a real agent under test, and as a sanity
check for customers setting up TrainForge for the first time. Exposes a
single ``POST /chat`` endpoint that speaks the spec's extended agent API
contract (``{"messages": [...]}`` in, ``{"response"?: str, "tool_calls"?: [...]}``
out; see :mod:`trainforge.agent_client`).

Modes (testing-spec-v1.md Phase 1e):

- ``golden``  : emit the next expected tool_call or the golden text.
- ``diverge`` : perturb the golden text, call the wrong tool name, or shuffle
                arguments - exercises the divergence / validator code paths.
- ``error``   : randomly return HTTP 500 or hang.

State inference: the server looks at the message history to figure out
(a) which scenario + agent-turn we're in (by matching the last customer
message) and (b) how many tool_call rounds the agent has already completed
(by counting non-customer messages since that customer index that contain
tool_calls). Everything is stateless at the connection level.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
import uuid
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from trainforge.schema import (
    AgentTurn,
    ArgumentType,
    CustomerTurn,
    ExpectedTool,
    Scenario,
    ToolArgumentSchema,
    load_scenarios,
)

log = logging.getLogger(__name__)

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11
    class StrEnum(str, Enum):
        pass


class Mode(StrEnum):
    GOLDEN = "golden"
    DIVERGE = "diverge"
    ERROR = "error"


MODES: tuple[Mode, ...] = (Mode.GOLDEN, Mode.DIVERGE, Mode.ERROR)


class MockAgentServer:
    """Wraps a ``ThreadingHTTPServer`` so tests can start/stop it cleanly."""

    def __init__(
        self,
        scenarios_path: str,
        *,
        port: int = 8080,
        host: str = "127.0.0.1",
        mode: Mode | str = Mode.GOLDEN,
        error_rate: float = 0.5,
        timeout_rate: float = 0.25,
        seed: int | None = None,
    ) -> None:
        try:
            parsed_mode = mode if isinstance(mode, Mode) else Mode(mode)
        except ValueError as exc:
            raise ValueError(
                f"mode must be one of {[m.value for m in MODES]}, got {mode!r}"
            ) from exc
        scenarios_file = load_scenarios(scenarios_path)
        self._scenario_index = _build_scenario_index(scenarios_file.scenarios)
        handler_cls = _make_handler(
            scenario_index=self._scenario_index,
            mode=parsed_mode,
            error_rate=error_rate,
            timeout_rate=timeout_rate,
            rng=random.Random(seed),
        )
        self._server = ThreadingHTTPServer((host, port), cast(Any, handler_cls))
        self._thread: threading.Thread | None = None
        self.port = self._server.server_address[1]
        self.host = host
        self.mode = parsed_mode

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/chat"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="trainforge-mock-agent"
        )
        self._thread.start()
        log.info("mock-agent (%s) listening on %s", self.mode.value, self.url)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)
        self._thread = None

    def serve_forever(self) -> None:
        """Blocking server loop; used by the CLI."""
        log.info("mock-agent (%s) listening on %s", self.mode.value, self.url)
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()


# ---------------------------------------------------------------------------
# Scenario index: last customer message -> the full agent turn to serve
# ---------------------------------------------------------------------------


def _build_scenario_index(scenarios: list[Scenario]) -> dict[str, AgentTurn]:
    """Map the customer-message string to the agent turn that follows it.

    Ambiguity: first occurrence wins. Distinct scenarios in a single file
    must use distinct customer wording, which is already true for MVP.
    """
    index: dict[str, AgentTurn] = {}
    for scenario in scenarios:
        pending_customer: CustomerTurn | None = None
        for turn in scenario.turns:
            if isinstance(turn, CustomerTurn):
                pending_customer = turn
                continue
            if isinstance(turn, AgentTurn) and pending_customer is not None:
                index.setdefault(pending_customer.message, turn)
                pending_customer = None
    return index


# ---------------------------------------------------------------------------
# Response planning
# ---------------------------------------------------------------------------


def _plan_response(
    agent_turn: AgentTurn, history: list[dict], mode: Mode, rng: random.Random
) -> dict:
    """Decide what to emit for the given (agent_turn, history) state.

    Returns a dict ready to ``json.dumps`` back to the client - either a
    ``{"response": str}`` text turn or a ``{"tool_calls": [...]}`` round.
    """
    position = _pending_tool_position(agent_turn, history)
    if position is None:
        # All tool_calls are done (or the turn has none). Return golden text.
        text = agent_turn.golden_response
        if mode == Mode.DIVERGE:
            text = _perturb_text(text, rng)
        return {"response": text}

    loop_idx, tool_pos = position
    loop = agent_turn.tool_loops[loop_idx]
    expected = loop.tools[tool_pos]
    call = _expected_to_call(expected, mode=mode, rng=rng)
    return {"response": "", "tool_calls": [call]}


def _pending_tool_position(
    agent_turn: AgentTurn, history: list[dict]
) -> tuple[int, int] | None:
    """Return (loop_index, position) of the next tool to emit, or None.

    Counts the number of agent messages with ``tool_calls`` since the last
    user message in ``history``, then walks the turn's tool_loops in
    order to locate the next slot.
    """
    last_user = -1
    for i, msg in enumerate(history):
        if _is_user_role(msg.get("role")):
            last_user = i

    tool_rounds = 0
    for msg in history[last_user + 1 :]:
        if msg.get("role") == "agent" and msg.get("tool_calls"):
            tool_rounds += len(msg["tool_calls"])

    consumed = tool_rounds
    for loop_idx, loop in enumerate(agent_turn.tool_loops):
        loop_size = len(loop.tools)
        if consumed < loop_size:
            return loop_idx, consumed
        consumed -= loop_size
    return None


def _expected_to_call(
    expected: ExpectedTool, *, mode: Mode, rng: random.Random
) -> dict:
    args = _fill_arguments(expected.arguments_schema, rng=rng)
    name = expected.name
    if mode == Mode.DIVERGE:
        name, args = _perturb_tool_call(expected, args, rng)
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "name": name,
        "arguments": args,
    }


def _fill_arguments(
    schema: dict[str, ToolArgumentSchema], *, rng: random.Random
) -> dict:
    """Fabricate a minimal valid argument dict.

    In ``golden`` mode we prefer the schema's declared ``expected`` literal
    when present, so the mock emits a call that passes validation. Arguments
    without an ``expected`` literal but with natural-language ``checks``
    fall back to a type-based sample - those scenarios may fail the LLM
    check in golden mode, which is a signal to the author that they need
    to set ``expected`` if they want a deterministic golden flow.
    """
    args: dict[str, object] = {}
    for key, arg in schema.items():
        if arg.expected is not None:
            args[key] = arg.expected
        else:
            args[key] = _sample(arg.type, rng)
    return args


def _sample(t: ArgumentType, rng: random.Random) -> object:
    if t == ArgumentType.STRING:
        return rng.choice(["example", "sample", "demo"])
    if t == ArgumentType.INTEGER:
        return rng.randint(1, 10)
    if t == ArgumentType.NUMBER:
        return round(rng.uniform(0.0, 100.0), 2)
    if t == ArgumentType.BOOLEAN:
        return bool(rng.getrandbits(1))
    if t == ArgumentType.ARRAY:
        return []
    if t == ArgumentType.OBJECT:
        return {}
    return "value"


# ---------------------------------------------------------------------------
# Perturbations (diverge mode)
# ---------------------------------------------------------------------------


def _perturb_text(golden: str, rng: random.Random) -> str:
    """Always returns a string different from ``golden``."""
    flip = rng.random()
    swapped = (
        golden.replace("cold", "warm")
        .replace("rainy", "sunny")
        .replace("indoor", "outdoor")
    )
    if flip < 0.4 and swapped != golden:
        return swapped
    if flip < 0.7:
        return "Sure, let me check that for you."
    return (golden + " Also, have a great day!").strip()


def _perturb_tool_call(
    expected: ExpectedTool, args: dict, rng: random.Random
) -> tuple[str, dict]:
    """Return ``(name, arguments)`` that differs from the expected shape."""
    flip = rng.random()
    if flip < 0.5:
        return expected.name + "_unknown", args
    if flip < 0.8 and expected.arguments_schema:
        # Drop one required key.
        args = dict(args)
        dropped = next(iter(expected.arguments_schema))
        args.pop(dropped, None)
        return expected.name, args
    return "totally_unrelated_tool", {"bogus": True}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


def _make_handler(
    *,
    scenario_index: dict[str, AgentTurn],
    mode: Mode,
    error_rate: float,
    timeout_rate: float,
    rng: random.Random,
) -> type[BaseHTTPRequestHandler]:
    """Factory because :class:`BaseHTTPRequestHandler` doesn't accept deps."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            log.debug("mock-agent " + format, *args)

        def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
            if self.path not in ("/chat", "/"):
                self._send_json(404, {"error": f"unknown path {self.path}"})
                return

            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return

            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                self._send_json(400, {"error": "'messages' must be a non-empty list"})
                return

            last_customer_msg = _last_user_content(messages)
            if last_customer_msg is None:
                self._send_json(
                    400,
                    {"error": "history must contain at least one user message"},
                )
                return

            agent_turn = scenario_index.get(last_customer_msg)
            if agent_turn is None:
                self._send_json(
                    404,
                    {"error": f"no scenario indexed for message: {last_customer_msg!r}"},
                )
                return

            if mode == Mode.ERROR:
                roll = rng.random()
                if roll < timeout_rate:
                    time.sleep(45.0)
                    try:
                        self._send_json(200, _plan_response(agent_turn, messages, Mode.GOLDEN, rng))
                    except Exception:  # pragma: no cover
                        pass
                    return
                if roll < timeout_rate + error_rate:
                    self._send_json(500, {"error": "simulated failure"})
                    return

            response = _plan_response(agent_turn, messages, mode, rng)
            self._send_json(200, response)

        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return cast(type[BaseHTTPRequestHandler], _Handler)


def _last_user_content(messages: list[dict]) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, dict) and _is_user_role(msg.get("role")):
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return None


def _is_user_role(value: object) -> bool:
    return value in {"user", "customer"}
