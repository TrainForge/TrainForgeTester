# Agent contracts

TrainForge speaks to your agent through one of two transports:

- **In-process** — TrainForge imports your Python function and calls
  it directly. No HTTP, no server, no LLM key required for
  deterministic scenarios. Recommended for hackathon, indie, notebook,
  and local development workflows.
- **HTTP** — TrainForge POSTs to your agent's endpoint over HTTP.
  Recommended for testing agents already running in production.

Both transports produce identical results; the runner doesn't care
which one you used.

## In-process transport

### Callable shape

```python
async def run(messages: list[dict]) -> dict:
    return {
        "response": "optional text",
        "tool_calls": [
            {"id": "...", "name": "...", "arguments": {...}},
            ...
        ],
    }
```

| Field | Required | Description |
|---|---|---|
| `response` | one of `response` or `tool_calls` required | Final text reply, if the agent is done with tools. |
| `tool_calls` | one of `response` or `tool_calls` required | Tool calls the agent wants to make. The runner injects the canned `expected_response` for each, then calls the agent again. |

Same shape as the HTTP response below. The contract is unified.

### Sync callables

Sync callables (`def run(...)`) are accepted; the runner wraps them via
`asyncio.to_thread`. Async-native frameworks (LangGraph, OpenAI Agents
SDK, Anthropic Skills) call cleanly without any boilerplate.

### Factory pattern

If your agent needs init (clients, configs, model objects), make a
factory and use the `module:factory()` syntax:

```python
# my_module.py
def make_agent():
    client = SomeClient(...)
    async def agent(messages):
        return await client.chat(messages)
    return agent
```

```bash
trainforge run --agent my_module:make_agent() --scenarios ...
```

The factory runs once at startup; the returned callable is used for
every scenario turn.

### Message shape (what your agent receives)

```python
[
    {"role": "user", "content": "..."},
    {"role": "agent", "content": "..."},
    {"role": "agent", "content": None, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."},
    {"role": "user", "content": "..."},
]
```

Roles: `user`, `agent`, `tool`. Same vocabulary as OpenAI's chat
completions, except `agent` replaces `assistant`. The runner mutates
the list as it goes (golden injection); your agent should treat it as
read-only.

The runner sends the full conversation with every call. The agent is
stateless from the runner's perspective.

## HTTP transport

### Wire format

```
POST <agent-url>
Content-Type: application/json

Request:
{
  "messages": [
    {"role": "user",  "content": "..."},
    {"role": "agent", "content": "..."},
    {"role": "agent", "content": null, "tool_calls": [
      {"id": "call_1", "name": "check_weather", "arguments": {"when": "tonight"}}
    ]},
    {"role": "tool",  "tool_call_id": "call_1", "name": "check_weather",
     "content": "Tonight: cold and rainy."},
    {"role": "user",  "content": "..."}
  ]
}

Response:
{
  "response":   "optional text",
  "tool_calls": [{"id": "...", "name": "...", "arguments": {...}}, ...]
}
```

At least one of `response` or `tool_calls` must be present in the
response. Agents without tool support can keep returning
`{"response": "..."}` only — scenarios without `tool_loops` are
fully backward-compatible.

On a tool-round agent message (the third message in the request above:
the agent emitted `tool_calls`), `content` is JSON `null`, not an empty
string. The runner sends the golden history this way verbatim. Agent
implementations that strictly require a string should treat `null`
the same as `""`.

### Implementation notes

- Connections aren't pooled across `chat()` calls. For large suites,
  use `--parallel N` to overlap requests via `asyncio.gather`.
- The transport doesn't follow HTTP redirects. 3xx responses are
  treated as errors (the spec says non-2xx is `AgentError`).

## Error mapping (both transports)

| Situation | Runner behavior |
|---|---|
| Agent returns non-2xx (HTTP) / raises Python exception (in-process) | Mark turn `agent_error`. Continue. |
| Agent times out | Retry once (HTTP). Second timeout → mark `agent_timeout`. In-process uses `asyncio.wait_for`. |
| Agent unreachable (HTTP only) | Mark scenario `agent_unreachable`. Skip. |
| Agent returns empty body / malformed shape | Treat as divergence. All checks fail. Continue. |
| LLM judge returns unparseable JSON | Retry once with stricter prompt. On second failure mark `eval_error`. |
| Malformed scenarios file | Refuse to start. |

## Choosing a transport

| You have... | Use |
|---|---|
| LangChain / CrewAI / LangGraph / OpenAI Agents SDK / Anthropic Skills / notebook code | In-process (`--agent module:fn`) |
| FastAPI / Flask / Express / production HTTP endpoint | HTTP (`--agent-url ...`) |
| Both (e.g., dev locally, deploy as HTTP) | In-process for `pytest`, HTTP for CI/staging smoke |

The in-process transport is faster, supports `node_assertions`, and
needs no LLM key for deterministic scenarios. Use it whenever you can.

## Related

- [Getting started](getting-started.md) — install + quickstart for both
  transports.
- [Scenario format](scenarios.md) — what scenarios look like.
- [CLI reference](cli.md) — `--agent` vs `--agent-url`, all flags.
