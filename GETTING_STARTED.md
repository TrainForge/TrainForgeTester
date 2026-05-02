# Getting Started

This guide helps you run TrainForge locally in a few minutes.

## 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2) Run tests

```bash
pytest -q
```

## 3) Start the mock agent

```bash
trainforge mock-agent \
  --scenarios scenarios/example_restaurant_booking.json \
  --port 8080
```

## 4) Run a scenario

```bash
export OPENAI_API_URL=https://api.openai.com/v1
export OPENAI_API_KEY=your-key

trainforge run \
  --scenarios scenarios/example_restaurant_booking.json \
  --agent-url http://127.0.0.1:8080/chat \
  --output results.json
```

## 5) Build a report

```bash
trainforge report --results results.json --output report.html
```

## 6) Compare two runs

```bash
trainforge diff --before before.json --after after.json --output regression.html
```

## Troubleshooting

- `unsupported scenarios version`: set scenario `version` to `"2.0"`.
- Agent connection errors: verify `--agent-url` and that the server is running.
- Empty responses from agent: check your agent API contract against `README.md`.

