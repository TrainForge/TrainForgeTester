# Contributing to TrainForge

Thanks for improving TrainForge.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project principles

- Keep scenario and results wire formats backward-compatible unless version is intentionally bumped.
- Prefer deterministic checks over probabilistic checks where possible.
- Keep CLI behavior and exit codes stable.
- Avoid hidden side effects in runner/evaluation code paths.

## Making changes

1. Open a focused branch.
2. Keep changes small and reviewable.
3. Add or update tests for behavior changes.
4. Update docs when user-facing behavior changes.

## Quality gate before PR

```bash
pytest -q
```

If you have formatters/linters configured locally, run them too.

## Pull request checklist

- [ ] Problem and approach are clearly described.
- [ ] Public API/CLI/schema impact is documented.
- [ ] Tests are added/updated and pass locally.
- [ ] Backward compatibility is explicitly considered.

