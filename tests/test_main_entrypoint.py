from __future__ import annotations

import runpy


def test_python_module_entrypoint_calls_cli(monkeypatch) -> None:
    called = {"ok": False}

    def _fake_cli() -> None:
        called["ok"] = True

    monkeypatch.setattr("trainforge.cli.cli", _fake_cli)
    runpy.run_module("trainforge.__main__", run_name="__main__")
    assert called["ok"] is True

