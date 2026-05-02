from __future__ import annotations

from types import SimpleNamespace

from trainforge.llm.openai_compatible_client import OpenAICompatibleClient


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.last_kwargs = kwargs
        msg = SimpleNamespace(content="  ok  ")
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])


class _FakeOpenAI:
    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_openai_compatible_client_uses_openai_ctor(monkeypatch) -> None:
    holder: dict[str, _FakeOpenAI] = {}

    def _factory_module(name: str):
        assert name == "openai"

        class _Mod:
            @staticmethod
            def OpenAI(**kwargs):  # type: ignore[no-untyped-def]
                inst = _FakeOpenAI(**kwargs)
                holder["client"] = inst
                return inst

        return _Mod()

    monkeypatch.setattr(
        "trainforge.llm.openai_compatible_client.import_module",
        _factory_module,
    )

    client = OpenAICompatibleClient(
        api_key="k",
        base_url="https://example.test/v1",
        model="m",
        timeout_seconds=12.0,
    )

    assert holder["client"].kwargs == {
        "base_url": "https://example.test/v1",
        "api_key": "k",
        "timeout": 12.0,
    }

    text = client.complete("sys", "usr")
    assert text == "ok"
    assert holder["client"].chat.completions.last_kwargs["model"] == "m"


def test_openai_compatible_client_passes_extra_body(monkeypatch) -> None:
    fake = _FakeOpenAI(base_url="x", api_key="x", timeout=1)

    class _Mod:
        OpenAI = lambda **kwargs: fake  # noqa: E731

    monkeypatch.setattr(
        "trainforge.llm.openai_compatible_client.import_module",
        lambda name: _Mod,
    )

    client = OpenAICompatibleClient(api_key="k", extra_body={"flag": True})
    client.complete("sys", "usr")
    assert fake.chat.completions.last_kwargs["extra_body"] == {"flag": True}

