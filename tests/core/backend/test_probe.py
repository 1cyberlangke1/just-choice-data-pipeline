"""能力探测测试：四项独立、失败不连坐、不发 max_tokens（客户端 mock）。"""

from __future__ import annotations

from typing import Any

import pytest

from jc.backend import probe as probe_module
from jc.backend.probe import probe_capabilities


class FakeCompletions:
    """假的 chat.completions：按参数决定成功还是抛错。"""

    def __init__(self, broken: set[str], box: dict[str, Any]):
        self.broken = broken
        self.box = box

    def create(self, **kwargs: Any) -> Any:
        """输入：请求参数。输出：占位响应或抛错。预期行为：记下参数供断言。"""
        self.box["calls"].append(kwargs)
        label = _label(kwargs)
        if label in self.broken:
            raise RuntimeError(f"{label} 不支持")
        return object()


def _label(kwargs: dict[str, Any]) -> str:
    """输入：请求参数。输出：这一项探针的名字。"""
    if "tools" in kwargs:
        return "tools"
    fmt = kwargs.get("response_format")
    if isinstance(fmt, dict):
        return str(fmt.get("type"))
    return "plain"


@pytest.fixture
def patch_openai(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """把 probe 里的 openai.OpenAI 换成假客户端。"""
    box: dict[str, Any] = {"calls": [], "broken": set(), "init": {}}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            box["init"] = kwargs
            self.chat = type("Chat", (), {})()
            self.chat.completions = FakeCompletions(box["broken"], box)

    monkeypatch.setattr(probe_module.openai, "OpenAI", FakeClient)
    return box


def test_all_four_probes_pass(patch_openai: dict[str, Any]) -> None:
    caps = probe_capabilities(base_url="u", api_key="k", model="m")
    assert (caps.plain, caps.json_object, caps.json_schema, caps.tools) == (True, True, True, True)
    assert caps.starting_mode() == "json_schema"
    assert len(caps.notes) == 4


def test_probe_failures_are_independent(patch_openai: dict[str, Any]) -> None:
    patch_openai["broken"] = {"json_schema"}
    caps = probe_capabilities(base_url="u", api_key="k", model="m")
    assert caps.json_schema is False
    assert caps.tools is True
    assert caps.starting_mode() == "tools"


def test_probe_never_sends_max_tokens(patch_openai: dict[str, Any]) -> None:
    probe_capabilities(base_url="u", api_key="k", model="m")
    for call in patch_openai["calls"]:
        assert "max_tokens" not in call


def test_probe_disables_sdk_retries(patch_openai: dict[str, Any]) -> None:
    """探测要快要干脆：SDK 层的重试必须关掉，不然一个坏端点要等好几轮。"""
    probe_capabilities(base_url="u", api_key="k", model="m")
    assert patch_openai["init"]["max_retries"] == 0


def test_probe_notes_never_contain_the_key(patch_openai: dict[str, Any]) -> None:
    patch_openai["broken"] = {"plain", "tools"}
    caps = probe_capabilities(base_url="u", api_key="nvapi-topsecret", model="m")
    assert "nvapi-topsecret" not in " ".join(caps.notes)
