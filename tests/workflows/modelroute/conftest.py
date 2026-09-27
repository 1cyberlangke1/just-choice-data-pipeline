"""模型路由 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

from typing import Any

import pytest

from workflows.modelroute import spec as sp

#: 默认 case：改缺陷任务、复杂度高、延迟紧、成本敏感、首次请求、私有数据、第一套句面。
CASE_DEFAULTS: dict[str, Any] = {
    "task": "codefix",
    "complexity": "high",
    "latency": "strict",
    "cost": "tight",
    "retry": "first",
    "data": "private",
    "phrasing": 0,
}


def _make_case(**overrides: Any) -> sp.RouteCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.RouteCase(**fields)


def _make_spec(**overrides: Any) -> sp.RouteSpec:
    """输入：case 覆盖项。输出：真值规格。预期行为：同 case 必得同一份规格（不掷骰子）。"""
    return sp.sample_spec(_make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.RouteConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：七张表**全开**——少一个任务会让 `tier` 少一个档，少一档复杂度或延迟会让
    `effort` 的三档分布塌掉（均衡断言当场红）。
    """
    return sp.RouteConfig(split=split, ood_only=ood_only)


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(task="reason")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(latency="relaxed")` 造一条真值规格。"""
    return _make_spec
