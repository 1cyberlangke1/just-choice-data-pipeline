"""函数调用 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations
from collections.abc import Callable
from jc.jsontypes import JsonValue



import pytest

from workflows.functions import spec as sp

#: 默认 case：画价格走势、语气中等、时间范围一周、写代码、第一套句面。
CASE_DEFAULTS: dict[str, JsonValue] = {
    "tool": "plot_price",
    "voice": "chatty",
    "window": "1w",
    "alias": "ticker",
    "phrasing": 0,
}


def _make_case(**overrides: object) -> sp.CallCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.CallCase(**fields)


def _make_spec(**overrides: object) -> sp.CallSpec:
    """输入：case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（计划由素材指纹派生，不掷骰子）。
    """
    return sp.sample_spec(_make_case(**overrides), "train", 0)


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.CallConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：四个工具都开着（每个工具各有自己一组题族），两种语气都开着
    （`arg_*` 题只在「提到该参数」的语气下出，缺一种语气就会让某族答案退化成常量），
    两个窗口、两种写法、两套句面都开着。
    """
    return sp.CallConfig(
        split=split,
        tools=sp.TOOL_NAMES,
        voices=sp.VOICES,
        windows=("1d", "1w"),
        aliases=sp.ALIASES,
        phrasings=sp.PHRASINGS,
        ood_only=ood_only,
    )


@pytest.fixture
def make_case() -> Callable[..., sp.CallCase]:
    """工厂夹具：`make_case(tool="top_movers", voice="terse")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Callable[..., sp.CallSpec]:
    """工厂夹具：`make_spec(tool="compare_returns")` 造一条真值规格。"""
    return _make_spec
