"""记录匹配 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.match import spec as sp

#: 默认维度：拉格、五栏全不一致、写法对 0-1、正序、不写额外栏。
CASE_DEFAULTS: dict[str, Any] = {
    "product": "lager",
    "pattern": sp.PATTERNS[0],
    "spelling": sp.SPELLINGS[0],
    "layout": "plain",
    "extra": "none",
}


def _make_case(**overrides: Any) -> sp.MatchCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.MatchCase(**fields)


def _make_spec(seed: int = 5, **overrides: Any) -> sp.MatchSpec:
    """输入：种子与 case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.MatchConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：商品与写法对各留两档，但**十二种字段关系全开**——关系表是八个题族答案
    分布的来源，砍掉一半会让 `band` / `variant_count` 退化成常量（均衡闸会当场标它），
    那样测的就不是「数据均衡」而是「测试配置很差」。
    """
    return sp.MatchConfig(
        split=split,
        products=sp.PRODUCTS[:2],
        patterns=sp.PATTERNS,
        spellings=sp.SPELLINGS[:2],
        layouts=("plain", "reversed"),
        extras=sp.EXTRAS,
        phrasings=(0, 1),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(layout="reversed")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(7, extra="origin")` 造一条真值规格。"""
    return _make_spec
