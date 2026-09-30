"""行情读数判读 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random

import pytest

from workflows.quoteread import spec as sp
from workflows.quoteread.features import QuoteCase


def _make_case(values: int = 0, units: int = 0, names: int = 0) -> QuoteCase:
    """输入：读数模式下标、单位模式下标、标的名集下标。输出：一个 case。"""
    return QuoteCase(values=values, units=units, names=names)


def _make_spec(seed: int = 5, values: int = 0, units: int = 0, names: int = 0) -> sp.QuoteSpec:
    """输入：种子与三个维度下标。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(values, units, names), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.QuoteConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：问法只留两种，但**三个维度全开**（八条读数模式 × 四条单位模式 × 同分布名集）
    ——它们是四个题族答案分布的唯一来源，砍掉一半会让 `tier` / `alert` 退化成常量
    （均衡闸会当场标它），那样测的就不是「数据均衡」而是「测试配置很差」。
    """
    return sp.QuoteConfig(split=split, phrasings=(0, 1))


@pytest.fixture
def make_case() -> object:
    """工厂夹具：`make_case(values=2, units=1)` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> object:
    """工厂夹具：`make_spec(7, values=2)` 造一条真值规格。"""
    return _make_spec
