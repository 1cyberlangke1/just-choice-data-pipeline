"""片段筛选与去冗余 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random

import pytest

from workflows.clipcut import spec as sp
from workflows.clipcut.features import ClipCase


def _make_case(topics: int = 0, flags: int = 0) -> ClipCase:
    """输入：主题序列下标、高亮序列下标。输出：一个 case。"""
    return ClipCase(topics=topics, flags=flags)


def _make_spec(seed: int = 5, topics: int = 0, flags: int = 0) -> sp.ClipSpec:
    """输入：种子与两个维度下标。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(topics, flags), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.ClipConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：问法只留两种，但**两个维度全开**（四条同分布主题序列 × 八条高亮序列）——
    它们是四个题族答案分布的唯一来源，砍掉一半会让 `first_kept` / `dup_any` 退化成常量
    （均衡闸会当场标它），那样测的就不是「数据均衡」而是「测试配置很差」。
    """
    return sp.ClipConfig(split=split, phrasings=(0, 1))


@pytest.fixture
def make_case() -> object:
    """工厂夹具：`make_case(topics=2, flags=3)` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> object:
    """工厂夹具：`make_spec(7, flags=2)` 造一条真值规格。"""
    return _make_spec
