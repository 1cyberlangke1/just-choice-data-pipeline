"""情感与情绪 workflow 测试的公共夹具：小规格构造器（零 token、不落盘）。"""

from __future__ import annotations

import random

import pytest

from workflows.sentiment import spec as sp
from workflows.sentiment.features import SentimentCase


def _make_case(arc: int = 0, topic: str = "support") -> SentimentCase:
    """输入：弧线下标、场景键。输出：一个 case。"""
    return SentimentCase(arc=arc, topic=topic)


def _make_spec(arc: int = 0, topic: str = "support", phrasing: int = 0) -> sp.SentimentSpec:
    """输入：弧线、场景、问法。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(0), _make_case(arc, topic), "train", 0, phrasing)


def tiny_config(split: str = "train") -> sp.SentimentConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：问法只留两种，但**两个维度全开**（15 条同分布弧线 × 5 个场景）——它们是四个
    题族答案分布的唯一来源，砍掉一半会让某些档位取不到（均衡闸会当场标它），那样测的就不是
    「数据均衡」而是「测试配置很差」。
    """
    return sp.SentimentConfig(split=split, phrasings=(0, 1))


@pytest.fixture
def make_case() -> object:
    """工厂夹具：`make_case(arc=3, topic="review")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> object:
    """工厂夹具：`make_spec(3, topic="review")` 造一条真值规格。"""
    return _make_spec
