"""有界任务推进 workflow 测试的公共夹具：小规格构造器（零 token、不落盘）。"""

from __future__ import annotations

import random

import pytest

from workflows.planstep import spec as sp
from workflows.planstep.features import PlanCase


def _make_case(graph: int = 0, done: int = 0) -> PlanCase:
    """输入：图形状下标、已完成项数。输出：一个 case。"""
    return PlanCase(graph=graph, done=done)


def _make_spec(graph: int = 0, done: int = 0, phrasing: int = 0) -> sp.PlanSpec:
    """输入：图形状、已完成项数、问法。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(0), _make_case(graph, done), "train", 0, phrasing)


def tiny_config(split: str = "train") -> sp.PlanConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：问法只留两种，但**两个维度全开**（10 张同分布图 × 6 个已完成项数）——它们是
    四个题族答案分布的唯一来源，砍掉一半会让某些档位取不到（均衡闸会当场标它），那样测的
    就不是「数据均衡」而是「测试配置很差」。
    """
    return sp.PlanConfig(split=split, phrasings=(0, 1))


@pytest.fixture
def make_case() -> object:
    """工厂夹具：`make_case(graph=3, done=2)` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> object:
    """工厂夹具：`make_spec(3, done=2)` 造一条真值规格。"""
    return _make_spec
