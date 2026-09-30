"""多批次排名聚合 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random

import pytest

from workflows.siftrank import spec as sp

#: 默认维度：缓坡、恒等映射、第一条排序提示。
CASE_DEFAULTS: dict[str, JsonValue] = {
    "shape": "stair",
    "perm": "identity",
    "flavor": "launch",
}


def _make_case(**overrides: object) -> sp.RankCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.RankCase(**fields)


def _make_spec(seed: int = 5, **overrides: object) -> sp.RankSpec:
    """输入：种子与 case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.RankConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：映射只留两种、形状只留三个，但**三条答案轴全开**——`peak` / `twin` / `stair`
    分别给出 cut 1/2/3、relation a_first/tie/b_first、stable 真/假/真；砍掉一半会让某些族
    退化成常量（均衡闸会当场标它），那样测的就不是「数据均衡」而是「测试配置很差」。
    """
    return sp.RankConfig(
        split=split,
        shapes=("peak", "twin", "stair"),
        perms=("identity", "reverse"),
        flavors=sp.ID_FLAVORS,
        phrasings=(0, 1),
    )


@pytest.fixture
def make_case() -> object:
    """工厂夹具：`make_case(shape="twin")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> object:
    """工厂夹具：`make_spec(7, perm="reverse")` 造一条真值规格。"""
    return _make_spec
