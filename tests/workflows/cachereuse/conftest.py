"""缓存复用判定 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random

import pytest

from workflows.cachereuse import spec as sp

#: 默认维度：5 步、约一半、只差动作、逐字相同、网页版。
CASE_DEFAULTS: dict[str, JsonValue] = {
    "length": 5,
    "band": "half",
    "diff_kind": "action",
    "spelling": sp.VERBATIM,
    "flavor": "web",
}


def _make_case(**overrides: object) -> sp.ReuseCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.ReuseCase(**fields)


def _make_spec(seed: int = 5, **overrides: object) -> sp.ReuseSpec:
    """输入：种子与 case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.ReuseConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：步数只留两档、写法只留一种，但**四个档位与两种分叉原因全开**——它们是四个题族
    答案分布的来源，砍掉一半会让 `reuse_band` / `diverge_at` 退化成常量（均衡闸会当场标它），
    那样测的就不是「数据均衡」而是「测试配置很差」。
    """
    return sp.ReuseConfig(
        split=split,
        lengths=(4, 5),
        bands=sp.BANDS,
        diff_kinds=sp.DIFF_KINDS,
        spellings=(sp.VERBATIM,),
        flavors=sp.ID_FLAVORS,
        phrasings=(0, 1),
    )


@pytest.fixture
def make_case() -> object:
    """工厂夹具：`make_case(band="none")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> object:
    """工厂夹具：`make_spec(7, spelling="synonym")` 造一条真值规格。"""
    return _make_spec
