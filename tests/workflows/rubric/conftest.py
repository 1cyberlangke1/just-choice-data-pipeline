"""打分与评分 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.rubric import spec as sp

#: 默认 case：判据表 `s0` / 排布 `l0` / 松紧 `tight` / 第一套句面。
CASE_DEFAULTS: dict[str, Any] = {
    "sheet": "s0",
    "layout": "l0",
    "slack": "tight",
    "phrasing": 0,
}

#: 小规格用的判据表：**六张**，每张恰好三条达标，六条判据各达标三次。
#: 为什么是六张而不是随便几张：三档各两张、三个处置各两张、每条判据 3/6 达标——三条都
#: 是均衡断言的前提，少一张就会让某一族的分布当场偏掉（均衡闸标红）。其中 `s2` 与 `s15`
#: 的最严重那一条**并列**（`top` 一族对它们一条都不出），正好把两阶段依赖也带上。
TINY_SHEETS: tuple[str, ...] = ("s0", "s2", "s10", "s11", "s15", "s16")


def _make_case(**overrides: Any) -> sp.RubricCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.RubricCase(**fields)


def _make_spec(**overrides: Any) -> sp.RubricSpec:
    """输入：case 覆盖项。输出：真值规格。预期行为：同种子必得同一份规格。"""
    return sp.sample_spec(_make_case(**overrides), "train", 0, 0, random.Random(3))


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.RubricConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：`layouts` / `slacks` 必须全开——少一种排布会让 `top` 的位置分布不再是均匀的
    拉丁方；`test_ood` 那一档要带上留出取值（排布 `l5`、第二套句面），否则筛出来是空的。
    """
    if split == "test_ood" or ood_only:
        return sp.RubricConfig(
            split=split,
            sheets=TINY_SHEETS,
            layouts=sp.LAYOUTS,
            slacks=sp.SLACKS,
            phrasings=(0, 1),
            ood_only=True,
        )
    return sp.RubricConfig(
        split=split,
        sheets=TINY_SHEETS,
        layouts=tuple(sorted(sp.SEEN_LAYOUTS)),
        phrasings=tuple(sorted(sp.SEEN_PHRASINGS)),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(layout="l3")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(sheet="s6")` 造一份真值规格。"""
    return _make_spec

