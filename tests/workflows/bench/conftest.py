"""评测与基准 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random


import pytest

from workflows.bench import spec as sp

#: 默认 case：成绩表 `k3s0` / 列序 `l0` / 第一套句面。
CASE_DEFAULTS: dict[str, JsonValue] = {"table": "k3s0", "layout": "l0", "phrasing": 0}

#: 小规格用的成绩表：**六个 k 各一张**（起点都取 0）。性质是构造出来的（见 `test_spec.py`）——
#: 六张表 × 六种列序 = 36 格上，五族仍然精确均匀：逐题答对 90（占 180 的一半）、答对条数
#: 六档各 6 次、达标 18、最好三档各 12、差距两档各 12。所以小规格跑一遍也能验五族的均衡。
TINY_TABLES: tuple[str, ...] = tuple(f"k{k}s0" for k in range(sp.TOTAL + 1))


def _make_case(**overrides: object) -> sp.BenchCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.BenchCase(**fields)


def _make_spec(**overrides: object) -> sp.BenchSpec:
    """输入：case 覆盖项。输出：真值规格。预期行为：同种子必得同一份规格。"""
    return sp.sample_spec(_make_case(**overrides), "train", 0, 0, random.Random(3))


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.BenchConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：`layouts` / `phrasings` 必须全开——少一种列序会让「每个候选在每个列位上各出现
    两次」这条断掉，逐列与 `best` 的分布就不再均匀；`test_ood` 那一档要带上留出取值
    （列序 `l5`、第二套句面），否则筛出来是空的。
    """
    if split == "test_ood" or ood_only:
        return sp.BenchConfig(
            split=split,
            tables=TINY_TABLES,
            layouts=sp.LAYOUTS,
            phrasings=(0, 1),
            ood_only=True,
        )
    return sp.BenchConfig(
        split=split,
        tables=TINY_TABLES,
        layouts=tuple(sorted(sp.SEEN_LAYOUTS)),
        phrasings=tuple(sorted(sp.SEEN_PHRASINGS)),
    )


@pytest.fixture
def make_case() -> object:
    """工厂夹具：`make_case(layout="l3")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> object:
    """工厂夹具：`make_spec(table="k0s1")` 造一份真值规格。"""
    return _make_spec
