"""界面元素定位 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random


import pytest

from workflows.elementpick import spec as sp

#: 默认 case：动作标签 `save` / 排布 `l0` / 有落点 / 没有多余同名行 / 没有近似元素 / 第一套句面。
CASE_DEFAULTS: dict[str, JsonValue] = {
    "label": "save",
    "layout": "l0",
    "shape": "hit",
    "extras": 0,
    "shadow": False,
    "phrasing": 0,
}

#: 小规格用的动作标签：三个就够（`target` / `role` 的分布由**位置**决定，与标签数无关），
#: 但三个标签之间不能互为子串，否则「完全一致」那条判据会被无关文本撞出假命中。
TINY_LABELS: tuple[str, ...] = sp.LABELS[:3]


def _make_case(**overrides: object) -> sp.ElementCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.ElementCase(**fields)


def _make_spec(**overrides: object) -> sp.ElementSpec:
    """输入：case 覆盖项。输出：真值规格。预期行为：同种子必得同一份规格。"""
    return sp.sample_spec(_make_case(**overrides), "train", 0, 0, random.Random(3))


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.ElementConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：`layouts` / `shapes` / `extras` / `shadows` 必须全开——少一种排布会让
    `target` 的位置分布与 `role` 的三档都不再均匀；少一种形状会让 `found` 的 50/50 塌掉；
    少一种条数会让 `decoy` 的三档少一格；少一种近似会让 `shadow` 的 50/50 塌掉。
    `test_ood` 那一档要带上留出取值（排布 `l5`、第二套句面），否则筛出来是空的。
    """
    if split == "test_ood" or ood_only:
        return sp.ElementConfig(split=split, labels=TINY_LABELS, ood_only=True)
    return sp.ElementConfig(
        split=split,
        labels=TINY_LABELS,
        layouts=tuple(sorted(sp.SEEN_LAYOUTS)),
        phrasings=tuple(sorted(sp.SEEN_PHRASINGS)),
    )


@pytest.fixture
def make_case() -> object:
    """工厂夹具：`make_case(shape="miss")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> object:
    """工厂夹具：`make_spec(label="delete")` 造一条真值规格。"""
    return _make_spec
