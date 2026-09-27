"""游戏与模拟控制 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.sim import spec as sp

#: 默认 case：局面 `b0` / 排布 `l0` / 第一套句面。
CASE_DEFAULTS: dict[str, Any] = {"board": "b0", "layout": "l0", "phrasing": 0}

#: 小规格用的局面：**二十四张里的前六张**，性质是构造出来的（见 `test_spec.py`）——
#: 合法动作数合计 18（占 36 的一半）、结束 3 张、没结束的 3 张都有合法动作、
#: 该用的那个动作的耗落在三个不同档。所以小规格跑一遍也能验四族的均衡。
TINY_BOARDS: tuple[str, ...] = sp.BOARDS[:6]


def _make_case(**overrides: Any) -> sp.SimCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.SimCase(**fields)


def _make_spec(**overrides: Any) -> sp.SimSpec:
    """输入：case 覆盖项。输出：真值规格。预期行为：同种子必得同一份规格。"""
    return sp.sample_spec(_make_case(**overrides), "train", 0, 0, random.Random(3))


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.SimConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：`layouts` / `phrasings` 必须全开——少一种排布会让逐位置的合法率与 `best`
    的位置分布不再是均匀的拉丁方；`test_ood` 那一档要带上留出取值（排布 `l5`、第二套句面），
    否则筛出来是空的。
    """
    if split == "test_ood" or ood_only:
        return sp.SimConfig(
            split=split,
            boards=TINY_BOARDS,
            layouts=sp.LAYOUTS,
            phrasings=(0, 1),
            ood_only=True,
        )
    return sp.SimConfig(
        split=split,
        boards=TINY_BOARDS,
        layouts=tuple(sorted(sp.SEEN_LAYOUTS)),
        phrasings=tuple(sorted(sp.SEEN_PHRASINGS)),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(layout="l3")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(board="b6")` 造一份真值规格。"""
    return _make_spec
