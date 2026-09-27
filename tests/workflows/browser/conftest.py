"""界面操作 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.browser import spec as sp

#: 默认 case：填写任务 / 普通对象 / 没做完 / 干扰槽置灰 / 排布 p0 / 第一套句面。
#: 挑「没做完」是因为它让四个题族**全都出得来**。
CASE_DEFAULTS: dict[str, Any] = {
    "task": "fill",
    "form": "plain",
    "done": False,
    "decoy_state": "disabled",
    "order": "p0",
    "phrasing": 0,
}


def _make_case(**overrides: Any) -> sp.ScreenCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.ScreenCase(**fields)


def _make_spec(**overrides: Any) -> sp.ScreenSpec:
    """输入：case 覆盖项。输出：真值规格。预期行为：同种子必得同一份规格。"""
    return sp.sample_spec(_make_case(**overrides), "train", 0, 0, random.Random(3))


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.ScreenConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：`orders` 与 `dones` 必须全开——少一种排布会让 `target` 的位置分布不再是
    均匀的拉丁方，少一档完成档会让 `done` 那一族塌成单边（均衡断言当场红）。
    `test_ood` 那一档要带上留出取值（任务意图 `switch`、第二套句面），否则筛出来是空的。
    """
    if split == "test_ood" or ood_only:
        return sp.ScreenConfig(
            split=split,
            tasks=("fill", "switch"),
            decoy_states=("disabled", "current"),
            phrasings=(0, 1),
            ood_only=True,
        )
    return sp.ScreenConfig(
        split=split,
        tasks=("fill", "open"),
        decoy_states=("disabled", "current"),
        orders=sp.ORDERS,
        phrasings=(0,),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(done=True)` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(order="p3")` 造一条真值规格。"""
    return _make_spec
