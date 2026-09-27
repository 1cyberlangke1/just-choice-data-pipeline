"""改动审查 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.review import spec as sp

#: 默认 case：文件搭配 t0 / 噪声搭配 n0 / 变更搭配 c0 / 排布 o0 / 第一套句面。
CASE_DEFAULTS: dict[str, Any] = {
    "triple": "t0",
    "noise": "n0",
    "change": "c0",
    "order": "o0",
    "phrasing": 0,
}


def _make_case(**overrides: Any) -> sp.ChangeCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.ChangeCase(**fields)


def _make_spec(**overrides: Any) -> sp.ChangeSpec:
    """输入：case 覆盖项。输出：真值规格。预期行为：同种子必得同一份规格。"""
    return sp.sample_spec(_make_case(**overrides), "train", 0, 0, random.Random(3))


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.ReviewConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：`orders` 与 `changes` 必须全开——少一种排布会让 `top` 的位置分布不再是
    均匀的拉丁方，少一套变更搭配会让 `gate` 那一族的三档不再是各三分之一
    （均衡断言当场红）。`test_ood` 那一档要带上留出取值（噪声搭配 `n2`、第二套句面），
    否则筛出来是空的。
    """
    if split == "test_ood" or ood_only:
        return sp.ReviewConfig(
            split=split,
            triples=("t0", "t1"),
            noises=sp.NOISE_KEYS,
            orders=sp.ORDERS,
            phrasings=(0, 1),
            ood_only=True,
        )
    return sp.ReviewConfig(
        split=split,
        noises=tuple(sorted(sp.SEEN_NOISES)),
        orders=sp.ORDERS,
        phrasings=tuple(sorted(sp.SEEN_PHRASINGS)),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(noise="n2")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(order="o3")` 造一条真值规格。"""
    return _make_spec
