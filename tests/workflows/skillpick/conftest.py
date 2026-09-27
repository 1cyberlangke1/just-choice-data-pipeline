"""技能选择 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.skillpick import spec as sp

#: 默认 case：请求标签 `read_code` / 排布 `l0` / 授权 `read` / 占用配对 `above` / 第一套句面。
CASE_DEFAULTS: dict[str, Any] = {
    "tag": "read_code",
    "layout": "l0",
    "grant": "read",
    "inuse": "above",
    "phrasing": 0,
}

#: 小规格用的标签：三个技能标签（**三个范围各一个**）+ 三个干扰标签。
#: 少一个范围会让 `scope` 那一族的三档不再是各三分之一；技能与干扰不配对会让 `fits`
#: 的 50/50 塌掉——两条都会让均衡断言当场红。
TINY_TAGS: tuple[str, ...] = (
    "read_code",
    "edit_code",
    "ship_release",
    "read_code_notes",
    "edit_code_style",
    "ship_release_note",
)


def _make_case(**overrides: Any) -> sp.SkillCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.SkillCase(**fields)


def _make_spec(**overrides: Any) -> sp.SkillSpec:
    """输入：case 覆盖项。输出：真值规格。预期行为：同种子必得同一份规格。"""
    return sp.sample_spec(_make_case(**overrides), "train", 0, 0, random.Random(3))


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.SkillConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：`layouts` / `grants` / `inuse_modes` 必须全开——少一种排布会让 `pick` 的
    位置分布不再是均匀的拉丁方；`test_ood` 那一档要带上留出取值（排布 `l5`、第二套句面），
    否则筛出来是空的。
    """
    if split == "test_ood" or ood_only:
        return sp.SkillConfig(
            split=split,
            tags=TINY_TAGS,
            layouts=sp.LAYOUTS,
            grants=sp.GRANTS,
            inuse_modes=sp.INUSE_KINDS,
            phrasings=(0, 1),
            ood_only=True,
        )
    return sp.SkillConfig(
        split=split,
        tags=TINY_TAGS,
        layouts=tuple(sorted(sp.SEEN_LAYOUTS)),
        phrasings=tuple(sorted(sp.SEEN_PHRASINGS)),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(layout="l3")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(tag="deploy")` 造一条真值规格。"""
    return _make_spec
