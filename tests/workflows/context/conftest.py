"""会话上下文压缩 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.context import spec as sp

#: 默认 case：报错小块 / 无关小块 / 类型搭配 read_run / 重复角色 keep / 排布 p0 / 第一套句面。
#: 挑 `extra=keep` 是因为它同时让 `spot`（恰好一块被丢）与 `top`（占用最多的唯一）都出得来。
CASE_DEFAULTS: dict[str, Any] = {
    "keep_form": "err_small",
    "drop_form": "off_small",
    "kind_set": "read_run",
    "extra": "keep",
    "order": "p0",
    "phrasing": 0,
}


def _make_case(**overrides: Any) -> sp.SessionCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.SessionCase(**fields)


def _make_spec(**overrides: Any) -> sp.SessionSpec:
    """输入：case 覆盖项。输出：真值规格。预期行为：同种子必得同一份规格。"""
    return sp.sample_spec(_make_case(**overrides), "train", 0, 0, random.Random(3))


def spec_with_compact(compact: bool, **overrides: Any) -> sp.SessionSpec:
    """输入：目标标签（要不要压缩）、case 覆盖项。输出：标签命中的规格。

    预期行为：预算由 RNG 抽，所以这里换个种子把目标标签找出来；找不到就报错
    （说明预算池里落不下这个标签，那是构造问题，不该在测试里被悄悄跳过）。
    """
    case = _make_case(**overrides)
    for seed in range(64):
        spec = sp.sample_spec(case, "train", 0, 0, random.Random(seed))
        if spec.compact is compact:
            return spec
    raise AssertionError(f"64 个种子里都没有 compact={compact} 的规格")


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.SessionConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：`extras` 与 `orders` 必须全开——少一个重复角色会让 `action` 的四档分布塌掉，
    少一种排布会让 `spot` / `top` 的位置分布不再是均匀的拉丁方（均衡断言当场红）。
    """
    return sp.SessionConfig(
        split=split,
        keep_forms=("err_small",),
        drop_forms=("off_small", "stale"),
        kind_sets=("read_run", "ask_read"),
        extras=sp.ACTIONS,
        orders=sp.ORDERS,
        phrasings=(0,),
        ood_only=ood_only,
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(extra="drop")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(order="p3")` 造一条真值规格。"""
    return _make_spec
