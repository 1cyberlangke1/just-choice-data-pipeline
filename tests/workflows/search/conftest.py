"""检索 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.search import spec as sp

#: 默认维度：退款主题、最佳结果是「完整命中」、被点名那两条互相矛盾、最佳在第 3 位。
CASE_DEFAULTS: dict[str, Any] = {
    "topic": "refund",
    "best_flavor": "full",
    "mix": sp.MIXES[1],
    "pair": (1, 2),
    "best_at": 3,
    "pair_kind": "conflict",
    "probe": 1,
    "query_style": 0,
}


def _make_case(**overrides: Any) -> sp.SearchCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.SearchCase(**fields)


def _make_spec(seed: int = 5, **overrides: Any) -> sp.SearchSpec:
    """输入：种子与 case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.SearchConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：两个主题覆盖「见过 / 没见过」，两档最佳结果覆盖「能直接回答 / 要换数据源」，
    配方只留「全是相关度 ≤ 2」的那条（它同时对两档最佳结果都成立）。
    """
    return sp.SearchConfig(
        split=split,
        topics=("refund", "webhook"),
        best_flavors=("full", "stale"),
        mixes=(sp.MIXES[1],),
        layouts=(((1, 2), 3),),
        probes=(1, 2, 3),
        pair_kinds=("conflict", "none"),
        query_styles=(0,),
        phrasings=(0, 1),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(pair_kind="both", best_flavor="empty")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(7, topic="webhook", best_at=4)` 造一条真值规格。"""
    return _make_spec
