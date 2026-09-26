"""工单分诊 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.routing import spec as sp

#: 默认维度：支付域、故障、专业版、全部用户受阻、无合规标记、等得久。
CASE_DEFAULTS: dict[str, Any] = {
    "area": "payments",
    "issue": "outage",
    "tier": "pro",
    "impact": "all_users",
    "urgency": "blocked",
    "compliance": "none",
    "wait": "long",
}


def _make_case(**overrides: Any) -> sp.TicketCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.TicketCase(**fields)


def _make_spec(seed: int = 5, **overrides: Any) -> sp.TicketSpec:
    """输入：种子与 case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.RoutingConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：两个产品域覆盖「关键 / 非关键」，两个问题类型覆盖「队列见过 / 没见过」，
    两个合规标记覆盖「不升级 / 升级」，其余各留两档。
    """
    return sp.RoutingConfig(
        split=split,
        areas=("payments", "storage"),
        issues=("outage", "bug"),
        tiers=("free", "enterprise"),
        impacts=("single_user", "all_users"),
        urgencies=("can_wait", "blocked"),
        compliances=("none", "security"),
        waits=("short", "long"),
        phrasings=(0, 1),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(compliance="gdpr", issue="feature")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(7, tier="enterprise", wait="short")` 造一条真值规格。"""
    return _make_spec

