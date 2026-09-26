"""内容审核 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.guardrail import spec as sp

#: 默认维度：公开可见、模型输出、违反四条（含三条高危）、无擦边句、一般受众、无前科、直述。
CASE_DEFAULTS: dict[str, Any] = {
    "reach": "public",
    "direction": "outbound",
    "violations": ("meds", "scam", "privacy", "harass"),
    "mentions": (),
    "audience": "general",
    "history": "clean",
    "wording": "plain",
}


def _make_case(**overrides: Any) -> sp.GuardCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.GuardCase(**fields)


def _make_spec(seed: int = 5, **overrides: Any) -> sp.GuardSpec:
    """输入：种子与 case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.GuardConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：每维两档（条款子集三档：空 / 低危 / 高危+中危），问法两档。
    """
    return sp.GuardConfig(
        split=split,
        reaches=("group", "public"),
        directions=("inbound", "outbound"),
        violations=((), ("spam",), ("meds", "scam", "harass")),
        mentions=((), ("meds",)),
        audiences=("general", "minors"),
        histories=("clean", "repeat"),
        wordings=("plain", "quoted"),
        phrasings=(0, 1),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(audience="minors")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(7, history="repeat")` 造一条真值规格。"""
    return _make_spec
