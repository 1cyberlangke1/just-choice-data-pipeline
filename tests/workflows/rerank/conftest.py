"""检索重排 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.rerank import spec as sp

#: 默认维度：备份主题、操作步骤方面、画像 0、无噪声句、直述措辞。
CASE_DEFAULTS: dict[str, Any] = {
    "topic": "backup",
    "aspect": "steps",
    "shape": sp.SHAPES[0],
    "traps": (),
    "wording": "plain",
}


def _make_case(**overrides: Any) -> sp.RerankCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.RerankCase(**fields)


def _make_spec(seed: int = 5, **overrides: Any) -> sp.RerankSpec:
    """输入：种子与 case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.RerankConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：主题 / 方面 / 噪声句 / 措辞各两档，但**十个画像全开**——画像表是七个
    题族答案分布的来源，砍掉一半会让 `noul:on_topic` 这类题族退化成常量（均衡闸会当场
    标它），那样测的就不是「数据均衡」而是「测试配置很差」。
    """
    return sp.RerankConfig(
        split=split,
        topics=("backup", "rotation"),
        aspects=("steps", "config"),
        shapes=sp.SHAPES,
        traps=((), ("forwarded",)),
        wordings=("plain", "titled"),
        phrasings=(0, 1),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(aspect="config")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(7, wording="terse")` 造一条真值规格。"""
    return _make_spec
