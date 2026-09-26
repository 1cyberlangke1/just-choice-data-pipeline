"""引文核查 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.citation import spec as sp

#: 默认维度：鉴权主题、「通过」结论、引文照抄原文、三节、无干扰、第二种措辞、
#: 声称出处落在中间那一节、编号写成 `[1]`。
CASE_DEFAULTS: dict[str, Any] = {
    "topic": "auth",
    "status": "verified",
    "quote_form": "verbatim",
    "sections": 3,
    "noise": "plain",
    "claim_style": 1,
    "where": "middle",
    "doc_style": "bracketed",
}


def _make_case(**overrides: Any) -> sp.CitationCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.CitationCase(**fields)


def _make_spec(seed: int = 5, **overrides: Any) -> sp.CitationSpec:
    """输入：种子与 case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（构造器不掷骰子，种子只是为了签名一致）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.CitationConfig:
    """输入：切分名。输出：采样参数（零 token，跑得动整条管线）。

    预期行为：四个主题、五个结论档、三种引文写法都在，小节数与编号写法给两档；
    干扰只留 `plain`——`decoy` / `off_topic` 会挤掉「两节」那一整格，覆盖面反而更窄。
    """
    return sp.CitationConfig(
        split=split,
        topics=sp.TOPIC_NAMES,
        statuses=sp.STATUSES,
        quote_forms=sp.QUOTE_FORMS,
        section_counts=(2, 3),
        noises=("plain",),
        claim_styles=(0, 1),
        wheres=sp.WHERES,
        doc_styles=sp.DOC_STYLES,
        phrasings=(0, 1),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(status="mislocated", sections=3)` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(7, status="fabricated", quote_form="reworded")` 造一条真值规格。"""
    return _make_spec
