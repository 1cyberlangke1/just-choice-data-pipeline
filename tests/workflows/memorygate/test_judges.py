"""记忆写入门控的判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations

import json

from jc.backend.types import Draft
from jc.jsontypes import JsonValue

from tests.workflows._fakes import draft as _draft
from tests.workflows.memorygate.conftest import (
    BASE,
    FRAGMENTS,
    FakeBackend,
    faithful,
    quote_for,
    sample_of,
    spec_for,
    trace_report,
)
from workflows.memorygate import features, judges, locales, render
from workflows.memorygate.features import MemoryGateCase


#: 一个普通 case：命中两条。
CASE = MemoryGateCase("M1+M3", features.FORM_SENTENCE)


def _drop_tag_line(state: str, loc: dict[str, JsonValue], tag: str) -> str:
    """输入：state、语言字典、判据编号。输出：抹掉判据表里那一行之后的 state。"""
    prefix = f"{render.LINE_PREFIX}{tag}"
    lines = [line for line in state.splitlines() if not line.strip().startswith(prefix)]
    return "\n".join(lines)


def test_hygiene_rejects_a_missing_hit() -> None:
    """输入：计划要 M1+M2、记忆里只写了 M1 的样本。输出：无。

    预期行为：不通过，理由是命中组合对不上——少写一条和多重一条一样，都是标签兑现不了。
    """
    case = MemoryGateCase("M1+M2", features.FORM_SENTENCE)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(BASE + FRAGMENTS["M1"]), spec_for(case)
    )
    assert not verdict.ok and not verdict.skipped
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_an_extra_hit() -> None:
    """输入：记忆多带了一条判据的样本。输出：无。预期行为：不通过，理由是命中组合对不上。"""
    case = MemoryGateCase("M1", features.FORM_SENTENCE)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(BASE + FRAGMENTS["M1"] + FRAGMENTS["M4"]), spec_for(case)
    )
    assert not verdict.ok
    assert "命中组合" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_does_not_match_the_criterion() -> None:
    """输入：多报一条 M1、引文却是 M3 那一段的反解。输出：无。

    预期行为：这条误报被丢掉，判定仍然通过——「抄出来的那段按判据根本不算命中」必须由代码把关。
    """
    case = MemoryGateCase("M3+M4", features.FORM_SENTENCE)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("M3"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["M3", "M4"]}
    assert "另有 1 条" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_is_fabricated() -> None:
    """输入：多报一条 M1、引文是记忆里根本没有的一句范围词。输出：无。

    预期行为：引文抄不出原文，同样按误报丢掉，判定通过。
    """
    case = MemoryGateCase("M3+M4", features.FORM_SENTENCE)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("M1"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["M3", "M4"]}


class _OverclaimBackend:
    """真话之外**多报一条 M1** 的假后端。

    字段：`report` 真话（由 `trace_report` 推出来）、`quote` 给那条多报的 M1 配的引文——
    分别验「引文在记忆里但按判据不算命中」与「引文在记忆里根本不存在」两条路径。
    """

    def __init__(self, case: MemoryGateCase, quote: str) -> None:
        """输入：注入特征、误报的引文。输出：实例。"""
        self.report = trace_report(case)
        self.quote = quote

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：真话 + 一条 M1 误报。预期行为：不联网。"""
        del task
        tags = [*self.report, {"tag": "M1", "quote": self.quote}]
        return _draft({"tags": tags})
