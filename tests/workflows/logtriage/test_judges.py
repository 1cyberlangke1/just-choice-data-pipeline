"""日志行分诊的判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations

import json

from jc.backend.types import Draft

from tests.workflows._fakes import draft as _draft
from tests.workflows.logtriage.conftest import (
    BASE,
    FRAGMENTS,
    FakeBackend,
    faithful,
    quote_for,
    sample_of,
    spec_for,
    trace_report,
)
from workflows.logtriage import features, judges, locales, render
from workflows.logtriage.features import LogTriageCase


#: 一个普通 case：命中两条。
CASE = LogTriageCase("N1+N3", features.SOURCE_APP)


def _drop_tag_line(state: str, tag: str) -> str:
    """输入：state、判据编号。输出：抹掉判据表里那一行之后的 state。"""
    prefix = f"{render.LINE_PREFIX}{tag}"
    lines = [line for line in state.splitlines() if not line.strip().startswith(prefix)]
    return "\n".join(lines)


def test_hygiene_rejects_a_missing_hit() -> None:
    """输入：计划要 N1+N2、日志里只写了 N1 的样本。输出：无。

    预期行为：不通过，理由是命中组合对不上——少写一条和多重一条一样，都是标签兑现不了。
    """
    case = LogTriageCase("N1+N2", features.SOURCE_APP)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(BASE + FRAGMENTS["N1"]), spec_for(case)
    )
    assert not verdict.ok and not verdict.skipped
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_an_extra_hit() -> None:
    """输入：日志多带了一条判据的样本。输出：无。预期行为：不通过，理由是命中组合对不上。"""
    case = LogTriageCase("N1", features.SOURCE_APP)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(BASE + FRAGMENTS["N1"] + FRAGMENTS["N4"]), spec_for(case)
    )
    assert not verdict.ok
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_a_pasted_rule() -> None:
    """输入：日志里整段贴了判据原文。输出：无。

    预期行为：不通过，理由是贴判据原文——那是「抄规格」的捷径，不是「读日志」（范式 §9.45）。
    """
    loc = locales.load("zh")
    rule = str(render.tag_copy(loc, "N1")["rule"])
    case = LogTriageCase("N1", features.SOURCE_APP)
    verdict = judges.HygieneJudge("zh").judge(sample_of(f"{BASE} {rule}"), spec_for(case))
    assert not verdict.ok and not verdict.skipped
    assert "判据原文" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_does_not_match_the_criterion() -> None:
    """输入：多报一条 N1、引文却是 N3 那一段的反解。输出：无。

    预期行为：这条误报被丢掉，判定仍然通过——「抄出来的那段按判据根本不算命中」必须由代码把关。
    """
    case = LogTriageCase("N3+N4", features.SOURCE_APP)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("N3"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["N3", "N4"]}
    assert "另有 1 条" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_is_fabricated() -> None:
    """输入：多报一条 N1、引文是日志里根本没有的一段探测词。输出：无。

    预期行为：引文抄不出原文，同样按误报丢掉，判定通过。
    """
    case = LogTriageCase("N3+N4", features.SOURCE_APP)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("N1"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["N3", "N4"]}


class _OverclaimBackend:
    """真话之外**多报一条 N1** 的假后端。

    字段：`report` 真话（由 `trace_report` 推出来）、`quote` 给那条多报的 N1 配的引文——
    分别验「引文在日志里但按判据不算命中」与「引文在日志里根本不存在」两条路径。
    """

    def __init__(self, case: LogTriageCase, quote: str) -> None:
        """输入：注入特征、误报的引文。输出：实例。"""
        self.report = trace_report(case)
        self.quote = quote

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：真话 + 一条 N1 误报。预期行为：不联网。"""
        del task
        tags = [*self.report, {"tag": "N1", "quote": self.quote}]
        return _draft({"tags": tags})
