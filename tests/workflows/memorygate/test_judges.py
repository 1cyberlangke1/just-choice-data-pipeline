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

#: 全空间 48 组 case。
CASES = features.build_cases(512)

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


def test_hygiene_rejects_a_state_without_a_content_block() -> None:
    """输入：一段没有记忆抬头的 state。输出：无。预期行为：不通过，理由是认不出记忆。"""
    spec, _ = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge({"state": "这里什么抬头都没有"}, spec)
    assert not verdict.ok
    assert "记忆" in verdict.reason


def test_frame_accepts_faithful_and_rejects_a_broken_table() -> None:
    """输入：忠实 state 与缺了一行的 state。输出：无。

    预期行为：判据表缺一条就报出来——那是构造 bug，标签会跟着一起错。
    """
    judge = judges.FrameJudge("zh")
    loc = locales.load("zh")
    spec, sample = faithful(CASE)
    assert judge.judge(sample, spec).ok
    broken = {"state": _drop_tag_line(str(sample["state"]), loc, "M2")}
    verdict = judge.judge(broken, spec)
    assert not verdict.ok
    assert "判据表" in verdict.reason


def test_trace_agrees_with_the_plan() -> None:
    """输入：诚实的反解。输出：无。预期行为：通过，raw 里留下反解出来的命中组合。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["M1", "M3"]}


def test_trace_catches_a_lie() -> None:
    """输入：反解漏报一条。输出：无。预期行为：不通过，理由里两条都列出来。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "M1" in verdict.reason and "M3" in verdict.reason


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


def test_quote_holds_needs_both_the_quote_and_the_criterion() -> None:
    """输入：几种引文。输出：无。

    预期行为：引文得原样在记忆里、且自己满足那条判据；空白差异不算抄错。
    """
    body = BASE + FRAGMENTS["M1"]
    assert judges.quote_holds("M1", quote_for("M1"), body)
    # 引文首尾多带了换行：折叠之后一致，照样算抄对了（模型常这么答）。
    assert judges.quote_holds("M1", f"\n{quote_for('M1')}\n", body)
    assert judges.quote_holds("M1", json.dumps([quote_for("M1")]), body)  # 引文被装进 JSON 数组字符串（范式 §9.88）
    assert not judges.quote_holds("M1", quote_for("M4"), body)  # 抄不出原文
    assert not judges.quote_holds("M1", "项目组", body)  # 抄出来了但不合判据
    assert not judges.quote_holds("M9", "项目组", body)  # 不认识的编号


class _UnknownIdBackend:
    """只会答一个不存在的判据编号的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 里带 M9 的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": [{"tag": "M9", "quote": "M9"}]})


class _MalformedBackend:
    """只会答一个畸形 tags 的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 是字符串列表的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": ["M1", "M3"]})


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
