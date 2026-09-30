"""日志行分诊的判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations


from jc.backend.types import Draft
from jc.jsontypes import JsonValue

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

#: 全空间 48 组 case。
CASES = features.build_cases(512)

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


def test_hygiene_rejects_a_state_without_a_content_block() -> None:
    """输入：一段没有日志抬头的 state。输出：无。预期行为：不通过，理由是认不出日志。"""
    spec, _ = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge({"state": "这里什么抬头都没有"}, spec)
    assert not verdict.ok
    assert "日志" in verdict.reason


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


def test_frame_accepts_faithful_and_rejects_a_broken_table() -> None:
    """输入：忠实 state 与缺了一行的 state。输出：无。

    预期行为：判据表缺一条就报出来——那是构造 bug，标签会跟着一起错。
    """
    judge = judges.FrameJudge("zh")
    spec, sample = faithful(CASE)
    assert judge.judge(sample, spec).ok
    broken = {"state": _drop_tag_line(str(sample["state"]), "N2")}
    verdict = judge.judge(broken, spec)
    assert not verdict.ok
    assert "判据表" in verdict.reason


def test_trace_agrees_with_the_plan() -> None:
    """输入：诚实的反解。输出：无。预期行为：通过，raw 里留下反解出来的命中组合。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["N1", "N3"]}


def test_trace_catches_a_lie() -> None:
    """输入：反解漏报一条。输出：无。预期行为：不通过，理由里两条都列出来。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "N1" in verdict.reason and "N3" in verdict.reason


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


def test_quote_holds_needs_both_the_quote_and_the_criterion() -> None:
    """输入：几种引文。输出：无。

    预期行为：引文得原样在日志里、且自己满足那条判据；空白差异不算抄错。
    """
    body = BASE + FRAGMENTS["N1"]
    assert judges.quote_holds("N1", quote_for("N1"), body)
    # 引文首尾多带了换行：折叠之后一致，照样算抄对了（模型常这么答）。
    assert judges.quote_holds("N1", f"\n{quote_for('N1')}\n", body)
    assert not judges.quote_holds("N1", quote_for("N4"), body)  # 抄不出原文
    assert not judges.quote_holds("N1", "事务已回滚", body)  # 抄出来了但不合判据
    assert not judges.quote_holds("N9", "事务已回滚", body)  # 不认识的编号


class _UnknownIdBackend:
    """只会答一个不存在的判据编号的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 里带 N9 的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": [{"tag": "N9", "quote": "N9"}]})


class _MalformedBackend:
    """只会答一个畸形 tags 的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 是字符串列表的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": ["N1", "N3"]})


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


def _draft(data: dict[str, JsonValue]) -> Draft:
    """输入：解析后的数据。输出：`Draft`。预期行为：记账字段填假值，不进产物。"""
    return Draft(
        data=data,
        raw_text="fake",
        backend="fake",
        provider="fake",
        model="fake",
        mode="fake",
    )
