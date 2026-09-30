"""数据库变更评审的判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue

from tests.workflows.sqlreview.conftest import (
    BASE_LINES,
    FakeBackend,
    faithful,
    packet_of,
    quote_for,
    render_lines,
    sample_of,
    spec_for,
    trace_report,
)
from workflows.sqlreview import features, judges, locales, render
from workflows.sqlreview.features import SqlReviewCase

#: 全空间 48 组 case。
CASES = features.build_cases(512)

#: 一个普通 case：命中两条，变更发生在事务型库上。
CASE = SqlReviewCase("Q1+Q3", features.WORKLOAD_OLTP)


def _drop_tag_line(state: str, tag: str) -> str:
    """输入：state、判据编号。输出：抹掉判据表里那一行之后的 state。"""
    prefix = f"{render.LINE_PREFIX}{tag}"
    lines = [line for line in state.splitlines() if not line.strip().startswith(prefix)]
    return "\n".join(lines)


def test_hygiene_rejects_a_missing_hit() -> None:
    """输入：计划要 Q1+Q2、记录里只写了 Q1 的样本。输出：无。

    预期行为：不通过，理由是命中组合对不上——少写一条和多重一条一样，都是标签兑现不了。
    """
    case = SqlReviewCase("Q1+Q2", features.WORKLOAD_OLTP)
    verdict = judges.HygieneJudge("zh").judge(sample_of(packet_of(("Q1",))), spec_for(case))
    assert not verdict.ok and not verdict.skipped
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_an_extra_hit() -> None:
    """输入：记录多带了一条判据的样本。输出：无。预期行为：不通过，理由是命中组合对不上。"""
    case = SqlReviewCase("Q1", features.WORKLOAD_OLTP)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(packet_of(("Q1", "Q4"))), spec_for(case)
    )
    assert not verdict.ok
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_a_packet_missing_a_structure_key() -> None:
    """输入：少了 `note:` 那一行的记录。输出：无。

    预期行为：不通过，理由点名缺了哪几行——四个结构键是 state 格式的一部分，少一行那几条钉在
    行上的判据就定位不到，标签会跟着一起错。
    """
    spec, sample = faithful(CASE)
    without_note = render_lines(
        {key: value for key, value in BASE_LINES.items() if key != "note"}
    )
    verdict = judges.HygieneJudge("zh").judge(sample_of(without_note), spec)
    assert not verdict.ok and not verdict.skipped
    assert "缺这几行" in verdict.reason


def test_hygiene_rejects_a_state_without_a_content_block() -> None:
    """输入：一段没有记录抬头的 state。输出：无。预期行为：不通过，理由是认不出记录。"""
    spec, _ = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge({"state": "这里什么抬头都没有"}, spec)
    assert not verdict.ok
    assert "评审记录" in verdict.reason


def test_hygiene_rejects_fence_lines_anywhere() -> None:
    """输入：整段被围栏包起来的 state。输出：无。预期行为：不通过，理由是残留。

    四条判据都是**子串**性质，加围栏不会改变命中组合，所以这条单验的是残留那道闸。
    """
    spec, sample = faithful(SqlReviewCase("Q1", features.WORKLOAD_OLTP))
    sample["state"] = f"```sql\n{sample['state']}\n```"
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "残留" in verdict.reason


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_hygiene_rejects_a_pasted_rule(lang: str) -> None:
    """输入：记录里整段贴了判据原文。输出：无。

    预期行为：不通过，理由是贴判据原文——那是「抄规格」的捷径，不是「写一屏变更记录」
    （范式 §9.45）。用 `Q1` 那条判据：贴进去之后按字面算仍然只命中 `Q1`，正好落在贴判据那道闸上
    （而不是先被命中组合那道闸拦下）。
    """
    loc = locales.load(lang)
    rule = str(render.tag_copy(loc, "Q1")["rule"])
    case = SqlReviewCase("Q1", features.WORKLOAD_OLTP)
    verdict = judges.HygieneJudge(lang).judge(
        sample_of(f"{packet_of(('Q1',))} {rule}", lang), spec_for(case)
    )
    assert not verdict.ok and not verdict.skipped
    assert "判据原文" in verdict.reason


def test_frame_accepts_faithful_and_rejects_a_broken_table() -> None:
    """输入：忠实 state 与缺了一行的 state。输出：无。

    预期行为：判据表缺一条就报出来——那是构造 bug，标签会跟着一起错。
    """
    judge = judges.FrameJudge("zh")
    spec, sample = faithful(CASE)
    assert judge.judge(sample, spec).ok
    broken = {"state": _drop_tag_line(str(sample["state"]), "Q2")}
    verdict = judge.judge(broken, spec)
    assert not verdict.ok
    assert "判据表" in verdict.reason


def test_frame_rejects_a_packet_missing_a_structure_key() -> None:
    """输入：记录少了 `note:` 那一行。输出：无。预期行为：不通过，理由点名缺了哪几行。"""
    spec, sample = faithful(CASE)
    without_note = render_lines(
        {key: value for key, value in BASE_LINES.items() if key != "note"}
    )
    verdict = judges.FrameJudge("zh").judge(sample_of(without_note), spec)
    assert not verdict.ok
    assert "缺这几行" in verdict.reason


def test_trace_agrees_with_the_plan() -> None:
    """输入：诚实的反解。输出：无。预期行为：通过，raw 里留下反解出来的命中组合。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["Q1", "Q3"]}


def test_trace_catches_a_lie() -> None:
    """输入：反解漏报一条。输出：无。预期行为：不通过，理由里两条都列出来。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "Q1" in verdict.reason and "Q3" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_does_not_match_the_criterion() -> None:
    """输入：多报一条 Q1、引文却是 Q3 那一段的反解。输出：无。

    预期行为：这条误报被丢掉，判定仍然通过——「抄出来的那段按判据根本不算命中」必须由代码把关。
    """
    case = SqlReviewCase("Q3+Q4", features.WORKLOAD_OLTP)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("Q3"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["Q3", "Q4"]}
    assert "另有 1 条" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_is_fabricated() -> None:
    """输入：多报一条 Q1、引文是记录里根本没有的一段。输出：无。

    预期行为：引文抄不出原文，同样按误报丢掉，判定通过。
    """
    case = SqlReviewCase("Q3+Q4", features.WORKLOAD_OLTP)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("Q1"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["Q3", "Q4"]}


def test_quote_holds_needs_both_the_quote_and_the_criterion() -> None:
    """输入：几种引文。输出：无。

    预期行为：引文得原样在记录里、且自己满足那条判据；空白差异不算抄错。
    """
    body = packet_of(("Q1", "Q4"))
    assert judges.quote_holds("Q1", quote_for("Q1"), body)
    # 引文首尾多带了换行：折叠之后一致，照样算抄对了（模型常这么答）。
    assert judges.quote_holds("Q1", f"\n{quote_for('Q1')}\n", body)
    assert not judges.quote_holds("Q1", quote_for("Q4"), body)  # 抄出来了但不合判据
    assert not judges.quote_holds("Q1", "note: 这个字段的回填脚本待确认", body)  # 同上
    assert not judges.quote_holds("Q9", "随便一段", body)  # 不认识的编号


def test_quote_holds_tracks_the_plan_on_every_faithful_packet() -> None:
    """输入：全空间 48 组 case 的忠实记录。输出：无。

    预期行为：本 workflow 的四条判据都是**纯子串**性质（不像 `turntake` 的「标点断在半路」看的是
    整段结尾），所以「引文命中」与「整段命中」在这四族里总是同进同退：`quote_holds` 的第二、三条
    条件不会互相打架。谁把某条判据改成看整段结尾，这条会当场变红。
    """
    for case in CASES:
        packet = packet_of(case.tags())
        for tag in features.TAG_IDS:
            assert judges.quote_holds(tag, quote_for(tag), packet) is (tag in case.tags())


class _UnknownIdBackend:
    """只会答一个不存在的判据编号的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 里带 Q9 的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": [{"tag": "Q9", "quote": "Q9"}]})


class _MalformedBackend:
    """只会答一个畸形 tags 的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 是字符串列表的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": ["Q1", "Q3"]})


class _OverclaimBackend:
    """真话之外**多报一条 Q1** 的假后端。

    字段：`report` 真话（由 `trace_report` 推出来）、`quote` 给那条多报的 Q1 配的引文——分别验
    「引文在记录里但按判据不算命中」与「引文在记录里根本不存在」两条路径。
    """

    def __init__(self, case: SqlReviewCase, quote: str) -> None:
        """输入：注入特征、误报的引文。输出：实例。"""
        self.report = trace_report(case)
        self.quote = quote

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：真话 + 一条 Q1 误报。预期行为：不联网。"""
        del task
        tags = [*self.report, {"tag": "Q1", "quote": self.quote}]
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
