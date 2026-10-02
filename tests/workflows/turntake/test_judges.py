"""回合结束判定的判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations

import json

from jc.backend.types import Draft

from tests.workflows._fakes import draft as _draft
from tests.workflows.turntake.conftest import (
    BASE,
    faithful,
    quote_for,
    sample_of,
    spec_for,
    text_of,
    trace_report,
)
from workflows.turntake import features, judges, locales, render
from workflows.turntake.features import TurnTakeCase

#: 全空间 48 组 case。
CASES = features.build_cases(512)

#: 一个普通 case：命中两条，通道是聊天消息。
CASE = TurnTakeCase("T1+T3", features.CHANNEL_IM)


def _drop_tag_line(state: str, tag: str) -> str:
    """输入：state、判据编号。输出：抹掉判据表里那一行之后的 state。"""
    prefix = f"{render.LINE_PREFIX}{tag}"
    lines = [line for line in state.splitlines() if not line.strip().startswith(prefix)]
    return "\n".join(lines)


def test_hygiene_rejects_a_missing_hit() -> None:
    """输入：计划要 T1+T2、输入里只写了 T1 的样本。输出：无。

    预期行为：不通过，理由是命中组合对不上——少写一条和多重一条一样，都是标签兑现不了。
    """
    case = TurnTakeCase("T1+T2", features.CHANNEL_IM)
    verdict = judges.HygieneJudge("zh").judge(sample_of(text_of(("T1",))), spec_for(case))
    assert not verdict.ok and not verdict.skipped
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_an_extra_hit() -> None:
    """输入：输入多带了一条判据的样本。输出：无。预期行为：不通过，理由是命中组合对不上。"""
    case = TurnTakeCase("T1", features.CHANNEL_IM)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(text_of(("T1", "T4"))), spec_for(case)
    )
    assert not verdict.ok
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_a_state_without_a_content_block() -> None:
    """输入：一段没有输入抬头的 state。输出：无。预期行为：不通过，理由是认不出输入。"""
    spec, _ = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge({"state": "这里什么抬头都没有"}, spec)
    assert not verdict.ok
    assert "输入" in verdict.reason


def test_hygiene_rejects_fence_lines_anywhere() -> None:
    """输入：整段被围栏包起来的 state。输出：无。预期行为：不通过，理由是残留。

    用一个只靠「括号开了没关」命中的 case：围栏是加在整段两头的，`T1` / `T2` 看的是整段
    结尾，包一层之后结尾就不是它们了——那是另一条闸（命中组合）该拦的，这里要单验残留。
    """
    spec, sample = faithful(TurnTakeCase("T3", features.CHANNEL_IM))
    sample["state"] = f"```text\n{sample['state']}\n```"
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "残留" in verdict.reason


def test_hygiene_rejects_a_pasted_rule() -> None:
    """输入：输入里整段贴了判据原文。输出：无。

    预期行为：不通过，理由是贴判据原文——那是「抄规格」的捷径，不是「写一段用户输入」（范式 §9.45）。
    """
    loc = locales.load("zh")
    rule = str(render.tag_copy(loc, "T1")["rule"])
    case = TurnTakeCase("T1", features.CHANNEL_IM)
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
    broken = {"state": _drop_tag_line(str(sample["state"]), "T2")}
    verdict = judge.judge(broken, spec)
    assert not verdict.ok
    assert "判据表" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_does_not_match_the_criterion() -> None:
    """输入：多报一条 T1、引文却是 T3 那一段的反解。输出：无。

    预期行为：这条误报被丢掉，判定仍然通过——「抄出来的那段按判据根本不算命中」必须由代码把关。
    """
    case = TurnTakeCase("T3+T4", features.CHANNEL_IM)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("T3"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["T3", "T4"]}
    assert "另有 1 条" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_is_fabricated() -> None:
    """输入：多报一条 T1、引文是输入里根本没有的一段探测词。输出：无。

    预期行为：引文抄不出原文，同样按误报丢掉，判定通过。
    """
    case = TurnTakeCase("T3+T4", features.CHANNEL_IM)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("T1"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["T3", "T4"]}


def test_quote_holds_needs_both_the_quote_and_the_criterion() -> None:
    """输入：几种引文。输出：无。

    预期行为：引文得原样在输入里、且自己满足那条判据；空白差异不算抄错。
    """
    body = text_of(("T1", "T4"))
    assert judges.quote_holds("T1", quote_for("T1"), body)
    # 引文首尾多带了换行：折叠之后一致，照样算抄对了（模型常这么答）。
    assert judges.quote_holds("T1", f"\n{quote_for('T1')}\n", body)
    assert judges.quote_holds("T1", json.dumps([quote_for("T1")]), body)  # 引文被装进 JSON 数组字符串（范式 §9.88）
    assert not judges.quote_holds("T1", quote_for("T4"), body)  # 抄出来了但不合判据
    assert not judges.quote_holds("T1", "会议室还是老地方", body)  # 抄出来了但不合判据
    assert not judges.quote_holds("T9", "会议室还是老地方", body)  # 不认识的编号


def test_quote_holds_rejects_a_quote_that_only_matches_out_of_context() -> None:
    """输入：拿一个孤立的逗号当引文、却声称命中「标点断在半路」。输出：无。

    预期行为：不算数——那条判据看的是**整段结尾**，引文本身是个逗号不等于整段以逗号收尾。
    真机踩过（`tt-10d` zh 侧）：反解多报一条 `T2`、引文就是句子中间随便一个「，」，判定于是
    把一条本来合格的样本拒掉了。
    """
    body = text_of(("T1", "T4"))  # 以「，然后」收尾，不以标点收尾
    assert "，" in body and not body.endswith("，")
    assert judges.quote_holds("T2", "，", body) is False
    assert judges.quote_holds("T1", quote_for("T1"), body) is True


class _UnknownIdBackend:
    """只会答一个不存在的判据编号的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 里带 T9 的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": [{"tag": "T9", "quote": "T9"}]})


class _MalformedBackend:
    """只会答一个畸形 tags 的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 是字符串列表的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": ["T1", "T3"]})


class _OverclaimBackend:
    """真话之外**多报一条 T1** 的假后端。

    字段：`report` 真话（由 `trace_report` 推出来）、`quote` 给那条多报的 T1 配的引文——分别验
    「引文在输入里但按判据不算命中」与「引文在输入里根本不存在」两条路径。
    """

    def __init__(self, case: TurnTakeCase, quote: str) -> None:
        """输入：注入特征、误报的引文。输出：实例。"""
        self.report = trace_report(case)
        self.quote = quote

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：真话 + 一条 T1 误报。预期行为：不联网。"""
        del task
        tags = [*self.report, {"tag": "T1", "quote": self.quote}]
        return _draft({"tags": tags})
