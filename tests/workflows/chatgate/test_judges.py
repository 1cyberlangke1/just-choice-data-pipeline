"""聊天回复门控的判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations

import json

from jc.backend.types import Draft
from jc.jsontypes import JsonValue

from tests.workflows._fakes import draft as _draft
from tests.workflows.chatgate.conftest import (
    BASE,
    CLEAN,
    FLAW_FRAGMENTS,
    FRAGMENTS,
    FakeBackend,
    candidates_of,
    faithful,
    message_of,
    quote_for,
    spec_for,
    trace_report,
)
from workflows.chatgate import features, judges, locales, render
from workflows.chatgate.features import ChatCase
from workflows.chatgate.spec import ChatGateSpec

#: 全空间 45 组 case。
CASES = features.build_cases(512)

#: 一个普通 case：命中两条（C3 + C4），同事私聊，该发第一条。
CASE = ChatCase(degree=2, scene=features.SCENE_COLLEAGUE, slot=0)


def _drop_table_line(state: str, prefix: str) -> str:
    """输入：state、表行前缀（判据编号或毛病编号）。输出：抹掉那一行之后的 state。"""
    head = f"{render.LINE_PREFIX}{prefix}"
    lines = [line for line in state.splitlines() if not line.strip().startswith(head)]
    return "\n".join(lines)


def _sample_with(text: str, spec: ChatGateSpec) -> dict[str, JsonValue]:
    """输入：要换成别的正文的消息、规格。输出：消息被换掉、三条候选仍忠实于计划的样本。

    预期行为：只动消息那一块——判别器要能单独就「消息写歪了」判不通过，而不是先撞上候选那一条。
    """
    candidates = candidates_of(spec.plan.reply.flaws)
    draft = render.MessageDraft(text=text, candidates=candidates)
    return {"state": render.to_state(draft, locales.load("zh"))}


def test_hygiene_accepts_every_faithful_sample() -> None:
    """输入：全空间 45 组 case 的忠实样本。输出：无。预期行为：一条都不报。"""
    judge = judges.HygieneJudge("zh")
    for case in CASES:
        spec, sample = faithful(case)
        verdict = judge.judge(sample, spec)
        assert verdict.ok and not verdict.skipped, f"{case.describe()}: {verdict.reason}"
    assert "兑现" in judge.judge(*reversed(faithful(CASE))).reason


def test_hygiene_rejects_a_missing_hit() -> None:
    """输入：计划要 C3+C4、消息里只写了 C3 的样本。输出：无。

    预期行为：不通过，理由是命中组合对不上——少写一条和多重一条一样，都是标签兑现不了。
    """
    spec = spec_for(CASE)
    verdict = judges.HygieneJudge("zh").judge(_sample_with(BASE + FRAGMENTS["C3"], spec), spec)
    assert not verdict.ok and not verdict.skipped
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_an_extra_hit() -> None:
    """输入：消息多带了一条判据的样本。输出：无。预期行为：不通过，理由是命中组合对不上。"""
    case = ChatCase(degree=1, scene=features.SCENE_COLLEAGUE, slot=1)
    spare = next(tag for tag in features.TAG_IDS if tag not in case.tags())
    body = BASE + "".join(FRAGMENTS[tag] for tag in case.tags()) + FRAGMENTS[spare]
    spec = spec_for(case)
    verdict = judges.HygieneJudge("zh").judge(_sample_with(body, spec), spec)
    assert not verdict.ok
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_a_wrong_candidate_flaw() -> None:
    """输入：该发的那条候选被安上了一个毛病。输出：无。

    预期行为：不通过，理由点名第几条候选、计划要的是什么——候选毛病是 `reply` 那一族的金标签，
    安错位置等于答案错。
    """
    spec, sample = faithful(CASE)
    state = str(sample["state"])
    clean = spec.plan.reply.clean_answer
    broken = state.replace(
        f"{render.REPLY_SEP}{CLEAN}",
        f"{render.REPLY_SEP}{FLAW_FRAGMENTS['F1']}",
        1,
    )
    assert broken != state
    verdict = judges.HygieneJudge("zh").judge({"state": broken}, spec)
    assert not verdict.ok and not verdict.skipped
    assert f"第 {clean + 1} 条候选的毛病" in verdict.reason


def test_hygiene_rejects_a_state_without_a_message_block() -> None:
    """输入：一段没有消息抬头的 state。输出：无。预期行为：不通过，理由是消息解析不出来。"""
    spec, _ = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge({"state": "这里什么抬头都没有"}, spec)
    assert not verdict.ok
    assert "消息" in verdict.reason


def test_hygiene_rejects_fence_lines_anywhere() -> None:
    """输入：整段被围栏包起来的 state。输出：无。预期行为：不通过，理由是残留。"""
    spec, sample = faithful(CASE)
    sample["state"] = f"```text\n{sample['state']}\n```"
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "残留" in verdict.reason


def test_hygiene_rejects_a_pasted_rule() -> None:
    """输入：消息里整段贴了判据原文。输出：无。

    预期行为：不通过，理由是贴判据原文——那是「抄规格」的捷径，不是「读消息」（范式 §9.45）。
    """
    loc = locales.load("zh")
    rule = str(render.tag_copy(loc, "C1")["rule"])
    case = ChatCase(degree=1, scene=features.SCENE_COLLEAGUE, slot=1)
    spec = spec_for(case)
    verdict = judges.HygieneJudge("zh").judge(_sample_with(f"{BASE} {rule}", spec), spec)
    assert not verdict.ok and not verdict.skipped
    assert "原文原样贴了进去" in verdict.reason


def test_frame_accepts_faithful_and_rejects_a_broken_tag_table() -> None:
    """输入：忠实 state 与缺了一行判据表的 state。输出：无。

    预期行为：判据表缺一条就报出来——那是构造 bug，标签会跟着一起错。
    """
    judge = judges.FrameJudge("zh")
    spec, sample = faithful(CASE)
    assert judge.judge(sample, spec).ok
    broken = {"state": _drop_table_line(str(sample["state"]), "C2")}
    verdict = judge.judge(broken, spec)
    assert not verdict.ok
    assert "判据表" in verdict.reason


def test_frame_rejects_a_broken_flaw_table() -> None:
    """输入：缺了一行毛病表的 state。输出：无。预期行为：不通过，理由是毛病表对不上。"""
    spec, sample = faithful(CASE)
    broken = {"state": _drop_table_line(str(sample["state"]), "F2")}
    verdict = judges.FrameJudge("zh").judge(broken, spec)
    assert not verdict.ok
    assert "毛病表" in verdict.reason


def test_trace_agrees_with_the_plan() -> None:
    """输入：诚实的反解。输出：无。预期行为：通过，raw 里留下反解出来的命中组合。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["C3", "C4"]}
    assert "第 1 条" in verdict.reason


def test_trace_catches_a_lie() -> None:
    """输入：反解漏报一条命中。输出：无。预期行为：不通过，理由里两条都列出来。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "C3" in verdict.reason and "C4" in verdict.reason


def test_trace_catches_a_wrong_clean_candidate() -> None:
    """输入：反解把「哪条候选干净」说成另一条。输出：无。预期行为：不通过，理由点名第几条。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_clean_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "该发的是第" in verdict.reason


def test_trace_treats_a_refusal_to_pick_as_a_failure() -> None:
    """输入：反解答 `clean=0`（说不出来）。输出：无。

    预期行为：**不通过**——state 里印着毛病表，读不出干净的那条，说明这条样本对读者不唯一。
    """
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(_CleanZeroBackend(spec.plan.hits), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "说不出来" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_does_not_match_the_criterion() -> None:
    """输入：多报一条 C1、引文却是 C3 那一段的反解。输出：无。

    预期行为：这条误报被丢掉，判定仍然通过——「抄出来的那段按判据根本不算命中」必须由代码把关。
    """
    spec, sample = faithful(CASE)
    backend = _OverclaimBackend(CASE, tag="C1", quote=quote_for("C3"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["C3", "C4"]}
    assert "另有 1 条" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_is_fabricated() -> None:
    """输入：多报一条 C1、引文是消息里根本没有的一段。输出：无。

    预期行为：引文抄不出原文，同样按误报丢掉，判定通过。
    """
    spec, sample = faithful(CASE)
    backend = _OverclaimBackend(CASE, tag="C1", quote=quote_for("C1"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["C3", "C4"]}


def test_trace_skips_an_out_of_range_clean_pick() -> None:
    """输入：反解把 `clean` 答成 9。输出：无。预期行为：记成「没判成」，不算数据不合格。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(_CleanZeroBackend(spec.plan.hits, picked=9), "zh").judge(
        sample, spec
    )
    assert verdict.skipped and verdict.ok
    assert "超出候选编号范围" in verdict.reason


def test_trace_skips_without_a_spec() -> None:
    """输入：没有 spec。输出：无。预期行为：记成「没判成」，不去打扰后端。"""
    judge = judges.TraceJudge(FakeBackend(), "zh")
    verdict = judge.judge({"state": "x"}, None)
    assert verdict.skipped and verdict.ok


def test_quote_holds_needs_both_the_quote_and_the_criterion() -> None:
    """输入：几种引文。输出：无。

    预期行为：引文得原样在消息里、且自己满足那条判据；空白差异不算抄错。
    """
    body = message_of(("C3",))
    assert judges.quote_holds("C3", quote_for("C3"), body)
    # 引文首尾多带了换行：折叠之后一致，照样算抄对了（模型常这么答）。
    assert judges.quote_holds("C3", f"\n{quote_for('C3')}\n", body)
    assert judges.quote_holds("C3", json.dumps([quote_for("C3")]), body)  # 引文被装进 JSON 数组字符串（范式 §9.88）
    assert not judges.quote_holds("C3", quote_for("C1"), body)  # 抄不出原文
    assert not judges.quote_holds("C1", BASE, body)  # 抄出来了但不合判据
    assert not judges.quote_holds("C9", BASE, body)  # 不认识的编号


class _UnknownIdBackend:
    """只会答一个不存在的判据编号的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 里带 C9 的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": [{"tag": "C9", "quote": "C9"}], "clean": 1})


class _MalformedBackend:
    """只会答一个畸形 tags 的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 是字符串列表的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": ["C1", "C3"], "clean": 1})


class _CleanZeroBackend:
    """真话之外把 `clean` 答成指定值（默认 0 = 说不出来）的假后端。"""

    def __init__(self, hits: tuple[str, ...], picked: int = 0) -> None:
        """输入：该命中的判据、要答的 clean 值。输出：实例。"""
        self.report = trace_report(hits)
        self.picked = picked

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：真话的 tags + 指定的 clean。预期行为：不联网。"""
        del task
        return _draft({"tags": list(self.report), "clean": self.picked})


class _OverclaimBackend:
    """真话之外**多报一条**指定判据的假后端。

    字段：`report` 真话（由 `trace_report` 推出来）、`tag` / `quote` 那条多报的判据与它的引文
    ——分别验「引文在消息里但按判据不算命中」与「引文在消息里根本不存在」两条路径。
    """

    def __init__(self, case: ChatCase, tag: str, quote: str) -> None:
        """输入：注入特征、误报的判据编号与引文。输出：实例。"""
        self.report = trace_report(case.tags())
        self.tag = tag
        self.quote = quote

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：真话 + 一条误报。预期行为：不联网。"""
        del task
        tags = [*self.report, {"tag": self.tag, "quote": self.quote}]
        return _draft({"tags": tags, "clean": 1})
