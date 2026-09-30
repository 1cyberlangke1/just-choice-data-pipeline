"""广告元素判定的判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations

import pytest

from jc.backend.types import AllProvidersDown, Draft, TransportError
from jc.jsontypes import JsonValue

from tests.workflows.adblock.conftest import (
    BASE,
    FakeBackend,
    faithful,
    quote_for,
    sample_of,
    spec_for,
    text_of,
    trace_report,
)
from workflows.adblock import features, judges, locales, render
from workflows.adblock.features import AdBlockCase

#: 全空间 48 组 case。
CASES = features.build_cases(512)

#: 一个普通 case：命中两条，元素长在文章正文页上。
CASE = AdBlockCase("A1+A3", features.SURFACE_ARTICLE)


def _drop_tag_line(state: str, tag: str) -> str:
    """输入：state、判据编号。输出：抹掉判据表里那一行之后的 state。"""
    prefix = f"{render.LINE_PREFIX}{tag}"
    lines = [line for line in state.splitlines() if not line.strip().startswith(prefix)]
    return "\n".join(lines)


def test_hygiene_accepts_every_faithful_sample() -> None:
    """输入：全空间 48 组 case 的忠实样本。输出：无。预期行为：一条都不报。"""
    judge = judges.HygieneJudge("zh")
    for case in CASES:
        spec, sample = faithful(case)
        verdict = judge.judge(sample, spec)
        assert verdict.ok and not verdict.skipped, f"{case.describe()}: {verdict.reason}"
    assert "兑现" in judge.judge(*reversed(faithful(CASE))).reason


def test_hygiene_rejects_a_missing_hit() -> None:
    """输入：计划要 A1+A2、片段里只写了 A1 的样本。输出：无。

    预期行为：不通过，理由是命中组合对不上——少写一条和多重一条一样，都是标签兑现不了。
    """
    case = AdBlockCase("A1+A2", features.SURFACE_ARTICLE)
    verdict = judges.HygieneJudge("zh").judge(sample_of(text_of(("A1",))), spec_for(case))
    assert not verdict.ok and not verdict.skipped
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_an_extra_hit() -> None:
    """输入：片段多带了一条判据的样本。输出：无。预期行为：不通过，理由是命中组合对不上。"""
    case = AdBlockCase("A1", features.SURFACE_ARTICLE)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(text_of(("A1", "A4"))), spec_for(case)
    )
    assert not verdict.ok
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_a_state_without_a_content_block() -> None:
    """输入：一段没有片段抬头的 state。输出：无。预期行为：不通过，理由是认不出片段。"""
    spec, _ = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge({"state": "这里什么抬头都没有"}, spec)
    assert not verdict.ok
    assert "片段" in verdict.reason


def test_hygiene_rejects_fence_lines_anywhere() -> None:
    """输入：整段被围栏包起来的 state。输出：无。预期行为：不通过，理由是残留。

    四条判据都是**子串**性质，加围栏不会改变命中组合，所以这条单验的是残留那道闸。
    """
    spec, sample = faithful(AdBlockCase("A1", features.SURFACE_ARTICLE))
    sample["state"] = f"```text\n{sample['state']}\n```"
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "残留" in verdict.reason


def test_hygiene_rejects_a_pasted_rule() -> None:
    """输入：片段里整段贴了判据原文。输出：无。

    预期行为：不通过，理由是贴判据原文——那是「抄规格」的捷径，不是「写一段页面元素」（范式 §9.45）。
    用 A4 那条判据：它的原文里带 `display:none`，所以贴进去之后按字面算仍然命中 A4，正好落在
    贴判据那道闸上（而不是先被命中组合那道闸拦下）。
    """
    loc = locales.load("zh")
    rule = str(render.tag_copy(loc, "A4")["rule"])
    case = AdBlockCase("A4", features.SURFACE_ARTICLE)
    verdict = judges.HygieneJudge("zh").judge(sample_of(f"{BASE} {rule}"), spec_for(case))
    assert not verdict.ok and not verdict.skipped
    assert "判据原文" in verdict.reason


def test_hygiene_skips_without_a_spec_or_a_text_state() -> None:
    """输入：没有 spec / state 不是文本。输出：无。预期行为：记成「没判成」，不当成不合格。"""
    judge = judges.HygieneJudge("zh")
    assert judge.judge({"state": "x"}, None).skipped
    spec, _ = faithful(CASE)
    assert judge.judge({"state": 42}, spec).skipped


def test_frame_accepts_faithful_and_rejects_a_broken_table() -> None:
    """输入：忠实 state 与缺了一行的 state。输出：无。

    预期行为：判据表缺一条就报出来——那是构造 bug，标签会跟着一起错。
    """
    judge = judges.FrameJudge("zh")
    spec, sample = faithful(CASE)
    assert judge.judge(sample, spec).ok
    broken = {"state": _drop_tag_line(str(sample["state"]), "A2")}
    verdict = judge.judge(broken, spec)
    assert not verdict.ok
    assert "判据表" in verdict.reason


def test_frame_skips_without_a_spec_or_a_text_state() -> None:
    """输入：没有 spec / state 不是文本。输出：无。预期行为：记成「没判成」。"""
    judge = judges.FrameJudge("zh")
    assert judge.judge({"state": "x"}, None).skipped
    spec, _ = faithful(CASE)
    assert judge.judge({"state": 42}, spec).skipped


def test_trace_agrees_with_the_plan() -> None:
    """输入：诚实的反解。输出：无。预期行为：通过，raw 里留下反解出来的命中组合。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["A1", "A3"]}


def test_trace_catches_a_lie() -> None:
    """输入：反解漏报一条。输出：无。预期行为：不通过，理由里两条都列出来。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "A1" in verdict.reason and "A3" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_does_not_match_the_criterion() -> None:
    """输入：多报一条 A1、引文却是 A3 那一段的反解。输出：无。

    预期行为：这条误报被丢掉，判定仍然通过——「抄出来的那段按判据根本不算命中」必须由代码把关。
    """
    case = AdBlockCase("A3+A4", features.SURFACE_ARTICLE)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("A3"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["A3", "A4"]}
    assert "另有 1 条" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_is_fabricated() -> None:
    """输入：多报一条 A1、引文是片段里根本没有的一段。输出：无。

    预期行为：引文抄不出原文，同样按误报丢掉，判定通过。
    """
    case = AdBlockCase("A3+A4", features.SURFACE_ARTICLE)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("A1"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["A3", "A4"]}


def test_trace_marks_transport_errors_as_skipped() -> None:
    """输入：反解时端点故障。输出：无。预期行为：记成「没判成」，不算数据不合格。"""
    spec, sample = faithful(CASE)
    judge = judges.TraceJudge(FakeBackend(trace_raises=TransportError("boom")), "zh")
    verdict = judge.judge(sample, spec)
    assert verdict.skipped and verdict.ok
    assert "端点故障" in verdict.reason


def test_trace_reraises_when_every_provider_is_down() -> None:
    """输入：全部 provider 都挂了。输出：无。预期行为：往上抛，让整批停下（不是这一条的错）。"""
    spec, sample = faithful(CASE)
    judge = judges.TraceJudge(FakeBackend(trace_raises=AllProvidersDown("all down")), "zh")
    with pytest.raises(AllProvidersDown):
        judge.judge(sample, spec)


def test_trace_skips_an_unknown_tag_id() -> None:
    """输入：反解答了一个不存在的判据编号。输出：无。

    预期行为：记成「没判成」——那是反解格式没对齐，不是这条数据错。
    """
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(_UnknownIdBackend(), "zh").judge(sample, spec)
    assert verdict.skipped and verdict.ok
    assert "不认识的判据编号" in verdict.reason


def test_trace_skips_a_malformed_draft() -> None:
    """输入：反解草稿里 tags 不是一个对象列表。输出：无。预期行为：记成「没判成」。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(_MalformedBackend(), "zh").judge(sample, spec)
    assert verdict.skipped and verdict.ok
    assert "判不了" in verdict.reason


def test_trace_prompt_carries_the_whole_state() -> None:
    """输入：一段 state。输出：无。预期行为：反解提示词把整段材料原样带上。"""
    _, sample = faithful(CASE)
    loc = locales.load("zh")
    prompt = judges.build_trace_prompt(str(sample["state"]), loc)
    assert str(sample["state"]) in prompt
    assert str(loc["verify"]["state_header"]) in prompt
    assert "{" not in prompt and "}" not in prompt


def test_quote_holds_needs_both_the_quote_and_the_criterion() -> None:
    """输入：几种引文。输出：无。

    预期行为：引文得原样在片段里、且自己满足那条判据；空白差异不算抄错。
    """
    body = text_of(("A1", "A4"))
    assert judges.quote_holds("A1", quote_for("A1"), body)
    # 引文首尾多带了换行：折叠之后一致，照样算抄对了（模型常这么答）。
    assert judges.quote_holds("A1", f"\n{quote_for('A1')}\n", body)
    assert not judges.quote_holds("A1", quote_for("A4"), body)  # 抄出来了但不合判据
    assert not judges.quote_holds("A1", "会议室还是老地方", body)  # 抄出来了但不合判据
    assert not judges.quote_holds("A9", "会议室还是老地方", body)  # 不认识的编号


def test_quote_holds_also_requires_the_whole_snippet_to_match() -> None:
    """输入：一段**按判据不算命中**的片段 + 一条引文。输出：无。

    预期行为：引文自己合判据、也在片段里，但整段片段不合那条判据时不算数——`quote_holds` 的第三条
    （范式 §9.47：判据是**整段文本**的性质，引文只是「你确实读到过这里」的凭据）。这里用 A2 验：
    把「赞助」写进**属性值**里（`title="赞助"`），它在片段里原样出现，但按判据（只有 `>` 之后、
    `<` 之前的文字才算）整段不算命中。
    """
    body = '<img title="赞助" src="https://example.com/x.png" alt="">'
    assert features.PATTERNS["A2"].search("赞助")
    assert not features.PATTERNS["A2"].search(body)
    assert judges.quote_holds("A2", "赞助", body) is False


def test_quote_holds_tracks_the_plan_on_every_faithful_snippet() -> None:
    """输入：全空间 48 组 case 的忠实片段。输出：无。

    预期行为：本 workflow 的四条判据都是**纯子串**性质（不像 `turntake` 的「标点断在半路」看的是
    整段结尾），所以「引文命中」与「整段命中」在这四族里总是同进同退：`quote_holds` 的第二、三条
    条件不会互相打架。谁把某条判据改成看整段结尾，这条会当场变红。
    """
    for case in CASES:
        snippet = text_of(case.tags())
        for tag in features.TAG_IDS:
            assert judges.quote_holds(tag, quote_for(tag), snippet) is (tag in case.tags())


class _UnknownIdBackend:
    """只会答一个不存在的判据编号的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 里带 A9 的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": [{"tag": "A9", "quote": "A9"}]})


class _MalformedBackend:
    """只会答一个畸形 tags 的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`tags` 是字符串列表的草稿。预期行为：不联网。"""
        del task
        return _draft({"tags": ["A1", "A3"]})


class _OverclaimBackend:
    """真话之外**多报一条 A1** 的假后端。

    字段：`report` 真话（由 `trace_report` 推出来）、`quote` 给那条多报的 A1 配的引文——分别验
    「引文在片段里但按判据不算命中」与「引文在片段里根本不存在」两条路径。
    """

    def __init__(self, case: AdBlockCase, quote: str) -> None:
        """输入：注入特征、误报的引文。输出：实例。"""
        self.report = trace_report(case)
        self.quote = quote

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：真话 + 一条 A1 误报。预期行为：不联网。"""
        del task
        tags = [*self.report, {"tag": "A1", "quote": self.quote}]
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
