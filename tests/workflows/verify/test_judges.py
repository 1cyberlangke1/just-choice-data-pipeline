"""判别器测试：两个零 token 的规则判别器 + 一次后端调用的反解判别器。"""

from __future__ import annotations

import pytest

from jc.backend.types import AllProvidersDown, GenerationBackend, TransportError
from workflows.verify import features, judges as judges_mod, locales, render
from workflows.verify.features import TagSet
from workflows.verify.spec import VerifySpec

CASE = TagSet("bug_fixed", "hedged", "partial_pass", "covered", "uncommitted")
CLEAN = "把订单列表偶发的空指针修掉了"


def spec_for(tags: TagSet = CASE, seq: int = 0) -> VerifySpec:
    """输入：一组特征。输出：一个规格。预期行为：切分与问法固定，测试只关心标签。"""
    return VerifySpec(tags=tags, phrasing=0, split="train", seq=seq)


def state_for(claim: str = CLEAN, lang: str = "zh", tags: TagSet = CASE) -> str:
    """输入：一句声明、语言、特征。输出：一条完整的 state 文本（后两行由代码写）。"""
    return render.to_state({"claim": claim}, tags, locales.load(lang))


# ---------- 卫生 ----------


def test_hygiene_judge_accepts_a_clean_state() -> None:
    """声明单行、无残留、不提证据 → 通过。"""
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state_for()})
    assert verdict.ok is True, verdict.reason
    assert verdict.skipped is False


def test_hygiene_judge_flags_a_multi_line_claim() -> None:
    """输入：声明写成两行。输出：Verdict。预期行为：判不通过并点名「多行」。"""
    verdict = judges_mod.HygieneJudge("zh").judge(
        {"state": state_for("第一行\n（顺手把注释也补了）")}
    )
    assert verdict.ok is False
    assert "多行" in verdict.reason


def test_hygiene_judge_flags_an_empty_claim() -> None:
    """「声明」那一段是空的 → 不通过（state 没法当一条声明看）。"""
    loc = locales.load("zh")
    state = f"{locales.header(loc, loc['meta']['claim_label'])}\n{locales.header(loc, loc['meta']['facts_label'])}"
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state})
    assert verdict.ok is False
    assert "空的" in verdict.reason


def test_hygiene_judge_flags_a_leaked_section_name() -> None:
    """输入：声明开头又写了一遍「声明：」。输出：Verdict。预期行为：判不通过并点名段落名。"""
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state_for("声明：把空指针修掉了")})
    assert verdict.ok is False
    assert "段落名" in verdict.reason


def test_hygiene_judge_flags_evidence_in_the_claim() -> None:
    """输入：声明里写了证据。输出：Verdict。

    预期行为：判不通过——声明与代码渲染的事实行一旦各说各话，这道题就有歧义了。
    """
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state_for("把空指针修掉了，测试全过了")})
    assert verdict.ok is False
    assert "证据" in verdict.reason


def test_hygiene_judge_reads_the_marker_from_the_locale() -> None:
    """输入：英文 state。输出：Verdict。

    预期行为：认得出 `[Claim]` 那一段，代码里没写死中文——换个语言照样认。
    """
    loc = locales.load("en")
    state = render.to_state({"claim": "fixed the flaky retry"}, CASE, loc)
    assert judges_mod.HygieneJudge("en").judge({"state": state}).ok is True
    assert judges_mod.HygieneJudge("en").judge({"state": state + "\n```"}).ok is False


# ---------- 框架 ----------


def test_frame_judge_accepts_the_code_rendered_frame() -> None:
    """输入：渲染器拼出来的 state。输出：Verdict。预期行为：通过，且不是跳过。"""
    verdict = judges_mod.FrameJudge("zh").judge({"state": state_for()}, spec_for())
    assert verdict.ok is True and verdict.skipped is False
    assert "一致" in verdict.reason


def test_frame_judge_rejects_facts_that_do_not_match_the_tags() -> None:
    """输入：事实行被换成别的特征算出来的。输出：Verdict。

    预期行为：判不通过——事实行与注入特征对不上说明渲染挂错了 tags，四个可重算的标签
    会跟着一起错。
    """
    loc = locales.load("zh")
    other = TagSet("bug_fixed", "hedged", "all_pass", "covered", "committed")
    broken = state_for().replace(
        render.facts_line(CASE.facts, loc), render.facts_line(other.facts, loc)
    )
    verdict = judges_mod.FrameJudge("zh").judge({"state": broken}, spec_for())
    assert verdict.ok is False
    assert "事实行" in verdict.reason


def test_frame_judge_rejects_gates_that_do_not_match_the_facts() -> None:
    """输入：门槛行与事实行对不上。输出：Verdict。

    预期行为：判不通过——两行由同一份事实算出，对不上就是渲染串味（题面会自相矛盾）。
    """
    loc = locales.load("zh")
    other = TagSet("bug_fixed", "hedged", "all_pass", "covered", "committed")
    broken = state_for().replace(
        render.gates_line(CASE.facts, loc), render.gates_line(other.facts, loc)
    )
    verdict = judges_mod.FrameJudge("zh").judge({"state": broken}, spec_for())
    assert verdict.ok is False
    assert "门槛行" in verdict.reason


def test_frame_judge_skips_without_spec_or_text() -> None:
    """没有 spec、或 state 不是文本，都记成跳过（没判成），不是数据不合格。"""
    no_spec = judges_mod.FrameJudge("zh").judge({"state": state_for()})
    no_text = judges_mod.FrameJudge("zh").judge({"state": 42}, spec_for())
    assert no_spec.ok is True and no_spec.skipped is True
    assert no_text.ok is True and no_text.skipped is True


# ---------- 反解 ----------


def test_build_verify_prompt_carries_the_state_and_the_options() -> None:
    """反解提示词要把 state 原样带上，并列出与数据集一致的两种选项。"""
    prompt = judges_mod.build_verify_prompt(state_for(), locales.load("zh"))
    assert CLEAN in prompt
    for name in (*features.CLAIM_KINDS, *features.TONES):
        assert name in prompt


def test_trace_judge_passes_when_reverse_parse_agrees(fake_backend: GenerationBackend) -> None:
    """反解两维都与注入一致 → 通过，并在 raw 里留下反解结果。"""
    judge = judges_mod.TraceJudge(backend=fake_backend(), locale="zh")
    verdict = judge.judge({"state": state_for()}, spec_for())
    assert verdict.ok is True and verdict.skipped is False
    assert verdict.raw["traced"] == {"kind": "bug_fixed", "tone": "hedged"}


def test_trace_judge_rejects_a_misread_kind(fake_backend: GenerationBackend) -> None:
    """反解说成另一个「声称做完的事」→ 不通过，理由里同时列出注入值与反解值。"""
    judge = judges_mod.TraceJudge(backend=fake_backend(trace_lies=True), locale="zh")
    verdict = judge.judge({"state": state_for()}, spec_for())
    assert verdict.ok is False
    assert "bug_fixed" in verdict.reason and "perf_improved" in verdict.reason


def test_trace_judge_rejects_a_misread_tone(fake_backend: GenerationBackend) -> None:
    """口气读错也算不通过——声明必须同时承载两维，只对一维不够。"""
    judge = judges_mod.TraceJudge(backend=fake_backend(trace_lies_tone=True), locale="zh")
    verdict = judge.judge({"state": state_for()}, spec_for())
    assert verdict.ok is False
    assert "hedged" in verdict.reason and "absolute" in verdict.reason


def test_trace_judge_skips_without_spec_or_backend() -> None:
    """没有 spec、或没有后端，都记成跳过（没判成），不是数据不合格。"""
    no_spec = judges_mod.TraceJudge(backend=None, locale="zh").judge({"state": "x"})
    no_backend = judges_mod.TraceJudge(backend=None, locale="zh").judge(
        {"state": "x"}, spec_for()
    )
    assert no_spec.ok is True and no_spec.skipped is True
    assert no_backend.ok is True and no_backend.skipped is True


def test_trace_judge_skips_an_endpoint_failure(fake_backend: GenerationBackend) -> None:
    """端点故障只算「没判成」——记成「不通过」会把端点抖动写成数据质量问题。"""
    judge = judges_mod.TraceJudge(
        backend=fake_backend(trace_raises=TransportError("网关 504")), locale="zh"
    )
    verdict = judge.judge({"state": state_for()}, spec_for())
    assert verdict.ok is True and verdict.skipped is True
    assert "端点故障" in verdict.reason
    assert verdict.raw == {"error": "TransportError"}


def test_trace_judge_reraises_when_every_provider_is_down(
    fake_backend: GenerationBackend,
) -> None:
    """所有 provider 都挂了要冒泡（整批停下是唯一的停机条件），不许吞成「跳过」。"""
    judge = judges_mod.TraceJudge(
        backend=fake_backend(trace_raises=AllProvidersDown("全挂")), locale="zh"
    )
    with pytest.raises(AllProvidersDown):
        judge.judge({"state": state_for()}, spec_for())


def test_trace_judge_skips_when_the_draft_misses_fields(
    fake_backend: GenerationBackend,
) -> None:
    """反解草稿缺字段 → 判不了，记成跳过并点名缺了哪个字段。"""
    judge = judges_mod.TraceJudge(backend=fake_backend(trace_data={"kind": "bug_fixed"}), locale="zh")
    verdict = judge.judge({"state": state_for()}, spec_for())
    assert verdict.ok is True and verdict.skipped is True
    assert "缺字段" in verdict.reason and "tone" in verdict.reason


def test_trace_judge_rejects_an_out_of_range_value(fake_backend: GenerationBackend) -> None:
    """反解给了一个不在取值域里的值 → 与注入值不等，判不通过（不是跳过）。"""
    judge = judges_mod.TraceJudge(
        backend=fake_backend(trace_data={"kind": "bug_fixed", "tone": "shouty"}), locale="zh"
    )
    verdict = judge.judge({"state": state_for()}, spec_for())
    assert verdict.ok is False
    assert "shouty" in verdict.reason


# ---------- 选项反解 ----------
