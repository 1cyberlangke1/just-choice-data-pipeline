"""判别器测试：两个零 token 的规则判别器 + 一次后端调用的反解判别器。"""

from __future__ import annotations

import pytest

from jc.backend.types import AllProvidersDown, GenerationBackend, TransportError
from workflows.modelroute import features, judges as judges_mod, locales, render
from workflows.modelroute.features import CATALOG, TagSet
from workflows.modelroute.spec import RouteSpec

CASE = TagSet("code_gen", "small", "batch", "tight", "all_ok")


def spec_for(tags: TagSet, seq: int = 0) -> RouteSpec:
    """输入：一组特征。输出：一个规格。预期行为：切分与问法固定，测试只关心标签。"""
    return RouteSpec(tags=tags, phrasing=0, split="train", seq=seq)


def state_for(request: str = "帮我把这段函数改成异步的", lang: str = "zh") -> str:
    """输入：一句请求、语言。输出：一条完整的 state 文本（候选表与门槛由代码写）。"""
    return render.to_state({"request": request}, CASE, locales.load(lang))


# ---------- 卫生 ----------


def test_hygiene_judge_accepts_a_clean_state() -> None:
    """请求单行、无残留 → 通过。"""
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state_for()})
    assert verdict.ok is True, verdict.reason
    assert verdict.skipped is False


@pytest.mark.parametrize("residue", ["```", "```json", "<br>", "<br/>"])
def test_hygiene_judge_rejects_format_residue(residue: str) -> None:
    """围栏、HTML 换行标签都算格式残留，且要点名是哪一个。"""
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state_for() + f"\n{residue}"})
    assert verdict.ok is False
    assert residue in verdict.reason


def test_hygiene_judge_flags_a_multi_line_request() -> None:
    """输入：请求写成两行。输出：Verdict。预期行为：判不通过并点名「多行」。"""
    verdict = judges_mod.HygieneJudge("zh").judge(
        {"state": state_for("第一行\n（材料在附件里）")}
    )
    assert verdict.ok is False
    assert "多行" in verdict.reason


def test_hygiene_judge_flags_an_empty_request() -> None:
    """「请求」那一段是空的 → 不通过（state 没法当一条请求看）。"""
    state = "【请求】\n【候选模型】（名字 | 提供方 | 窗口 | 能力 | 价格 | 时延）"
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state})
    assert verdict.ok is False
    assert "空的" in verdict.reason


def test_hygiene_judge_flags_a_leaked_section_name() -> None:
    """输入：请求开头又写了一遍「请求：」。输出：Verdict。预期行为：判不通过并点名段落名。"""
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state_for("请求：帮我改代码")})
    assert verdict.ok is False
    assert "段落名" in verdict.reason


def test_hygiene_judge_reads_the_marker_from_the_locale() -> None:
    """输入：英文 state。输出：Verdict。

    预期行为：认得出 `[Request]` 那一段，代码里没写死中文——换个语言照样认。
    """
    loc = locales.load("en")
    state = render.to_state({"request": "summarize the log"}, CASE, loc)
    assert judges_mod.HygieneJudge("en").judge({"state": state}).ok is True
    assert judges_mod.HygieneJudge("en").judge({"state": state + "\n```"}).ok is False


def test_hygiene_judge_skips_a_non_text_state() -> None:
    """state 不是文本就放过：那不是「格式残留」，是上游接线错了。"""
    verdict = judges_mod.HygieneJudge("zh").judge({"state": {"not": "text"}})
    assert verdict.ok is True
    assert "不是文本" in verdict.reason


# ---------- 框架 ----------


def test_frame_judge_accepts_the_code_rendered_frame() -> None:
    """输入：渲染器拼出来的 state。输出：Verdict。预期行为：通过，且不是跳过。"""
    verdict = judges_mod.FrameJudge("zh").judge({"state": state_for()}, spec_for(CASE))
    assert verdict.ok is True and verdict.skipped is False
    assert "一致" in verdict.reason


def test_frame_judge_rejects_a_missing_catalog_row() -> None:
    """输入：候选表被删掉一行。输出：Verdict。预期行为：判不通过并说明解析出几行。"""
    loc = locales.load("zh")
    state = render.to_state({"request": "x"}, CASE, loc)
    broken = state.replace(
        f"{render.catalog_line(CATALOG[2], loc)}\n", ""
    )
    verdict = judges_mod.FrameJudge("zh").judge({"state": broken}, spec_for(CASE))
    assert verdict.ok is False
    assert "候选表" in verdict.reason


def test_frame_judge_rejects_needs_that_do_not_match_the_tags() -> None:
    """输入：门槛行被换成别的特征算出来的。输出：Verdict。

    预期行为：判不通过——门槛与注入特征对不上说明渲染挂错了 tags，标签会跟着一起错。
    """
    loc = locales.load("zh")
    state = render.to_state({"request": "x"}, CASE, loc)
    other = TagSet("chat", "small", "batch", "normal", "all_ok")
    broken = state.replace(
        render.needs_line(CASE.needs, loc), render.needs_line(other.needs, loc)
    )
    verdict = judges_mod.FrameJudge("zh").judge({"state": broken}, spec_for(CASE))
    assert verdict.ok is False
    assert "门槛行" in verdict.reason


def test_frame_judge_skips_without_spec_or_text() -> None:
    """没有 spec、或 state 不是文本，都记成跳过（没判成），不是数据不合格。"""
    no_spec = judges_mod.FrameJudge("zh").judge({"state": state_for()})
    no_text = judges_mod.FrameJudge("zh").judge({"state": 42}, spec_for(CASE))
    assert no_spec.ok is True and no_spec.skipped is True
    assert no_text.ok is True and no_text.skipped is True


# ---------- 反解 ----------


def test_build_verify_prompt_carries_the_state_and_the_options() -> None:
    """反解提示词要把 state 原样带上，并列出与数据集一致的任务类型选项。"""
    prompt = judges_mod.build_verify_prompt(state_for(), locales.load("zh"))
    assert "帮我把这段函数改成异步的" in prompt
    for name in features.TASK_KINDS:
        assert name in prompt


def test_trace_judge_passes_when_reverse_parse_agrees(fake_backend: GenerationBackend) -> None:
    """反解与注入一致 → 通过，并在 raw 里留下反解结果。"""
    judge = judges_mod.TraceJudge(backend=fake_backend(), locale="zh")
    verdict = judge.judge({"state": state_for()}, spec_for(CASE))
    assert verdict.ok is True and verdict.skipped is False
    assert verdict.raw["traced"]["kind"] == "code_gen"


def test_trace_judge_rejects_when_reverse_parse_disagrees(
    fake_backend: GenerationBackend,
) -> None:
    """反解说成另一个任务类型 → 不通过，理由里同时列出注入值与反解值。"""
    judge = judges_mod.TraceJudge(backend=fake_backend(trace_lies=True), locale="zh")
    verdict = judge.judge({"state": state_for()}, spec_for(CASE))
    assert verdict.ok is False
    assert "code_gen" in verdict.reason and "long_summary" in verdict.reason


def test_trace_judge_skips_without_spec_or_backend() -> None:
    """没有 spec、或没有后端，都记成跳过（没判成），不是数据不合格。"""
    no_spec = judges_mod.TraceJudge(backend=None, locale="zh").judge({"state": "x"})
    no_backend = judges_mod.TraceJudge(backend=None, locale="zh").judge(
        {"state": "x"}, spec_for(CASE)
    )
    assert no_spec.ok is True and no_spec.skipped is True
    assert no_backend.ok is True and no_backend.skipped is True


def test_trace_judge_skips_an_endpoint_failure(fake_backend: GenerationBackend) -> None:
    """端点故障只算「没判成」——记成「不通过」会把端点抖动写成数据质量问题。"""
    judge = judges_mod.TraceJudge(
        backend=fake_backend(trace_raises=TransportError("网关 504")), locale="zh"
    )
    verdict = judge.judge({"state": state_for()}, spec_for(CASE))
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
        judge.judge({"state": state_for()}, spec_for(CASE))


def test_trace_judge_skips_when_the_draft_misses_fields(
    fake_backend: GenerationBackend,
) -> None:
    """反解草稿缺字段 → 判不了，记成跳过并点名缺了哪个字段。"""
    judge = judges_mod.TraceJudge(backend=fake_backend(trace_data={}), locale="zh")
    verdict = judge.judge({"state": state_for()}, spec_for(CASE))
    assert verdict.ok is True and verdict.skipped is True
    assert "缺字段" in verdict.reason and "kind" in verdict.reason


def test_trace_judge_skips_an_out_of_range_value(fake_backend: GenerationBackend) -> None:
    """反解给了一个不在取值域里的值 → 与注入值不等，判不通过（不是跳过）。"""
    judge = judges_mod.TraceJudge(
        backend=fake_backend(trace_data={"kind": "not_a_kind"}), locale="zh"
    )
    verdict = judge.judge({"state": state_for()}, spec_for(CASE))
    assert verdict.ok is False
    assert "not_a_kind" in verdict.reason


# ---------- 选项反解 ----------
