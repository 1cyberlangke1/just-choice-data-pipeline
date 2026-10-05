"""打分与评分的判别器测试：三个判别器都要能放过好数据、拦住坏数据。"""

from __future__ import annotations

import pytest

from jc.judge import leftover_in
from jc.jsontypes import JsonValue

from workflows.rubric import features, judges, locales, render
from workflows.rubric.features import TagSet
from workflows.rubric.spec import RubricSpec

from tests.workflows.rubric.conftest import FakeBackend, body_for

TAGS = TagSet(placeholders="yes", citations="no", length="short", form="prose", data="none")
GOOD = {"body": body_for(TAGS.as_dict())}


def sample(state: object) -> dict[str, JsonValue]:
    """输入：state。输出：最小可判样本。"""
    return {"state": state}


def spec(tags: TagSet = TAGS) -> RubricSpec:
    """输入：可选特征覆盖。输出：与之对齐的真值规格。"""
    return RubricSpec(tags=tags, phrasing=0, split="train", seq=0)


def state_of(tags: TagSet = TAGS, data: dict[str, str] | None = None) -> str:
    """输入：可选特征、可选草稿覆盖。输出：拼好的 state。"""
    loc = locales.load("zh")
    body = body_for(tags.as_dict())
    return render.to_state({**{"body": body}, **(data or {})}, tags, loc)


def test_hygiene_passes_a_clean_state() -> None:
    """输入：干净 state。输出：无。预期行为：通过，且不记成「没判成」。"""
    verdict = judges.HygieneJudge("zh").judge(sample(state_of()), spec())
    assert verdict.ok and not verdict.skipped


def test_hygiene_catches_leftovers_and_empty_body() -> None:
    """输入：带围栏行 / 空材料的 state。输出：无。预期行为：都判不通过。"""
    judge = judges.HygieneJudge("zh")
    fenced = state_of().replace("【待评材料】", "```\n【待评材料】")
    assert not judge.judge(sample(fenced), spec()).ok
    assert "残留" in judge.judge(sample(fenced), spec()).reason
    tagged = state_of().replace("TODO: fill in", "<br>TODO: fill in")
    assert not judge.judge(sample(tagged), spec()).ok
    empty = "\n".join(
        line for line in state_of().splitlines() if line not in GOOD["body"].splitlines()
    )
    verdict = judge.judge(sample(empty), spec())
    assert not verdict.ok
    assert "空" in verdict.reason


def test_hygiene_catches_a_shape_mismatch() -> None:
    """输入：材料是「分条」、规格却写着「行文」。输出：无。

    预期行为：判不通过——材料形状与注入特征打架时，标签会跟着一起错。
    """
    bulleted = TagSet(placeholders="no", citations="no", length="short", form="bulleted", data="none")
    verdict = judges.HygieneJudge("zh").judge(sample(state_of(bulleted)), spec())
    assert not verdict.ok
    assert "分条" in verdict.reason


def test_hygiene_rejects_a_non_text_state() -> None:
    """输入：非文本 state。输出：无。预期行为：判不通过，不抛异常。"""
    verdict = judges.HygieneJudge("zh").judge(sample({"not": "text"}), spec())
    assert not verdict.ok
    assert "不是文本" in verdict.reason


def test_hygiene_still_checks_shape_without_a_spec() -> None:
    """输入：没有 spec 的干净 state。输出：无。

    预期行为：没有 spec 就查不了形状，但残留与空材料这两条照样查——判别器可能被单独
    拿去跑产物。
    """
    verdict = judges.HygieneJudge("zh").judge(sample(state_of()), None)
    assert verdict.ok and not verdict.skipped
    fenced = state_of().replace("【待评材料】", "```\n【待评材料】")
    assert not judges.HygieneJudge("zh").judge(sample(fenced), None).ok


def test_frame_catches_a_score_row_that_prints_a_number_for_a_body_dimension() -> None:
    """输入：评分单把「按材料数」换成了数字。输出：无。

    预期行为：判不通过——那两行只能从材料里数出来，印数字等于把材料变成装饰。
    """
    loc = locales.load("zh")
    mark = str(loc["meta"]["count_mark"])
    broken = state_of().replace(f"| {mark}", "| 4")
    verdict = judges.FrameJudge("zh").judge(sample(broken), spec())
    assert not verdict.ok
    assert "评分单" in verdict.reason


@pytest.mark.parametrize("switch", ["trace_lies_placeholders", "trace_lies_citations"])
def test_trace_catches_a_misread_dimension(switch: str) -> None:
    """输入：反解说反话的假后端。输出：无。预期行为：判不通过，理由里两边的值都在。"""
    backend = FakeBackend(**{switch: True})
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert not verdict.ok and not verdict.skipped
    assert "注入" in verdict.reason and "反解成" in verdict.reason


def test_trace_skips_without_spec_or_backend() -> None:
    """输入：没有 spec / 没接后端的反解判别器。输出：无。预期行为：都记成「没判成」。"""
    assert judges.TraceJudge(FakeBackend(), "zh").judge(sample(state_of()), None).skipped
    verdict = judges.TraceJudge(None, "zh").judge(sample(state_of()), spec())
    assert verdict.ok and verdict.skipped
    assert "没有后端" in verdict.reason


def test_trace_prompt_repeats_the_atomic_questions() -> None:
    """输入：一段 state。输出：提示词。预期行为：两维原子题与它们的选项都在提示词里。"""
    loc = locales.load("zh")
    prompt = judges.build_trace_prompt(state_of(), loc)
    assert str(loc["verify"]["intro"]) in prompt
    for dim in judges.TRACE_FAMILIES:
        assert str(loc["trace"][dim]["q"]) in prompt
        for name, desc in loc["trace"][dim]["criteria"].items():
            assert str(name) in prompt and str(desc) in prompt


def test_trace_options_are_named_after_the_feature_values() -> None:
    """输入：两个语言的反解选项。输出：无。

    预期行为：选项名就是 `features` 里的取值（no / yes）——提示词里写成 true / false
    而草稿模型要求 no / yes，等于把「答对了」与「标签对了」拆成两件事。
    """
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        for dim in judges.TRACE_FAMILIES:
            assert set(loc["trace"][dim]["criteria"]) == set(features.DIMS[dim])
    holders = judges.TraceDraft.model_fields["placeholders"].annotation
    assert set(holders.__args__) == set(features.PLACEHOLDERS)


def test_leftover_lines_are_the_fence_variants() -> None:
    """输入：残留清单。输出：无。预期行为：围栏的常见变体都在，HTML 换行标签也在。"""
    assert "```" in judges.LEFTOVER_LINES
    assert "```json" in judges.LEFTOVER_LINES
    assert leftover_in("```\n正文", judges.LEFTOVER_LINES, judges.LEFTOVER_TAGS) == "```"
    assert leftover_in("正文<br>更多", judges.LEFTOVER_LINES, judges.LEFTOVER_TAGS) == "<br>"
    assert leftover_in("干净的正文", judges.LEFTOVER_LINES, judges.LEFTOVER_TAGS) == ""
