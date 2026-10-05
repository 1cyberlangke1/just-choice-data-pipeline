"""改动审查的判别器测试：三个判别器都要能放过好数据、拦住坏数据。"""

from __future__ import annotations

import pytest

from jc.backend.types import AllProvidersDown, TransportError
from jc.jsontypes import JsonValue
from jc import llmjudge

from workflows.review import judges, locales, render
from workflows.review.features import TagSet
from workflows.review.spec import ReviewSpec

from tests.workflows.review.conftest import FakeBackend, body_for

TAGS = TagSet(secret="yes", debug="no", file_kind="source", change_kind="modified", size="small")
GOOD = {"body": body_for(TAGS.as_dict())}


def sample(state: object) -> dict[str, JsonValue]:
    """输入：state。输出：最小可判样本。"""
    return {"state": state}


def spec() -> ReviewSpec:
    """输入：无。输出：与 `TAGS` 对齐的真值规格。"""
    return ReviewSpec(tags=TAGS, phrasing=0, split="train", seq=0)


def state_of(data: dict[str, str] | None = None) -> str:
    """输入：可选草稿覆盖。输出：拼好的 state。"""
    loc = locales.load("zh")
    return render.to_state({**GOOD, **(data or {})}, TAGS, loc)


def test_hygiene_catches_leftovers_and_empty_body() -> None:
    """输入：带围栏行 / 空正文的 state。输出：无。预期行为：都判不通过。"""
    judge = judges.HygieneJudge("zh")
    assert not judge.judge(sample(state_of().replace("【改动内容】", "```\n【改动内容】"))).ok
    assert not judge.judge(sample(state_of({"body": "   "}))).ok


def test_hygiene_catches_a_missing_line_prefix() -> None:
    """输入：正文里有一行不带前缀的 state。输出：无。预期行为：判不通过。

    预期行为（续）：这条**绕开渲染期**手拼——`to_state` 会把「上下文行没写空格前缀」
    补成规范形态（范式 §9.19），所以判别器要兜的是**别处来的、没规范化的** state。
    """
    broken = (
        "【改动内容】\n"
        "+ x = 1\n"
        "y = 2\n"
        "【改动清单】（文件类型 | 变更类型 | 规模）\n"
        "- source | modified | small"
    )
    verdict = judges.HygieneJudge("zh").judge(sample(broken))
    assert not verdict.ok
    assert "前缀" in verdict.reason


def test_hygiene_catches_a_line_count_mismatch() -> None:
    """输入：行数与清单里的规模对不上。输出：无。预期行为：判不通过。"""
    broken = state_of().replace("- source | modified | small", "- source | modified | large")
    verdict = judges.HygieneJudge("zh").judge(sample(broken))
    assert not verdict.ok
    assert "规模" in verdict.reason


def test_hygiene_catches_a_repeated_section_name() -> None:
    """输入：改动里又写了一遍段落名。输出：无。预期行为：判不通过。"""
    assert not judges.HygieneJudge("zh").judge(sample(state_of({"body": "改动内容：+ x = 1"}))).ok


def test_hygiene_catches_a_table_name_in_the_body() -> None:
    """输入：改动里写了表名。输出：无。预期行为：判不通过。"""
    broken = state_of({"body": "+ x = 1\n+ y = 2  # 见严重度表"})
    assert not judges.HygieneJudge("zh").judge(sample(broken)).ok


def test_hygiene_skips_non_text_state() -> None:
    """输入：非文本 state。输出：无。预期行为：不判它（`ok` 为真），不当成坏数据。"""
    verdict = judges.HygieneJudge("zh").judge(sample({"not": "text"}))
    assert verdict.ok
    assert "不是文本" in verdict.reason


def test_frame_passes_a_clean_state() -> None:
    """输入：干净 state。输出：无。预期行为：通过。"""
    verdict = judges.FrameJudge("zh").judge(sample(state_of()), spec())
    assert verdict.ok and not verdict.skipped


def test_frame_skips_without_spec_or_text() -> None:
    """输入：没有 spec / 非文本 state。输出：无。预期行为：都记成「没判成」。"""
    judge = judges.FrameJudge("zh")
    assert judge.judge(sample(state_of()), None).skipped
    assert judge.judge(sample({"not": "text"}), spec()).skipped


@pytest.mark.parametrize("switch", ["trace_lies_secret", "trace_lies_debug"])
def test_trace_catches_a_misread_dimension(switch: str) -> None:
    """输入：反解说反话的假后端。输出：无。预期行为：判不通过，理由里两边的值都在。"""
    backend = FakeBackend(**{switch: True})
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert not verdict.ok and not verdict.skipped
    assert "注入" in verdict.reason and "反解成" in verdict.reason


def test_trace_records_a_transport_failure_as_skipped() -> None:
    """输入：反解时端点故障。输出：无。

    预期行为：记成「没判成」——记成「不通过」会把端点抖动写成数据质量问题。
    """
    backend = FakeBackend(trace_raises=TransportError("端点抖动"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert verdict.ok and verdict.skipped
    assert "端点故障" in verdict.reason


def test_trace_reraises_a_fatal_failure() -> None:
    """输入：全部 provider 都挂了。输出：无。预期行为：整批该停下，异常往上抛。"""
    backend = FakeBackend(trace_raises=AllProvidersDown("全挂了"))
    with pytest.raises(AllProvidersDown):
        judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())


def test_trace_skips_without_spec() -> None:
    """输入：没有 spec。输出：无。预期行为：记成「没判成」。"""
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample(state_of()), None)
    assert verdict.ok and verdict.skipped


def test_trace_skips_without_a_backend() -> None:
    """输入：没接后端的反解判别器。输出：无。预期行为：记成「没判成」，不炸。"""
    verdict = judges.TraceJudge(None, "zh").judge(sample(state_of()), spec())
    assert verdict.ok and verdict.skipped
    assert "没有后端" in verdict.reason


def test_trace_prompt_repeats_the_atomic_questions() -> None:
    """输入：一段 state。输出：提示词。预期行为：两族原子题与它们的选项都在提示词里。

    选项名走 `jc.llmjudge.trace_options`——**列出来的是草稿收得下的那组取值**（`yes` / `no`），
    不是数据集那道 noul 题的 criteria key（`true` / `false`）。两套词摆在一起模型得自己翻译，
    真机上就翻译错过（拒稿理由全是「注入 yes，反解成 no」）。
    """
    loc = locales.load("zh")
    prompt = judges.build_trace_prompt(state_of(), loc)
    for family in judges.TRACE_FAMILIES:
        assert str(loc["fam"][family]["q"][0]) in prompt
        for name, desc in llmjudge.trace_options(family, loc):
            assert str(name) in prompt and str(desc) in prompt
    assert "   - true：" not in prompt
    assert "   - false：" not in prompt
