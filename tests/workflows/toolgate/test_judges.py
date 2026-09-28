"""工具调用门控判别器的测试：卫生、框架、反解三关各自的通过 / 拦下 / 跳过。"""

from __future__ import annotations

import pytest

from jc.backend.types import AllProvidersDown, GenerationBackend, TransportError
from workflows.toolgate import features, judges, locales, render
from workflows.toolgate.features import TagSet
from workflows.toolgate.spec import ToolGateSpec

from tests.workflows.toolgate.conftest import FakeBackend

CASE = TagSet(tool="http_request", reach="external", payload="benign", policy="strict", approval="absent")
INTENT = "调一下库存服务的查询接口，看看还有多少货"
LOCALE = "zh"


def spec_of(tags: TagSet = CASE) -> ToolGateSpec:
    """输入：特征。输出：一条真值规格。预期行为：序号固定，便于断言。"""
    return ToolGateSpec(tags=tags, phrasing=0, split="train", seq=3)


def state_of(tags: TagSet = CASE, intent: str = INTENT) -> str:
    """输入：特征、意图。输出：渲染好的 state。"""
    return render.to_state({"intent": intent}, tags, locales.load(LOCALE))


def test_hygiene_judge_accepts_a_clean_state() -> None:
    """输入：干净 state。输出：无。预期行为：通过，且理由说清「单行无残留」。"""
    verdict = judges.HygieneJudge(LOCALE).judge({"state": state_of()})
    assert verdict.ok and not verdict.skipped


@pytest.mark.parametrize(
    "residue",
    [
        "```",
        "```json",
        "<br>",
        "<br />",
    ],
)
def test_hygiene_judge_rejects_format_residue(residue: str) -> None:
    """输入：state 里混进围栏 / HTML 标签。输出：无。预期行为：不通过，理由里点名残留。"""
    state = state_of() + f"\n{residue}\n"
    verdict = judges.HygieneJudge(LOCALE).judge({"state": state})
    assert not verdict.ok
    assert residue in verdict.reason


def test_hygiene_judge_flags_a_multi_line_intent() -> None:
    """输入：意图写成两行。输出：无。预期行为：不通过，理由点名「不止一行」。"""
    verdict = judges.HygieneJudge(LOCALE).judge({"state": state_of(intent=f"{INTENT}\n（顺手记一下）")})
    assert not verdict.ok
    assert "不止一行" in verdict.reason


def test_hygiene_judge_flags_an_empty_intent() -> None:
    """输入：意图空白。输出：无。预期行为：不通过，理由点名「空的」。"""
    verdict = judges.HygieneJudge(LOCALE).judge({"state": state_of(intent="   ")})
    assert not verdict.ok
    assert "空的" in verdict.reason


def test_hygiene_judge_flags_a_leaked_section_name() -> None:
    """输入：意图开头又写了一遍段落名。输出：无。预期行为：不通过。"""
    verdict = judges.HygieneJudge(LOCALE).judge({"state": state_of(intent="调用意图：读一下 config.yaml")})
    assert not verdict.ok
    assert "段落名" in verdict.reason


def test_hygiene_judge_reads_the_marker_from_the_locale() -> None:
    """输入：英文 state。输出：无。

    预期行为：认得出 `[Call intent]` 那一段，代码里没写死中文——换个语言照样认；
    而且拿另一种语言去判同一段文本会认不出来（段落名是按语言读的，不是靠猜）。
    """
    loc = locales.load("en")
    state = render.to_state({"intent": "read config.yaml and show me the timeout"}, CASE, loc)
    assert judges.HygieneJudge("en").judge({"state": state}).ok is True
    assert judges.HygieneJudge("en").judge({"state": state + "\n```"}).ok is False
    assert judges.HygieneJudge("zh").judge({"state": state}).ok is False


def test_hygiene_judge_skips_a_non_text_state() -> None:
    """输入：state 不是文本。输出：无。预期行为：跳过（不算不通过）。"""
    verdict = judges.HygieneJudge(LOCALE).judge({"state": 123})
    assert verdict.ok and not verdict.skipped


def test_frame_judge_accepts_the_code_rendered_frame() -> None:
    """输入：代码渲染的 state + 真值 spec。输出：无。预期行为：通过。"""
    verdict = judges.FrameJudge(LOCALE).judge({"state": state_of()}, spec_of())
    assert verdict.ok and not verdict.skipped


def test_frame_judge_rejects_a_state_row_that_does_not_match_the_tags() -> None:
    """输入：状态行被换成别的特征。输出：无。预期行为：不通过，理由点名状态行。"""
    other = TagSet(tool="http_request", reach="host", payload="benign", policy="strict", approval="granted")
    verdict = judges.FrameJudge(LOCALE).judge({"state": state_of(other)}, spec_of())
    assert not verdict.ok
    assert "状态行" in verdict.reason


def test_frame_judge_rejects_a_tampered_rule_table() -> None:
    """输入：策略表少了一行。输出：无。预期行为：不通过，理由点名策略表。"""
    loc = locales.load(LOCALE)
    state = state_of()
    broken = state.replace(render.rule_lines(loc)[0] + "\n", "")
    verdict = judges.FrameJudge(LOCALE).judge({"state": broken}, spec_of())
    assert not verdict.ok
    assert "策略表" in verdict.reason


def test_frame_judge_skips_without_spec_or_text() -> None:
    """输入：没有 spec / state 不是文本。输出：无。预期行为：一律跳过，不算不通过。"""
    judge = judges.FrameJudge(LOCALE)
    assert judge.judge({"state": state_of()}).skipped
    assert judge.judge({"state": 42}, spec_of()).skipped


def test_build_trace_prompt_carries_the_state_and_the_options() -> None:
    """输入：state。输出：无。预期行为：反解提示词里带上全文与两族的全部选项。"""
    loc = locales.load(LOCALE)
    prompt = judges.build_trace_prompt(state_of(), loc)
    assert INTENT in prompt
    assert "【请求全文】" in prompt
    for family in judges.TRACE_FAMILIES:
        for name in features.CHOICE_OPTIONS[family]:
            assert name in prompt


def test_trace_judge_passes_when_reverse_parse_agrees(fake_backend: type[FakeBackend]) -> None:
    """输入：从不撒谎的假后端。输出：无。预期行为：两维都对得上，通过。"""
    verdict = judges.TraceJudge(fake_backend(), LOCALE).judge({"state": state_of()}, spec_of())
    assert verdict.ok and not verdict.skipped


def test_trace_judge_rejects_a_misread_tool(fake_backend: type[FakeBackend]) -> None:
    """输入：反解时把工具说反。输出：无。预期行为：不通过，理由里列出注入值与反解值。"""
    judge = judges.TraceJudge(fake_backend(trace_lies=True), LOCALE)
    verdict = judge.judge({"state": state_of()}, spec_of())
    assert not verdict.ok and not verdict.skipped
    assert "tool" in verdict.reason and "注入" in verdict.reason


def test_trace_judge_rejects_a_misread_payload(fake_backend: type[FakeBackend]) -> None:
    """输入：反解时把内容性质说反。输出：无。预期行为：同样不通过——两维都得对。"""
    judge = judges.TraceJudge(fake_backend(trace_lies_payload=True), LOCALE)
    verdict = judge.judge({"state": state_of()}, spec_of())
    assert not verdict.ok and not verdict.skipped


def test_trace_judge_skips_without_spec_or_backend(fake_backend: type[FakeBackend]) -> None:
    """输入：没有 spec / 没有后端。输出：无。预期行为：跳过。"""
    assert judges.TraceJudge(fake_backend(), LOCALE).judge({"state": state_of()}).skipped
    assert judges.TraceJudge(None, LOCALE).judge({"state": state_of()}, spec_of()).skipped


def test_trace_judge_skips_an_endpoint_failure(fake_backend: type[FakeBackend]) -> None:
    """输入：反解时端点抛 TransportError。输出：无。

    预期行为：记成**跳过**而不是拒绝——端点抖动不是数据质量问题（PLAN 3.1.4）。
    """
    judge = judges.TraceJudge(fake_backend(trace_raises=TransportError("boom")), LOCALE)
    verdict = judge.judge({"state": state_of()}, spec_of())
    assert verdict.skipped and verdict.ok
    assert "端点故障" in verdict.reason


def test_trace_judge_reraises_when_every_provider_is_down(
    fake_backend: type[FakeBackend],
) -> None:
    """输入：反解时抛 `AllProvidersDown`。输出：无。

    预期行为：原样抛出——那是「整批该停下」，吞掉它会产出一整个「全被拒」的假数据集。
    """
    judge = judges.TraceJudge(fake_backend(trace_raises=AllProvidersDown("全挂了")), LOCALE)
    with pytest.raises(AllProvidersDown):
        judge.judge({"state": state_of()}, spec_of())


def test_trace_judge_skips_when_the_draft_misses_fields(
    fake_backend: type[FakeBackend],
) -> None:
    """输入：反解草稿缺一维。输出：无。预期行为：跳过（判不了，不等于答错）。"""
    judge = judges.TraceJudge(fake_backend(trace_data={"tool": "read_file"}), LOCALE)
    verdict = judge.judge({"state": state_of()}, spec_of())
    assert verdict.skipped and verdict.ok
    assert "缺字段" in verdict.reason


def test_trace_judge_rejects_a_value_outside_the_labels(
    fake_backend: type[FakeBackend],
) -> None:
    """输入：反解给出另一组合法取值。输出：无。预期行为：不通过（取值合法但与注入不符）。"""
    judge = judges.TraceJudge(
        fake_backend(trace_data={"tool": "send_message", "payload": "harmful"}), LOCALE
    )
    verdict = judge.judge({"state": state_of()}, spec_of())
    assert not verdict.ok and not verdict.skipped


def test_option_names_follows_the_artifact_order() -> None:
    """输入：一道打乱过的题。输出：无。预期行为：按产物里的顺序返回选项名。"""
    question = {"type": "choice", "criteria": {"deny": "a", "allow": "b"}}
    assert judges.option_names(question) == ["deny", "allow"]
    assert judges.option_names({"type": "noul"}) == []
