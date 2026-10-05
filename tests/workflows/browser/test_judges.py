"""浏览器 / GUI 操作的判别器测试：三个判别器都要能放过好数据、拦住坏数据。"""

from __future__ import annotations

import pytest

from jc.backend.types import AllProvidersDown, TransportError
from jc.jsontypes import JsonValue

from workflows.browser import features, judges, locales, render
from workflows.browser.features import TagSet
from workflows.browser.spec import BrowserSpec

from tests.workflows.browser.conftest import FakeBackend

TAGS = TagSet(intent="submit", state="ready", sensitive="yes", decoy="same_role", screen="form")
LABEL = features.wanted_label(TAGS)
GOOD = {"request": f"确认没问题了，点一下{LABEL}。"}


def sample(state: object) -> dict[str, JsonValue]:
    """输入：state。输出：最小可判样本。"""
    return {"state": state}


def spec(tags: TagSet = TAGS) -> BrowserSpec:
    """输入：可选特征。输出：与它对齐的真值规格。"""
    return BrowserSpec(tags=tags, phrasing=0, split="train", seq=0)


def state_of(data: dict[str, str] | None = None, tags: TagSet = TAGS) -> str:
    """输入：可选草稿覆盖、特征。输出：拼好的 state。"""
    return render.to_state({**GOOD, **(data or {})}, tags, locales.load("zh"))


def test_hygiene_catches_html_line_breaks() -> None:
    """输入：那句话里混进 `<br>`。输出：无。预期行为：判不通过。"""
    assert not judges.HygieneJudge("zh").judge(sample(state_of({"request": f"上一句<br>下一句{LABEL}"}))).ok


def test_hygiene_catches_a_repeated_section_name() -> None:
    """输入：那句话里又写了一遍段落名。输出：无。预期行为：判不通过。"""
    broken = state_of({"request": f"用户请求：{GOOD['request']}"})
    assert not judges.HygieneJudge("zh").judge(sample(broken)).ok


def test_hygiene_lets_a_non_text_state_through() -> None:
    """输入：state 不是文本。输出：无。预期行为：不判、也不炸（当没查到东西）。"""
    verdict = judges.HygieneJudge("zh").judge(sample(123))
    assert verdict.ok


def test_frame_passes_the_injected_tables() -> None:
    """输入：与特征一致的 state。输出：无。预期行为：通过。"""
    verdict = judges.FrameJudge("zh").judge(sample(state_of()), spec())
    assert verdict.ok


def test_frame_catches_a_wrong_element_table() -> None:
    """输入：元素表某一行被改成别的状态。输出：无。预期行为：判不通过。"""
    broken = state_of().replace("| ready", "| hidden", 1)
    verdict = judges.FrameJudge("zh").judge(sample(broken), spec())
    assert not verdict.ok
    assert "元素表" in verdict.reason


def test_frame_catches_a_wrong_table_for_another_spec() -> None:
    """输入：拿另一组特征的 spec 去判。输出：无。预期行为：判不通过（渲染挂错 tags 就长这样）。"""
    other = TagSet(intent="open", state="ready", sensitive="no", decoy="none", screen="list")
    assert not judges.FrameJudge("zh").judge(sample(state_of()), spec(other)).ok


def test_frame_catches_a_missing_gate_table() -> None:
    """输入：门控表被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    head = state_of().split(locales.header(loc, loc["meta"]["gates_label"]))[0]
    verdict = judges.FrameJudge("zh").judge(sample(head), spec())
    assert not verdict.ok
    assert "门控表" in verdict.reason


def test_frame_catches_a_tampered_gate_text() -> None:
    """输入：门控表某一行的说明被改掉。输出：无。预期行为：判不通过。"""
    broken = state_of().replace("- blocked |", "- blocked | 随便什么")
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_frame_catches_a_tampered_criteria_row() -> None:
    """输入：意图表某一行被改掉。输出：无。预期行为：判不通过。"""
    broken = state_of().replace("- fill | field | type | 1 | 2", "- fill | field | type | 1 | 3")
    verdict = judges.FrameJudge("zh").judge(sample(broken), spec())
    assert not verdict.ok
    assert "意图表" in verdict.reason


def test_frame_skips_without_spec_or_text() -> None:
    """输入：没有真值规格 / state 不是文本。输出：无。预期行为：都记成跳过。"""
    assert judges.FrameJudge("zh").judge(sample(state_of())).skipped
    assert judges.FrameJudge("zh").judge(sample(123), spec()).skipped


def test_trace_passes_when_the_model_agrees(fake_backend: type[FakeBackend]) -> None:
    """输入：从不撒谎的假后端。输出：无。预期行为：通过。"""
    verdict = judges.TraceJudge(fake_backend(), "zh").judge(sample(state_of()), spec())
    assert verdict.ok and not verdict.skipped


#: 反解三种「说错一项」的写法：假后端开关 → 判决理由里必须点名的字段。
#: 三条各自验一个字段（意图 / 行号 / 元素名），参数化是为了加一个字段只加一行。
MISREAD_FIELDS: tuple[tuple[str, str], ...] = (
    ("trace_lies_intent", "intent"),
    ("trace_lies_row", "row"),
    ("trace_lies_label", "label"),
)


@pytest.mark.parametrize("switch, field", MISREAD_FIELDS)
def test_trace_catches_a_misread_field(
    fake_backend: type[FakeBackend], switch: str, field: str
) -> None:
    """输入：反解把某一项说错的假后端。输出：无。

    预期行为：判不通过，而且理由里点名是哪个字段对不上——「抓到了」但说不出抓到什么，
    等于判别器只能靠整条不通过来报警。
    """
    verdict = judges.TraceJudge(fake_backend(**{switch: True}), "zh").judge(
        sample(state_of()), spec()
    )
    assert not verdict.ok
    assert field in verdict.reason


def test_trace_marks_endpoint_failure_as_skipped(fake_backend: type[FakeBackend]) -> None:
    """输入：反解时端点故障。输出：无。

    预期行为：记成**跳过**而不是不通过——端点抖动不是数据质量问题（PLAN 3.1.4）。
    """
    backend = fake_backend(trace_raises=TransportError("boom"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert verdict.ok and verdict.skipped


def test_trace_reraises_a_fatal_backend_error(fake_backend: type[FakeBackend]) -> None:
    """输入：全部 provider 都挂了。输出：无。

    预期行为：直接往上抛——那是整批该停下的唯一条件，吞掉它就会产出一整个「全被拒」的
    假数据集（PLAN 3.1.4）。
    """
    backend = fake_backend(trace_raises=AllProvidersDown("全挂了"))
    with pytest.raises(AllProvidersDown):
        judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())


def test_trace_compares_the_label_after_stripping() -> None:
    """输入：反解回来的元素名前后带空白。输出：无。

    预期行为：算通过——元素名是抄回来的，抄的时候多一个空格不该判成数据错了。
    """
    from jc.backend.types import Draft  # noqa: PLC0415  （只在这个用例里用）

    class Padded:
        """把 label 前后各加一个空格的后端。"""

        name = "padded"

        def generate(self, task: object) -> Draft:
            """输入：任务。输出：把真值抄回来但带空格的草稿。"""
            tags = TagSet(**task.spec["tags"])
            return Draft(
                data={
                    "intent": tags.intent,
                    "row": features.wanted_index(tags),
                    "label": f" {features.wanted_label(tags)} ",
                },
                raw_text="<fake>",
                backend="fake",
                provider="fake",
                model="fake-model",
                mode="json_schema",
                usage={},
            )

    verdict = judges.TraceJudge(Padded(), "zh").judge(sample(state_of()), spec())
    assert verdict.ok and not verdict.skipped


def test_build_trace_prompt_asks_all_three_questions_with_the_shared_options() -> None:
    """输入：一段 state。输出：无。

    预期行为：反解提示词把意图那一族连同选项原样再问一遍（选项与数据集里的一致，
    否则「答对了」与「标签对了」不是一回事），并追加上行号与元素名两个问题。
    """
    loc = locales.load("zh")
    prompt = judges.build_trace_prompt(state_of(), loc)
    assert state_of() in prompt
    assert judges.TRACE_FAMILIES == ("intent", "row", "label")
    assert str(loc["fam"]["intent"]["q"][0]) in prompt
    for name, desc in loc["fam"]["intent"]["criteria"].items():
        assert str(name) in prompt
        assert str(desc) in prompt
    assert str(loc["verify"]["q_row"]) in prompt
    assert str(loc["verify"]["q_label"]) in prompt
