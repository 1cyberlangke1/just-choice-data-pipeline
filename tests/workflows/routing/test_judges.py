"""工单分诊的判别器测试：三个判别器都要能放过好数据、拦住坏数据。"""

from __future__ import annotations


from jc.jsontypes import JsonValue

from workflows.routing import features, judges, locales, render
from workflows.routing.features import TagSet
from workflows.routing.spec import RoutingSpec

from tests.workflows.routing.conftest import FakeBackend, body_for

TAGS = TagSet(issue="bug", urgency="deadline", tier="pro", compliance="none", area="storage")
GOOD = {"body": body_for(TAGS.as_dict())}


def sample(state: object) -> dict[str, JsonValue]:
    """输入：state。输出：最小可判样本。"""
    return {"state": state}


def spec() -> RoutingSpec:
    """输入：无。输出：与 `TAGS` 对齐的真值规格。"""
    return RoutingSpec(tags=TAGS, phrasing=0, split="train", seq=0)


def state_of(data: dict[str, str] | None = None) -> str:
    """输入：可选草稿覆盖。输出：拼好的 state。"""
    loc = locales.load("zh")
    return render.to_state({**GOOD, **(data or {})}, TAGS, loc)


def test_hygiene_catches_leftovers_multiline_and_empty() -> None:
    """输入：带围栏 / 多行 / 空正文的 state。输出：无。预期行为：都判不通过。"""
    judge = judges.HygieneJudge("zh")
    assert not judge.judge(sample(state_of().replace("【工单正文】", "```\n【工单正文】"))).ok
    assert not judge.judge(sample(state_of({"body": "第一行\n第二行"}))).ok
    assert not judge.judge(sample(state_of({"body": "   "}))).ok


def test_hygiene_catches_html_line_breaks() -> None:
    """输入：正文里混进 `<br>`。输出：无。预期行为：判不通过。"""
    assert not judges.HygieneJudge("zh").judge(sample(state_of({"body": "上一句<br>下一句"}))).ok


def test_hygiene_catches_a_repeated_section_name() -> None:
    """输入：正文里又写了一遍段落名。输出：无。预期行为：判不通过。"""
    broken = state_of({"body": "工单正文：刚才那句话"})
    assert not judges.HygieneJudge("zh").judge(sample(broken)).ok


def test_hygiene_catches_attribute_words_in_the_body() -> None:
    """输入：正文里写了属性词。输出：无。预期行为：判不通过。"""
    assert not judges.HygieneJudge("zh").judge(sample(state_of({"body": "我们是企业版客户"}))).ok


def test_frame_passes_the_injected_row_and_tables() -> None:
    """输入：与特征一致的 state。输出：无。预期行为：通过。"""
    verdict = judges.FrameJudge("zh").judge(sample(state_of()), spec())
    assert verdict.ok


def test_frame_catches_a_wrong_attr_row() -> None:
    """输入：属性行被改成别的取值。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    broken = state_of().replace(render.attr_lines(TAGS, loc)[0], "- storage | free | none")
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_frame_catches_a_missing_queue_table() -> None:
    """输入：派单表被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    head = state_of().split(locales.header(loc, loc["meta"]["queue_label"]))[0]
    assert not judges.FrameJudge("zh").judge(sample(head), spec()).ok


def test_frame_catches_a_tampered_priority_row() -> None:
    """输入：优先级表某一行的档位被改掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    first = render.priority_lines(loc)[0]
    broken = state_of().replace(first, first.replace("| 0 | 0 | 1", "| 3 | 3 | 3"))
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_frame_catches_a_missing_flag_table() -> None:
    """输入：附加规则被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    head = state_of().split(locales.header(loc, loc["meta"]["flags_label"]))[0]
    assert not judges.FrameJudge("zh").judge(sample(head), spec()).ok


def test_trace_catches_a_misread_issue(fake_backend: type[FakeBackend]) -> None:
    """输入：反解把「是什么问题」说反。输出：无。预期行为：判不通过。"""
    backend = fake_backend(trace_lies_issue=True)
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert not verdict.ok
    assert "issue" in verdict.reason


def test_trace_catches_a_misread_urgency(fake_backend: type[FakeBackend]) -> None:
    """输入：反解把「有多急」说反。输出：无。预期行为：判不通过。"""
    backend = fake_backend(trace_lies_urgency=True)
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert not verdict.ok
    assert "urgency" in verdict.reason


def test_trace_draft_literals_match_the_feature_tables() -> None:
    """输入：反解草稿的取值域。输出：无。

    预期行为：`TraceDraft` 的两维取值域与 `features.ISSUES` / `features.URGENCIES` 一致——
    对不上就会出现「模型答对了但字段类型不符」这种静默失败。
    """
    assert set(judges.TraceDraft.model_fields["issue"].annotation.__args__) == set(
        features.ISSUES
    )
    assert set(judges.TraceDraft.model_fields["urgency"].annotation.__args__) == set(
        features.URGENCIES
    )
