"""检索结果判读的判别器测试：三个判别器都要能放过好数据、拦住坏数据。"""

from __future__ import annotations

import typing

from jc.jsontypes import JsonValue
from jc import llmjudge

from workflows.search import judges, locales, render
from workflows.search.features import TagSet
from workflows.search.spec import SearchSpec

from tests.workflows.search.conftest import TEXTS, FakeBackend

TAGS = TagSet(
    query_kind="informational",
    topicality="on_topic",
    spam="clean",
    authority="community",
    freshness="fresh",
)
GOOD = dict(TEXTS[("informational", "on_topic", "clean")])


def sample(state: str) -> dict[str, JsonValue]:
    """输入：state。输出：最小可判样本。"""
    return {"state": state}


def accepted_values(annotation: object) -> set[str]:
    """输入：草稿字段的注解。输出：它收得下的那组取值（字符串形式）。

    预期行为：`Literal[...]` 取它的实参；`bool` 按提示词里那两行选项名报 `true` / `false`。
    认不出的注解直接报错——尺子认不出就该红，不能悄悄放行。
    """
    if annotation is bool:
        return {"true", "false"}
    args = typing.get_args(annotation)
    if args:
        return {str(item) for item in args}
    raise AssertionError(f"认不出这个注解该收哪些取值：{annotation!r}")


def test_trace_option_names_are_the_draft_values() -> None:
    """反解提示词列出来的选项名，必须正是草稿模型收得下的那组取值。

    输入：每个语言的 `trace_options` 与 `TraceDraft`。输出：无。
    预期行为：两套词必须**逐字**对上。这条盯的是真机上的那类错：`spam` 那道题在数据集里是
    noul，选项名是 `true` / `false`，而草稿要的是 `clean` / `spam`——提示词照 criteria 列，
    模型就得自己翻译，实测 6 条里 5 条翻译错（把 `clean` 读成 `spam`，好数据被拒）。
    """
    for locale in locales.available():
        loc = locales.load(locale)
        for family in judges.TRACE_FAMILIES:
            names = {name for name, _ in llmjudge.trace_options(family, loc)}
            field = judges.TraceDraft.model_fields[family]
            assert names == accepted_values(field.annotation), (
                f"{locale} 的 {family} 那一题：提示词列的是 {sorted(names)}，"
                f"草稿收的是 {sorted(accepted_values(field.annotation))}"
            )


def test_trace_prompt_shows_the_draft_values() -> None:
    """输入：一份反解提示词。输出：无。预期行为：提示词里出现的是取值本身，不是 criteria key。"""
    prompt = judges.build_trace_prompt(str(TEXTS[("informational", "on_topic", "clean")]["title"]), locales.load("en"))
    assert "   - spam: " in prompt
    assert "   - clean: " in prompt
    assert "   - true: " not in prompt
    assert "   - false: " not in prompt


def spec() -> SearchSpec:
    """输入：无。输出：与 `TAGS` 对齐的真值规格。"""
    return SearchSpec(tags=TAGS, phrasing=0, split="train", seq=0)


def state_of(data: dict[str, str] | None = None) -> str:
    """输入：可选草稿覆盖。输出：拼好的 state。"""
    loc = locales.load("zh")
    return render.to_state({**GOOD, **(data or {})}, TAGS, loc)


def test_hygiene_catches_leftovers_and_multiline() -> None:
    """输入：带围栏 / 多行 / 空段的 state。输出：无。预期行为：都判不通过。"""
    judge = judges.HygieneJudge("zh")
    assert not judge.judge(sample(state_of().replace("【标题】", "```\n【标题】"))).ok
    assert not judge.judge(sample(state_of({"title": "第一行\n第二行"}))).ok
    assert not judge.judge(sample(state_of({"title": "   "}))).ok


def test_frame_catches_a_wrong_source_row() -> None:
    """输入：来源行被改成别的取值。输出：无。预期行为：判不通过。"""
    broken = state_of().replace("- community | fresh", "- official | stale")
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_frame_catches_a_missing_rule_table() -> None:
    """输入：规则表被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    head = state_of().split(locales.header(loc, loc["meta"]["rules_label"]))[0]
    assert not judges.FrameJudge("zh").judge(sample(head), spec()).ok


def test_trace_catches_a_misread_query_kind(fake_backend: type[FakeBackend]) -> None:
    """输入：反解说反话的假后端。输出：无。预期行为：判不通过。"""
    verdict = judges.TraceJudge(fake_backend(trace_lies_kind=True), "zh").judge(
        sample(state_of()), spec()
    )
    assert not verdict.ok


def test_trace_catches_a_misread_spam_flag(fake_backend: type[FakeBackend]) -> None:
    """输入：反解把「是不是垃圾」说反。输出：无。预期行为：判不通过。"""
    verdict = judges.TraceJudge(fake_backend(trace_lies_spam=True), "zh").judge(
        sample(state_of()), spec()
    )
    assert not verdict.ok
