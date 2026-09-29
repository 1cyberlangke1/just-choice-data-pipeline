"""语言文件测试：结构不变量（抬头、节指代、档位、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from workflows.citation import features, judges, locales, questions, render


def test_both_languages_exist() -> None:
    """输入：无。输出：无。预期行为：zh / en 都在（少一个就是漏分发）。"""
    assert locales.available() == ["en", "zh"]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_marks_are_non_empty_and_pairwise_non_prefix(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：五种抬头两两不互为前缀——`【第 1 节】` 与 `【第 1 节附注】` 那种关系会让解析
    先命中短的那个，整段 state 当场错位。
    """
    loc = locales.load(lang)
    state = loc["state"]
    marks = [
        str(state["doc_mark"]),
        *[str(state["sec_mark"]).format(n=index + 1) for index in range(features.SECTIONS)],
        str(state["cite_mark"]),
        str(state["claim_mark"]),
    ]
    assert all(marks)
    assert len(set(marks)) == len(marks)
    for left in marks:
        for right in marks:
            if left != right:
                assert not right.startswith(left), f"{left!r} 是 {right!r} 的前缀"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_section_references_are_filled_and_one_based(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：`sec_ref` 与 `claim_value` 填完之后不含花括号，且说的是同一件事（第几节）。
    """
    loc = locales.load(lang)
    for index in range(features.SECTIONS):
        for key in ("sec_ref", "claim_value"):
            text = str(loc["state"][key]).format(n=index + 1)
            assert "{" not in text and "}" not in text
            assert str(index + 1) in text
    assert str(loc["state"]["sec_ref"]) == str(loc["state"]["claim_value"])


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_verdict_options_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：三种结论的文案与 `features.VERDICTS` 一一对应。"""
    options = locales.load(lang)["fam"]["verdict"]["opt"]
    assert isinstance(options, dict)
    assert sorted(options) == sorted(features.VERDICTS)
    assert all(str(text).strip() for text in options.values())
    assert len(set(str(text) for text in options.values())) == len(options)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：四族的问法池非空；两族 noul 有 true/false 判据。"""
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert isinstance(spec["q"], list) and spec["q"], f"{family} 没有问法"
    for family in ("in_doc", "in_claimed"):
        assert str(loc["fam"][family]["criteria_true"]).strip()
        assert str(loc["fam"][family]["criteria_false"]).strip()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_placeholders_are_all_filled(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：提示词渲染完之后**一个花括号都不剩**——漏填会在 `format` 上抛错，而多写的
    花括号会原样漏给模型，让它以为那是要填的槽。
    """
    prompt = render.build_prompt(locales.load(lang))
    assert "{" not in prompt and "}" not in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_verify_prompt_has_no_unfilled_slots(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：反解提示词也不留花括号。"""
    prompt = judges.build_trace_prompt("[state]", locales.load(lang))
    assert "{" not in prompt and "}" not in prompt


def test_pick_wraps_around() -> None:
    """输入：越界的问法编号。输出：无。预期行为：回绕（不抛异常）。"""
    assert locales.pick(["a", "b"], 3) == "b"


def test_locale_dict_is_json_shaped() -> None:
    """输入：一种语言。输出：无。预期行为：抬头与节指代都是非空字符串。"""
    loc: dict[str, JsonValue] = locales.load("zh")
    assert str(loc["state"]["doc_mark"])
    assert "{n}" in str(loc["state"]["sec_mark"])
    assert "{n}" in str(loc["state"]["sec_ref"])
