"""语言文件测试：结构不变量（抬头、口径、档位、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from workflows.rerank import features, locales, questions, render

MARKS = ("query_mark", "terms_mark", "cands_mark", "rule_mark")


def test_both_languages_exist() -> None:
    """输入：无。输出：无。预期行为：zh / en 都在（少一个就是漏分发）。"""
    assert locales.available() == ["en", "zh"]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_marks_are_non_empty_and_pairwise_non_prefix(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四个抬头标记两两不互为前缀——`【查询】` 与 `【查询词】` 那种关系会让解析
    先命中短的那个，整段 state 当场错位。
    """
    loc = locales.load(lang)
    marks = [str(loc["state"][key]) for key in MARKS]
    assert all(marks)
    for left in marks:
        for right in marks:
            if left != right:
                assert not right.startswith(left), f"{left!r} 是 {right!r} 的前缀"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_rule_rows_are_present_and_non_empty(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：排序口径至少一行、都不是空串。"""
    rows = render.rule_rows(locales.load(lang))
    assert len(rows) >= 1
    assert all(row.strip() for row in rows)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_count_tiers_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：`count` 的档位与 `features.RELEVANT` 一一对应。"""
    loc = locales.load(lang)
    options = loc["fam"]["count"]["opt"]
    assert isinstance(options, dict)
    assert sorted(options) == sorted(f"n{value}" for value in features.RELEVANT)
    assert all(str(text).strip() for text in options.values())


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：四族的问法池非空；choice 族有选项文案。"""
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert isinstance(spec["q"], list) and spec["q"], f"{family} 没有问法"
    for family in ("count", "order"):
        assert loc["fam"][family]["opt"]
    assert loc["fam"]["pair"]["criteria"]["true"] and loc["fam"]["pair"]["criteria"]["false"]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_placeholders_are_all_filled(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：提示词渲染完之后**一个花括号都不剩**——漏填会在 `format` 上抛错，而多写的
    花括号会原样漏给模型，让它以为那是要填的槽。
    """
    loc = locales.load(lang)
    for relevant in features.RELEVANT:
        for terms in features.TERM_COUNTS:
            plan = features.plan_for(features.RerankCase(terms, relevant, "first"), 0)
            prompt = render.build_prompt(plan, loc)
            assert "{" not in prompt and "}" not in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_candidate_reference_template_is_filled(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：`cand` 模板填完之后不含花括号。"""
    loc = locales.load(lang)
    for index in range(features.CANDIDATES):
        assert "{" not in render.cand_ref(loc, index)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_verify_prompt_has_no_unfilled_slots(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：反解提示词也不留花括号。"""
    from workflows.rerank import judges

    prompt = judges.build_trace_prompt("[state]", locales.load(lang))
    assert "{" not in prompt and "}" not in prompt


def test_pick_wraps_around() -> None:
    """输入：越界的问法编号。输出：无。预期行为：回绕（不抛异常）。"""
    assert locales.pick(["a", "b"], 3) == "b"


def test_min_term_chars_is_per_language() -> None:
    """输入：两种语言。输出：无。

    预期行为：中文的下限是 2、英文是 3——中文一个字就是词，拿英文的下限去卡中文会把
    `推荐`、`北京` 这种最正常的词判成「太短」（真机第一批白拒过四条）。
    """
    assert render.min_term_chars(locales.load("zh")) == 2
    assert render.min_term_chars(locales.load("en")) == 3


def test_min_term_chars_rejects_a_bad_knob() -> None:
    """输入：被改成字符串的下限。输出：无。预期行为：报人话，不静默当成 0。"""
    with pytest.raises(ValueError, match="min_term_chars"):
        render.min_term_chars({"meta": {"min_term_chars": "two"}})


@pytest.mark.parametrize(
    ("lang", "term", "ok"),
    [("zh", "推荐", True), ("zh", "的", False), ("en", "recommend", True), ("en", "of", False)],
)
def test_two_character_chinese_terms_pass_the_floor(lang: str, term: str, ok: bool) -> None:
    """输入：两种语言各两个词。输出：无。预期行为：中文两字词过、单字/虚词不过。"""
    loc = locales.load(lang)
    plan = features.RerankPlan(terms=2, best_pos=0, overlaps=(2, 1, 0, 0, 0))
    terms = [term, "zzzzz"]
    candidates = [f"{term} zzzzz here", f"{term} once", "nothing", "nothing two", "nothing three"]
    if ok:
        render.check_parts("query with " + term + " zzzzz", terms, candidates, plan, loc)
    else:
        with pytest.raises(ValueError, match="太短"):
            render.check_parts("query with " + term + " zzzzz", terms, candidates, plan, loc)


def test_locale_dict_is_json_shaped() -> None:
    """输入：一种语言。输出：无。预期行为：`meta.list_sep` 与 `meta.cand` 都是非空字符串。"""
    loc: dict[str, JsonValue] = locales.load("zh")
    assert str(loc["meta"]["list_sep"])
    assert "{n}" in str(loc["meta"]["cand"])
