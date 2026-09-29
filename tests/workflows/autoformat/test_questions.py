"""结构恢复的题族测试：四族题的题面、选项、金答案与问法变体。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion
from tests.workflows.autoformat import conftest as cf
from workflows.autoformat import features, locales, parse, questions, render
from workflows.autoformat.features import BLOCK_KINDS, CALLOUT_KINDS

LOC = locales.load("zh")
PLAIN = features.DocCase("build", 64, "plain", "na", "na")
CALLOUT = features.DocCase("build", 64, "callout", "warning", "na")
LIST = features.DocCase("build", 64, "list", "ordered", "na")
MIXED = features.DocCase("build", 64, "mixed", "tip", "vcs")


def families(items: list[LabeledQuestion]) -> list[str]:
    """输入：题目表。输出：题族名表（同族会出现多次，顺序即题序）。"""
    return [item.family for item in items]


def test_build_refuses_to_ask_before_rendering() -> None:
    """输入：还没回填文字的规格。输出：无。

    预期行为：报错——`texts` 是空的说明调用方漏了 `render`，猜一份答案只会掩盖接线错误。
    """
    with pytest.raises(ValueError, match="必须先渲染"):
        questions.build(cf.spec_for(PLAIN), LOC, 0)


def test_plain_document_asks_join_and_block_type_only() -> None:
    """输入：一份只有标题与段落的文档。输出：无。

    预期行为：没有提示框就不出 `callout_kind`、没有列表就不出 `list_ordered`——题族是
    「当前文档真的成立的那些」。
    """
    spec = cf.filled(PLAIN)
    items = questions.build(spec, LOC, 0)
    names = families(items)
    assert set(names) == {"join", "block_type"}
    assert names.count("block_type") == len(spec.plan.slots)
    assert names.count("join") == len(render.layout(spec.plan, spec.texts, LOC).asked_pairs(LOC))


@pytest.mark.parametrize("case", [PLAIN, CALLOUT, LIST, MIXED])
def test_accompanying_families_appear_exactly_when_the_plan_supports_them(
    case: features.DocCase,
) -> None:
    """输入：四种文档。输出：无。

    预期行为：伴随题族与计划的块**一一对应**——有提示框块就恰好一道 `callout_kind`、有列表块就
    恰好一道 `list_ordered`，没有就不出。
    """
    spec = cf.filled(case)
    items = questions.build(spec, LOC, 0)
    names = families(items)
    assert set(names) - {"join", "block_type"} == (
        {"callout_kind"} if spec.plan.callout_kinds() else set()
    ) | ({"list_ordered"} if spec.plan.list_flags() else set())
    for extra in ("callout_kind", "list_ordered"):
        assert names.count(extra) <= 1


def test_join_answers_come_from_the_block_split() -> None:
    """输入：一份 `mixed` 文档。输出：无。

    预期行为：每道 `join` 题的金答案就是「这两行在不在同一块」，题面里点名的两个行号都是
    state 里真有的行号。
    """
    spec = cf.filled(MIXED)
    rendered = render.layout(spec.plan, spec.texts, LOC)
    items = [item for item in questions.build(spec, LOC, 0) if item.family == "join"]
    pairs = rendered.asked_pairs(LOC)
    assert len(items) == len(pairs)
    for item, (left, right) in zip(items, pairs, strict=True):
        assert item.answer is rendered.same_block(left, right)
        assert item.question["type"] == "noul"
        assert left in str(item.question["instructions"])
        assert right in str(item.question["instructions"])
        assert sorted(item.question["criteria"]) == ["false", "true"]


def test_block_type_answers_follow_the_block_order() -> None:
    """输入：一份 `mixed` 文档。输出：无。

    预期行为：逐块出题、答案就是那一块的类型；六个选项的描述两两不同（考生才分得清）。
    """
    spec = cf.filled(MIXED)
    items = [item for item in questions.build(spec, LOC, 0) if item.family == "block_type"]
    assert [item.answer for item in items] == list(spec.plan.kinds())
    for item in items:
        criteria = item.question["criteria"]
        assert sorted(criteria) == sorted(BLOCK_KINDS)
        assert len(set(criteria.values())) == len(criteria)
        assert item.question["type"] == "choice"


def test_callout_and_list_answers_follow_the_plan() -> None:
    """输入：一份带提示框与有序列表的文档。输出：无。预期行为：两族答案都取自计划。"""
    items = questions.build(cf.filled(MIXED), LOC, 0)
    callout = next(item for item in items if item.family == "callout_kind")
    assert callout.answer == "tip"
    assert sorted(callout.question["criteria"]) == sorted(CALLOUT_KINDS)
    ordered = next(item for item in items if item.family == "list_ordered")
    assert ordered.answer is True
    assert ordered.question["type"] == "noul"
    assert sorted(ordered.question["criteria"]) == ["false", "true"]


@pytest.mark.parametrize("phrasing", [0, 1])
def test_phrasing_index_picks_a_different_wording(phrasing: int) -> None:
    """输入：两个问法编号。输出：无。

    预期行为：题面按编号取不同的模板，但**金答案一模一样**——问法不该影响标签。
    """
    spec = cf.filled(MIXED)
    items = questions.build(spec, LOC, phrasing)
    assert families(items) == families(questions.build(spec, LOC, 0))
    assert [item.answer for item in items] == [item.answer for item in questions.build(spec, LOC, 0)]
    wording = str(next(item for item in items if item.family == "block_type").question["instructions"])
    other = str(
        next(
            item
            for item in questions.build(spec, LOC, 1 - phrasing)
            if item.family == "block_type"
        ).question["instructions"]
    )
    assert wording != other


def test_questions_are_self_contained_and_have_no_leftover_slots() -> None:
    """输入：两种语言 × 全空间 345 组。输出：无。

    预期行为：题面自带「内容块」的定义（只读 state 与题面的读者必须能自己算答案），
    且渲染完一个花括号都不剩。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for case in features.build_cases(features.DocConfig()):
            spec = cf.filled(case, lang)
            for item in questions.build(spec, loc, 0):
                question = item.question
                instructions = str(question["instructions"])
                assert str(loc["doc"]["context"]) in instructions
                assert "{" not in instructions and "}" not in instructions
                for text in question["criteria"].values():
                    assert "{" not in str(text) and "}" not in str(text)
                assert item.answer in question["criteria"] or isinstance(item.answer, bool)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_list_ordered_questions_name_the_step_words(lang: str) -> None:
    """输入：两种语言 × 一份带列表的文档。输出：无。

    预期行为：「列表有没有序」那一条的题面与判据都点名**顺序词**。标签是按「每一项都带顺序词」算的，
    只问「有没有先后顺序」，读者会按语义去猜——「Set up your workstation / Meet the team」这种并列项
    读起来也像有先后顺序，真机上读者答 True、标签是 False，两边当场吵起来。
    """
    loc = locales.load(lang)
    items = questions.build(cf.filled(MIXED, lang), loc, 0)
    item = next(one for one in items if one.family == "list_ordered")
    text = str(item.question["instructions"]) + " ".join(
        str(value) for value in item.question["criteria"].values()
    )
    for word in parse.seq_words(loc):
        assert word in text, (lang, word)


def test_block_options_carry_a_name_and_a_description() -> None:
    """输入：两种语言。输出：无。预期行为：六个选项都是「名字：解释」，且解释两两不同。"""
    for lang in locales.available():
        loc = locales.load(lang)
        options = questions.block_options(loc)
        assert sorted(options) == sorted(BLOCK_KINDS)
        assert len(set(options.values())) == len(options)
        for kind, text in options.items():
            assert str(loc["type"][kind]) in text
            assert str(loc["type_desc"][kind]) in text


def test_callout_options_cover_every_kind() -> None:
    """输入：两种语言。输出：无。预期行为：三个选项的键与种类表一一对应，文案两两不同。"""
    for lang in locales.available():
        options = questions.callout_options(locales.load(lang))
        assert sorted(options) == sorted(CALLOUT_KINDS)
        assert len(set(options.values())) == len(options)


def test_choice_refuses_duplicate_options_and_unknown_answers() -> None:
    """输入：几组坏选项。输出：无。

    预期行为：选项描述重复、答案不在选项里，都在构造期报人话——两个看起来一样的选项考生没法分辨。
    """
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("f", "q", {"a": "一样", "b": "一样"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("f", "q", {"a": "甲", "b": "乙"}, "c")


def test_fill_leaves_unknown_placeholders_alone() -> None:
    """输入：一个带未知占位符的模板。输出：无。

    预期行为：只替换点名的那几个——未知的保持原样，由「题面不许剩花括号」那条测试抓出来。
    """
    assert questions._fill("{a} 与 {b}", a="L001") == "L001 与 {b}"


def test_families_order_is_fixed() -> None:
    """输入：无。输出：无。预期行为：先「并段」、再「逐块分类」、最后是伴随题（题号顺序稳定）。"""
    assert questions.FAMILIES == ("join", "block_type", "callout_kind", "list_ordered")
    items = questions.build(cf.filled(MIXED), LOC, 0)
    order = [questions.FAMILIES.index(name) for name in families(items)]
    assert order == sorted(order)


def test_option_descriptions_are_not_shared_between_families() -> None:
    """输入：一份带提示框的文档。输出：无。

    预期行为：`block_type` 的选项表与 `callout_kind` 的选项表取的是两份文案——混在一起会让
    「这一块是哪一类」和「提示框是哪一类」变成同一道题。
    """
    items = questions.build(cf.filled(MIXED), LOC, 0)
    block = next(item for item in items if item.family == "block_type")
    callout = next(item for item in items if item.family == "callout_kind")
    left = set(block.question["criteria"].values())
    right = set(callout.question["criteria"].values())
    assert not (left & right)
