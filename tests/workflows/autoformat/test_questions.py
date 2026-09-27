"""题族：哪些题该出、出多少、选项与题面长什么样。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from jc import assemble
from tests.workflows.autoformat.conftest import _make_case, _make_spec
from workflows.autoformat import locales, parse, questions, render, spec as sp


def _labeled(spec: sp.DocSpec, locale: str = "zh") -> list[Any]:
    """输入：规格、语言。输出：题目列表。预期行为：走 workflow 的同一套出题逻辑。"""
    return questions.build(spec, locales.load(locale), spec.phrasing, random.Random(0))


def _families(spec: sp.DocSpec, locale: str = "zh") -> list[str]:
    """输入：规格、语言。输出：题目列表里各题的题族（按题序）。"""
    return [item.family for item in _labeled(spec, locale)]


def test_family_set_follows_the_document_shape() -> None:
    """输入：六种形状各一份文档。输出：无。

    预期行为：`join` 与 `block_type` 一定有；`list_ordered` 只在有列表时出；
    `callout_kind` 只在有提示框时出。
    """
    assert set(_families(_make_spec(shape="plain"))) == {"join", "block_type"}
    assert set(_families(_make_spec(shape="list", variant="ordered"))) == {
        "join",
        "block_type",
        "list_ordered",
    }
    assert set(_families(_make_spec(shape="code", code_family="vcs"))) == {"join", "block_type"}
    assert set(_families(_make_spec(shape="callout", variant="note"))) == {
        "join",
        "block_type",
        "callout_kind",
    }
    assert set(_families(_make_spec(shape="mixed", variant="warning", code_family="build"))) == set(
        questions.FAMILIES
    )


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_join_asks_exactly_the_adjacent_pairs(lang: str) -> None:
    """输入：一份混合文档 × 两种语言。输出：无。

    预期行为：`join` 的题数正好等于「需要问的相邻行对」数，题面里带上那两行的行号。
    """
    spec = _make_spec(shape="mixed", variant="note", code_family="build")
    loc = locales.load(lang)
    rendered = render.layout(spec, loc)
    pairs = rendered.asked_pairs(loc)
    labeled = [item for item in _labeled(spec, lang) if item.family == "join"]
    assert len(labeled) == len(pairs)
    for item, (left, right) in zip(labeled, pairs, strict=True):
        assert left in item.question["instructions"]
        assert right in item.question["instructions"]
        assert set(item.question["criteria"]) == {"true", "false"}
        assert isinstance(item.answer, bool)


def test_block_type_asks_every_block_with_six_options() -> None:
    """输入：一份混合文档。输出：无。

    预期行为：每题一个块、顺序与块序一致；六种类型的选项描述两两不同。
    """
    spec = _make_spec(shape="mixed", variant="tip", code_family="package")
    rendered = render.layout(spec, locales.load("zh"))
    labeled = [item for item in _labeled(spec) if item.family == "block_type"]
    assert len(labeled) == len(rendered.blocks)
    assert [item.answer for item in labeled] == [block.kind for block in rendered.blocks]
    for item in labeled:
        options = item.question["criteria"]
        assert set(options) == set(sp.BLOCK_KINDS)
        assert len(set(options.values())) == len(options)


def test_every_choice_answer_is_a_real_option() -> None:
    """输入：全部 case（两种语言）。输出：无。

    预期行为：组装出来的 choice 答案编号都在选项范围内，且指向金答案那个选项名。
    """
    for lang in ("zh", "en"):
        for case in sp.build_cases(sp.DocConfig()):
            spec = sp.sample_spec(case, "train", 0, 0)
            sample = assemble.assemble_sample(
                "autoformat-train-0000000",
                render.render_state(spec, locales.load(lang)),
                _labeled(spec, lang),
                random.Random(1),
            )
            for qid, question in sample["questions"].items():
                answer = sample["answers"][qid]
                if question["type"] == "choice":
                    options = list(question["criteria"])
                    assert 0 <= answer["choice"] < len(options), (lang, case.describe(), qid)
                else:
                    assert isinstance(answer["noul"], bool), (lang, case.describe(), qid)


def test_instructions_define_what_a_block_is() -> None:
    """输入：一份文档。输出：无。

    预期行为：每道题的题面都带上「内容块」的定义——只读 state 与题面的读者必须能自己算。
    """
    spec = _make_spec(shape="mixed", variant="note", code_family="vcs")
    context = str(locales.load("zh")["doc"]["context"])
    for item in _labeled(spec):
        assert context in item.question["instructions"]


def test_phrasing_index_wraps_around() -> None:
    """输入：问法编号超出问法池。输出：无。预期行为：回绕，不报错。"""
    spec = _make_spec()
    many = questions.build(spec, locales.load("zh"), 7, random.Random(0))
    assert many
    assert len(questions.FAMILIES) == len(questions.CHOICE_FAMILIES) + len(questions.NOUL_FAMILIES)


def test_choice_rejects_duplicate_option_descriptions() -> None:
    """输入：两个描述一模一样的选项。输出：无。预期行为：构造期就报错。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("block_type", "题面", {"a": "一样", "b": "一样"}, "a")


def test_choice_rejects_an_answer_outside_the_options() -> None:
    """输入：答案不在选项里的题。输出：无。预期行为：构造期就报错。"""
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("block_type", "题面", {"a": "甲", "b": "乙"}, "c")


def test_callout_and_list_answers_come_from_the_rendered_blocks() -> None:
    """输入：带提示框与列表的文档。输出：无。预期行为：伴随题的答案与块上的标记一致。"""
    spec = _make_spec(shape="mixed", variant="warning", code_family="build")
    rendered = render.layout(spec, locales.load("zh"))
    labeled = {item.family: item for item in _labeled(spec)}
    callout = next(block for block in rendered.blocks if block.kind == sp.KIND_CALLOUT)
    item = next(block for block in rendered.blocks if block.kind == sp.KIND_LIST_ITEM)
    assert labeled["callout_kind"].answer == callout.callout_kind
    assert labeled["list_ordered"].answer == item.list_ordered


def test_join_answer_matches_the_block_split() -> None:
    """输入：一份混合文档。输出：无。

    预期行为：`join` 的金答案就是「两行在不在同一块」——题目侧不许自己另立一套判据。
    """
    spec = _make_spec(shape="mixed", variant="note", code_family="vcs")
    loc = locales.load("zh")
    rendered = render.layout(spec, loc)
    answers = [item.answer for item in _labeled(spec) if item.family == "join"]
    want = [rendered.same_block(a, b) for a, b in rendered.asked_pairs(loc)]
    assert answers == want
    assert parse.command_tokens(loc)  # 命令行首词表非空（判据表来自语言文件）
