"""渲染与反解：往返必须逐块一致，认不出的行必须报错而不是错认。"""

from __future__ import annotations

import pytest

from tests.workflows.autoformat.conftest import _make_case, _make_spec
from workflows.autoformat import locales, parse, render, spec as sp
from workflows.autoformat.workflow import AutoformatWorkflow

LANGS = ("zh", "en")
MARKUP = ("#", "- ", "* ", "> ", "```", "1. ", "2. ", "3. ")


def _all_cases() -> list[sp.DocCase]:
    """输入：无。输出：默认配置下的全部 case（两种语言共用同一批素材）。"""
    return sp.build_cases(sp.DocConfig())


@pytest.mark.parametrize("lang", LANGS)
def test_roundtrip_matches_the_plan_on_every_case(lang: str) -> None:
    """输入：全部 case × 两种语言。输出：无。

    预期行为：渲染 → 反解之后，块的**数量、类型、行区间**与计划逐项一致，
    「并段」「提示框种类」「列表有序性」三类答案也一致。这是本 workflow 的地基：
    判别器就是靠这条往返独立重算的。
    """
    loc = locales.load(lang)
    for case in _all_cases():
        spec = sp.sample_spec(case, "train", 0, 0)
        rendered = render.layout(spec, loc)
        parsed = parse.parse_state(rendered.state(), loc)
        where = f"{lang}/{case.describe()}"
        assert parsed.kinds() == tuple(block.kind for block in rendered.blocks), where
        assert len(parsed.blocks) == len(rendered.blocks), where
        for index, block in enumerate(parsed.blocks):
            first = rendered.line_number(block.line_ids[0])
            last = rendered.line_number(block.line_ids[-1])
            assert (first, last) == rendered.spans[index], where
        for left, right in rendered.asked_pairs(loc):
            assert rendered.same_block(left, right) == parsed.same_block(left, right), where
        callouts = parsed.callouts()
        if callouts:
            want = next(b.callout_kind for b in rendered.blocks if b.kind == sp.KIND_CALLOUT)
            assert callouts[0].callout_kind == want, where
        items = parsed.list_items()
        if items:
            want = next(b.list_ordered for b in rendered.blocks if b.kind == sp.KIND_LIST_ITEM)
            assert items[0].list_ordered == want, where


@pytest.mark.parametrize("lang", LANGS)
def test_no_wrapped_line_looks_like_a_heading(lang: str) -> None:
    """输入：全部 case × 两种语言。输出：无。

    预期行为：散文块里除最后一行外**没有短行**——短行是「标题 / 列表项」的判据，
    折行断出一个短行会让整块的切分错位（实测踩过：中文在空格处断出 35 字的行）。
    """
    loc = locales.load(lang)
    for case in _all_cases():
        rendered = render.layout(sp.sample_spec(case, "train", 0, 0), loc)
        for index, (start, end) in enumerate(rendered.spans):
            if rendered.blocks[index].kind == sp.KIND_CODE:
                continue
            for number in range(start, end):
                text = rendered.texts[number - 1]
                assert not parse.is_short_unpunctuated(text), f"{case.describe()}: {text!r}"


@pytest.mark.parametrize("lang", LANGS)
def test_state_keeps_the_line_numbers_and_loses_the_markup(lang: str) -> None:
    """输入：全部 case × 两种语言。输出：无。

    预期行为：每行都是 `Lxxx|` 开头；正文里**没有** Markdown 标记打头
    （这就是「格式已丢失」的可检查形态）。
    """
    loc = locales.load(lang)
    for case in _all_cases():
        state = render.render_state(sp.sample_spec(case, "train", 0, 0), loc)
        for number, raw in enumerate(state.split("\n"), start=1):
            assert raw.startswith(parse.line_id(number) + "|"), raw
            body = raw.split("|", 1)[1].strip()
            for token in MARKUP:
                assert not body.startswith(token), f"{case.describe()}: {body!r}"


def test_parse_rejects_a_line_without_a_number() -> None:
    """输入：一行没有行号前缀的文本。输出：无。预期行为：报 ValueError 并给出原文。"""
    with pytest.raises(ValueError, match="行号格式不对"):
        parse.parse_state("第一行没有行号\nL002| 第二行", locales.load("zh"))


def test_parse_reports_a_line_out_of_order_as_is() -> None:
    """输入：行号跳号的文本。输出：无。预期行为：照原样读出来（跳号本身不是语法错误）。"""
    parsed = parse.parse_state("L001| 构建系统迁移\nL009| 正文如下。", locales.load("zh"))
    assert [line.id for line in parsed.lines] == ["L001", "L009"]


@pytest.mark.parametrize("lang", LANGS)
def test_reparse_reads_both_languages_through_the_merged_table(lang: str) -> None:
    """输入：两种语言各渲染一份带提示框的文档。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，靠合并表也能把提示框认出来。
    """
    spec = _make_spec(shape="callout", variant="warning")
    state = render.render_state(spec, locales.load(lang))
    parsed = AutoformatWorkflow().reparse(state)
    assert [block.kind for block in parsed.blocks] == ["heading", "paragraph", "callout", "paragraph"]
    assert parsed.callouts()[0].callout_kind == "warning"


def test_reparse_rejects_a_non_string_state() -> None:
    """输入：一个字典。输出：无。预期行为：报 TypeError（state 必须是字符串）。"""
    with pytest.raises(TypeError, match="字符串"):
        AutoformatWorkflow().reparse({"grid": []})


def test_render_is_pure() -> None:
    """输入：同一份规格渲染两次。输出：无。预期行为：逐字节一致。"""
    spec = _make_spec(shape="mixed", variant="note", code_family="package")
    loc = locales.load("zh")
    assert render.render_state(spec, loc) == render.render_state(spec, loc)


def test_different_cases_render_different_text() -> None:
    """输入：只差一个维度的两份规格。输出：无。预期行为：文本必须不同。"""
    loc = locales.load("zh")
    ordered = sp.sample_spec(_make_case(shape="list", variant="ordered"), "t", 0, 0)
    bullet = sp.sample_spec(_make_case(shape="list", variant="bullet"), "t", 0, 0)
    assert render.render_state(ordered, loc) != render.render_state(bullet, loc)


def test_wrap_text_keeps_every_line_within_reach() -> None:
    """输入：长短不一的几段文本。输出：无。

    预期行为：末行之外的每行都不超过行宽；拼回去（去掉断点处的空白）等于原文；
    空文本折出空列表。
    """
    for text in ("短句。", "a" * 200, "中文" * 60, "one two three four five six seven eight nine ten " * 4):
        for width in (56, 64, 72):
            lines = render.wrap_text(text, width)
            assert lines
            for line in lines[:-1]:
                assert len(line) <= width
            assert "".join(lines).replace(" ", "") == text.strip().replace(" ", "")
    assert render.wrap_text("", 64) == []


def test_needs_blank_only_relaxes_for_lists_and_headings() -> None:
    """输入：四种块序列。输出：无。预期行为：只有「同列表相邻项」与「标题接正文」不空行。"""
    head = render.Block(sp.KIND_HEADING, "标题")
    para = render.Block(sp.KIND_PARAGRAPH, "正文。")
    item_a = render.Block(sp.KIND_LIST_ITEM, "甲", list_id=1)
    item_b = render.Block(sp.KIND_LIST_ITEM, "乙", list_id=1)
    item_other = render.Block(sp.KIND_LIST_ITEM, "丙", list_id=2)
    assert render.needs_blank((head, para), 1) is False
    assert render.needs_blank((item_a, item_b), 1) is False
    assert render.needs_blank((item_a, item_other), 1) is True
    assert render.needs_blank((para, item_a), 1) is True
    assert render.needs_blank((head, item_a), 1) is True
    assert render.needs_blank((head, para), 0) is False
