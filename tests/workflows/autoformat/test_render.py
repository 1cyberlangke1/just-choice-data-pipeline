"""结构恢复的渲染测试：折行、空行规则、逐槽位字面校验与提示词。"""

from __future__ import annotations

import pytest

from tests.workflows.autoformat import conftest as cf
from jc.workflow import DraftError
from workflows.autoformat import features, locales, parse, render
from workflows.autoformat.features import KIND_CODE

#: 全空间 345 组：下面好几条测试都拿它当「都要过」的清单。
ALL_CASES = features.build_cases(features.DocConfig())
LOC = locales.load("zh")
PLAIN = features.DocCase("build", 64, "plain", "na", "na")
CALLOUT = features.DocCase("build", 64, "callout", "note", "na")
CODE = features.DocCase("build", 64, "code", "na", "vcs")
MIXED = features.DocCase("build", 64, "mixed", "note", "vcs")


@pytest.mark.parametrize("width", features.WRAPS)
def test_wrap_text_keeps_non_last_lines_above_the_short_limit(width: int) -> None:
    """输入：一段长散文 × 每种行宽。输出：无。

    预期行为：非末行**一定**比短行上限长（否则反解会把它读成标题）、没有空行、拼回来与原文一致
    （英文按词边界断、中文按字符断，两种都不丢字符）。
    """
    english = "we agreed to keep the rollout slow and reversible " * 4
    for text in ("甲" * 200, english.strip()):
        lines = render.wrap_text(text, width)
        assert len(lines) > 1
        assert all(line.strip() for line in lines)
        assert all(len(line) > features.SHORT_LIMIT for line in lines[:-1]), (width, lines)
        if " " in text:
            assert " ".join(lines) == " ".join(text.split())
        else:
            assert "".join(lines) == text


def test_wrap_text_leaves_short_text_alone() -> None:
    """输入：一段比行宽还短的文本。输出：无。预期行为：原样一行，不硬塞空行。"""
    assert render.wrap_text("short", 64) == ["short"]
    assert render.wrap_text("  spaced  ", 64) == ["spaced"]


def test_forbidden_words_follow_the_plan() -> None:
    """输入：四种计划。输出：无。

    预期行为：提示词里「不许出现」的那张表 = 全部标记词 − 这份计划**确实要用**的那些；
    提示框关键词按种类减；步骤词**只在这份计划没有有序列表时**列入（无序列表项带上它，整张列表会被
    读成有序的）；署名前缀一律列入（英文的 `--` 与中文的 `——` 都是散文里常见的破折号，
    写进别的块会被读成引用）。
    """
    all_callouts = set(parse.callout_words(LOC))
    steps = set(parse.seq_words(LOC))
    marks = set(parse.attribution_marks(LOC))
    plain = set(render.forbidden_words(features.build_plan(PLAIN), LOC))
    assert plain == all_callouts | steps | marks
    callout = set(render.forbidden_words(features.build_plan(CALLOUT), LOC))
    assert callout == (all_callouts - set(LOC["callout_word"]["note"])) | steps | marks
    ordered = features.DocCase("build", 64, "list", "ordered", "na")
    assert not (set(render.forbidden_words(features.build_plan(ordered), LOC)) & steps)
    bullet = features.DocCase("build", 64, "list", "bullet", "na")
    assert set(render.forbidden_words(features.build_plan(bullet), LOC)) >= steps


def test_build_prompt_states_every_requirement_without_leftover_slots() -> None:
    """输入：两种语言 × 全空间 345 组。输出：无。

    预期行为：提示词里逐块给出类型与字面要求、逐条列出不许出现的标记词，且**一个花括号都不剩**
    ——漏填会在 `format` 上抛错，多写的花括号会原样漏给模型，让它以为那是要填的槽。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for case in ALL_CASES:
            plan = features.build_plan(case)
            prompt = render.build_prompt(loc, plan)
            assert "{" not in prompt and "}" not in prompt, case.describe()
            assert str(loc["topic_name"][case.topic]) in prompt
            assert str(loc["doc"]["mark_rule"]) in prompt
            assert str(loc["doc"]["output"]) in prompt
            for slot in plan.slots:
                assert str(loc["type"][slot.kind]) in prompt
            for word in render.forbidden_words(plan, loc):
                assert word in prompt, f"{lang} {case.describe()} 少了禁止词 {word!r}"


def test_build_prompt_forbids_the_marker_words_that_are_not_required() -> None:
    """输入：一个带 `note` 提示框的计划。输出：无。

    预期行为：别的种类的关键词被逐条列进「不许出现」，本种类的不列——模型才不会顺手把
    「警告：」写进正文里，把那一块读成另一种提示框。
    """
    prompt = render.build_prompt(LOC, features.build_plan(CALLOUT))
    lines = prompt.split("\n")
    assert str(LOC["doc"]["forbidden_header"]) in lines
    forbidden_line = lines[lines.index(str(LOC["doc"]["forbidden_header"])) + 1]
    for kind in ("tip", "warning"):
        for word in LOC["callout_word"][kind]:
            assert str(word) in forbidden_line
    for word in LOC["callout_word"]["note"]:
        assert str(word) not in forbidden_line


def test_build_task_carries_the_plan_and_the_case() -> None:
    """输入：一个规格。输出：无。预期行为：任务 id、域、响应模型与 spec 都对得上（假后端靠它造草稿）。"""
    spec = cf.spec_for(MIXED)
    task = render.build_task(spec, LOC)
    assert task.task_id == "autoformat-train-0000001-render"
    assert task.domain == "autoformat"
    assert task.response_model is render.SlotDraft
    assert task.spec["case"] == MIXED.as_dict()
    assert task.spec["kinds"] == list(spec.plan.kinds())
    assert task.spec["callout_kinds"] == ["note"]
    assert len(task.spec["list_flags"]) == len(spec.plan.list_flags())


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_check_draft_accepts_every_faithful_draft(lang: str) -> None:
    """输入：两种语言 × 全空间 345 组。输出：无。预期行为：一条都不报错。"""
    loc = locales.load(lang)
    for case in ALL_CASES:
        plan = features.build_plan(case)
        render.check_draft(plan, cf.texts_for(plan, loc), loc)


@pytest.mark.parametrize(
    ("switch", "case", "match"),
    [
        ("heading_punct", PLAIN, "不该以标点结尾"),
        ("short_paragraph", PLAIN, "长度"),
        ("drop_text", PLAIN, "计划要"),
        ("callout_wrong_word", CALLOUT, "没以"),
        ("code_wrong_tool", CODE, "有一行不以"),
    ],
)
def test_check_draft_rejects_broken_drafts(switch: str, case: features.DocCase, match: str) -> None:
    """输入：各种「写歪」的开关。输出：无。预期行为：当场报错，理由里点名坏在哪。"""
    plan = features.build_plan(case)
    texts = cf.FakeBackend(**{switch: True}).texts(plan, LOC)
    with pytest.raises(ValueError, match=match):
        render.check_draft(plan, texts, LOC)


def test_check_draft_rejects_empty_and_multiline_prose() -> None:
    """输入：空槽位与带换行的散文槽位。输出：无。预期行为：两条都当场报错（散文必须写成一段）。"""
    plan = features.build_plan(PLAIN)
    texts = cf.texts_for(plan, LOC)
    with pytest.raises(ValueError, match="是空的"):
        render.check_draft(plan, ["   ", *texts[1:]], LOC)
    with pytest.raises(ValueError, match="带了换行"):
        render.check_draft(plan, [texts[0] + "\n第二行", *texts[1:]], LOC)


def test_check_draft_rejects_a_length_mismatch() -> None:
    """输入：少写一块。输出：无。预期行为：报错并点出「给了几段、计划要几块」。"""
    plan = features.build_plan(PLAIN)
    with pytest.raises(ValueError, match="计划要"):
        render.check_draft(plan, cf.texts_for(plan, LOC)[:-1], LOC)


def test_contract_violations_are_marked_retryable() -> None:
    """输入：一条长度写歪的草稿。输出：无。

    预期行为：抛的是 `jc.workflow.DraftError`（不是光秃秃的 `ValueError`），且类上带
    `retryable_for_case`——管线看到这个标记才会**重渲染一次**。真机 en 批次里四成拒稿
    都是「列表项 45~49 字符 vs 上限 40」这类契约级失败，单发就直接丢样本（PLAN 10.83）。
    """
    plan = features.build_plan(PLAIN)
    texts = cf.FakeBackend(short_paragraph=True).texts(plan, LOC)
    with pytest.raises(DraftError) as caught:
        render.check_draft(plan, texts, LOC)
    assert getattr(caught.value, "retryable_for_case", False) is True


def test_asked_pairs_are_balanced_in_every_document() -> None:
    """输入：两种语言 × 全空间 345 组。输出：无。

    预期行为：每一份文档里「是（同一块）」与「不是」的条数**相等**；某一边一对都没有时，
    另一边最多留 `parse.LONE_PAIRS` 对。不配平的话，段落折行带来的「是」会压倒块边界带来的
    「不是」——train 全量实测 505 对里「是」占 79%，模型永远答「是」就能拿 79 分。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for case in ALL_CASES:
            spec = cf.filled(case, lang)
            rendered = render.layout(spec.plan, spec.texts, loc)
            pairs = rendered.asked_pairs(loc)
            same = [pair for pair in pairs if rendered.same_block(*pair)]
            diff = [pair for pair in pairs if not rendered.same_block(*pair)]
            if same and diff:
                assert len(same) == len(diff), (lang, case.describe())
            else:
                assert len(pairs) <= parse.LONE_PAIRS, (lang, case.describe())


def test_code_blocks_are_exempt_from_the_marker_word_checks() -> None:
    """输入：一个含长选项与提示框关键词的命令行块。输出：无。

    预期行为：通过——读法里「整段命令行 → 代码块」排在标记词判据之前，所以英文署名前缀 `--`
    与长选项同形不会造成假拒稿（不豁免的话每个英文代码块都会被判成引用）。
    """
    loc = locales.load("en")
    plan = features.build_plan(CODE)
    texts = cf.texts_for(plan, loc)
    assert any("--" in text for text in texts)
    index = next(i for i, slot in enumerate(plan.slots) if slot.kind == KIND_CODE)
    texts[index] = "git commit -m 'Note: fix'\nhg log --limit 5"
    render.check_draft(plan, texts, loc)


def test_blocks_from_rejects_a_length_mismatch() -> None:
    """输入：块计划与一段文字。输出：无。

    预期行为：报错——静默截断会让「模型少写一块」变成一个看不出来的错位。
    """
    with pytest.raises(ValueError, match="计划有"):
        render.blocks_from(features.build_plan(PLAIN), ["只有一段"])


def test_layout_puts_blank_lines_between_blocks_with_two_exceptions() -> None:
    """输入：一份 `mixed` 文档的忠实文字。输出：无。

    预期行为：块与块之间空一行；两个例外——同一个列表里相邻的列表项之间**不**空行、标题**紧贴**
    后面的正文。这两条例外同时也是「并段」那一族题目的来源。
    """
    plan = features.build_plan(MIXED)
    rendered = render.layout(plan, cf.texts_for(plan, LOC), LOC)
    blocks = rendered.blocks
    assert len(blocks) > 5
    for index in range(1, len(blocks)):
        start = rendered.spans[index][0]
        end = rendered.spans[index - 1][1]
        assert (start > end + 1) is render.needs_blank(blocks, index), index
    assert render.needs_blank(blocks, 1) is False  # 标题紧贴正文
    item_indexes = [i for i, block in enumerate(blocks) if block.kind == "list_item"]
    assert len(item_indexes) >= 2
    assert all(render.needs_blank(blocks, i) is False for i in item_indexes[1:])


def test_layout_never_wraps_a_code_block() -> None:
    """输入：一份带命令行块的文档。输出：无。预期行为：命令行整块不折行（折了就不是能敲的命令了）。"""
    plan = features.build_plan(CODE)
    texts = cf.texts_for(plan, LOC)
    rendered = render.layout(plan, texts, LOC)
    for index, slot in enumerate(plan.slots):
        if slot.kind != KIND_CODE:
            continue
        start, end = rendered.spans[index]
        assert end - start + 1 == len(texts[index].split("\n"))


def test_render_state_is_line_ids_plus_bodies_and_round_trips() -> None:
    """输入：一份 `mixed` 文档。输出：无。

    预期行为：每行是 `L007| 正文`（空行没有尾空格）、行号与位置一一对应、反解读出来的块类型
    与计划逐项相同。
    """
    plan = features.build_plan(MIXED)
    state = render.render_state(plan, cf.texts_for(plan, LOC), LOC)
    lines = state.split("\n")
    assert lines[0] == "L001| Migration memo"
    for index, line in enumerate(lines, start=1):
        assert line.partition("|")[0] == parse.line_id(index)
    assert any(line.endswith("|") for line in lines)
    assert not any(line.endswith("| ") for line in lines)
    assert render.parse_state(state).kinds() == plan.kinds()


def test_rendered_coordinates_line_up_with_the_blocks() -> None:
    """输入：一份渲染结果。输出：无。

    预期行为：行号能查到所属块、认不出来的给 None；`asked_pairs` 只问两个非空、且都不是命令行的
    相邻对（题目侧与判别侧共用同一套判据）。
    """
    plan = features.build_plan(MIXED)
    rendered = render.layout(plan, cf.texts_for(plan, LOC), LOC)
    assert rendered.line_number("L001") == 1
    assert rendered.line_number("L999") is None
    assert rendered.block_index_of("L001") == 0
    assert rendered.block_index_of("L999") is None
    assert rendered.same_block("L001", "L001")
    assert not rendered.same_block("L001", "L999")
    tokens = parse.command_tokens(LOC)
    pairs = rendered.asked_pairs(LOC)
    assert pairs
    for left, right in pairs:
        assert rendered.line_number(left) + 1 == rendered.line_number(right)
        assert not parse.is_code_line(rendered.texts[rendered.line_number(left) - 1], tokens)
        assert not parse.is_code_line(rendered.texts[rendered.line_number(right) - 1], tokens)


def test_build_prompt_carries_the_validation_error_when_given() -> None:
    """输入：同一份计划，一次带失败理由、一次不带。输出：无。

    预期行为：带理由时末尾多出「上一版没通过校验」那一段，并**原样**带上 `check_draft` 的原话；
    不带时一个字都不提——提了模型会以为自己写错过。
    """
    plan = features.build_plan(PLAIN)
    plain = render.build_prompt(LOC, plan)
    told = render.build_prompt(LOC, plan, "第 2 块（paragraph） 长度 58 不在 74~240")
    header = str(LOC["doc"]["fix_header"])
    assert header in told
    assert told.endswith("第 2 块（paragraph） 长度 58 不在 74~240"), "理由要附在末尾"
    assert header not in plain
    assert "长度 58" not in plain
