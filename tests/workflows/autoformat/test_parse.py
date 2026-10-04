"""结构恢复的读法测试：只按文本本身读回「行 + 内容块」。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from workflows.autoformat import locales, parse
from workflows.autoformat.features import (
    KIND_CALLOUT,
    KIND_CODE,
    KIND_HEADING,
    KIND_LIST_ITEM,
    KIND_PARAGRAPH,
    KIND_QUOTE,
    SHORT_LIMIT,
)

#: 一份手工拼的文档，七种情况各来一块：标题紧贴正文、两项列表、命令行、引用、提示框。
DOC = "\n".join(
    [
        "L001| Build migration memo",
        "L002| we agreed to keep the rollout slow and reversible and to write down every move "
        "before production.",
        "L003|",
        "L004| platform and release leads",
        "L005| monitoring and alerts",
        "L006|",
        "L007| make --version",
        "L008| cargo --help",
        "L009|",
        "L010| measure first and automate later, we said that in the team many times but few "
        "projects ever did it--Alex",
        "L011|",
        "L012| Note: run the same commands in staging ahead of time, they behave the same way there.",
    ]
)

#: 上面那份文档期望的块序列。
DOC_KINDS = (
    KIND_HEADING,
    KIND_PARAGRAPH,
    KIND_LIST_ITEM,
    KIND_LIST_ITEM,
    KIND_CODE,
    KIND_QUOTE,
    KIND_CALLOUT,
)


def state(*bodies: str) -> str:
    """输入：各行正文（空串表示空行）。输出：契约形态的 state。预期行为：行号三位零填充。"""
    return "\n".join(f"{parse.line_id(index)}|" + (f" {body}" if body else "") for index, body in enumerate(bodies, start=1))


def test_line_id_is_zero_padded() -> None:
    """输入：几个行号。输出：无。预期行为：三位零填充，超过三位就原样展开。"""
    assert parse.line_id(1) == "L001"
    assert parse.line_id(42) == "L042"
    assert parse.line_id(1000) == "L1000"


def test_parse_lines_reads_ids_and_keeps_blank_lines() -> None:
    """输入：一段 state。输出：无。预期行为：空行也是行（正文为空串），行号顺序与输入一致。"""
    lines = parse.parse_lines(state("第一行", "", "第三行"))
    assert [line.id for line in lines] == ["L001", "L002", "L003"]
    assert [line.text for line in lines] == ["第一行", "", "第三行"]


@pytest.mark.parametrize("bad", ["随便一行没有行号的文本", "L01| 位数不够", "L001 缺竖线", ""])
def test_parse_lines_rejects_bad_line_ids(bad: str) -> None:
    """输入：几种行号格式不对的输入。输出：无。预期行为：报人话（不静默跳过那一行）。"""
    with pytest.raises(ValueError, match="行号格式不对"):
        parse.parse_lines(bad)


@pytest.mark.parametrize(
    ("text", "terminal", "short"),
    [
        ("", False, False),
        ("结尾有句号。", True, False),
        ("半句话没有标点", False, True),
        ("a" * SHORT_LIMIT, False, True),
        ("a" * (SHORT_LIMIT + 1), False, False),
    ],
)
def test_terminal_and_short_judgements(text: str, terminal: bool, short: bool) -> None:
    """输入：几种行文。输出：无。预期行为：空串两条都不算；短行 = 长度不超上限且无句末标点。"""
    assert parse.ends_terminal(text) is terminal
    assert parse.is_short_unpunctuated(text) is short


def test_command_tokens_union_and_short_line_excludes_code() -> None:
    """输入：语言字典。输出：无。

    预期行为：工具名是各命令族的并集；命令行**不算**短行（它是代码块的一部分），
    所以「短行」这条判据不会把命令读成标题。
    """
    loc = locales.load("en")
    tokens = parse.command_tokens(loc)
    assert {"make", "git", "docker", "pip"} <= tokens
    assert parse.is_code_line("git status", tokens)
    assert not parse.is_code_line("gitter status", tokens)
    assert not parse.is_short_line("git status", tokens)
    assert parse.is_short_line("short heading", tokens)


@pytest.mark.parametrize(
    ("reader", "broken", "match"),
    [
        (parse.command_tokens, {"code_tool": ["make"]}, "不是一个表"),
        (parse.command_tokens, {"code_tool": {"build": "make"}}, "必须是列表"),
        (parse.callout_index, {"callout_word": ["Note:"]}, "不是一个表"),
        (parse.callout_index, {"callout_word": {"note": "Note:"}}, "必须是列表"),
        (parse.callout_index, {"callout_word": {"note": ["A"], "tip": ["A"]}}, "同时属于"),
        (parse.seq_words, {"seq_word": {"step": "First"}}, "必须是列表"),
        (parse.attribution_marks, {"mark": {"attribution": []}}, "非空列表"),
        (parse.attribution_marks, {"mark": {"attribution": ["--", "--"]}}, "重复"),
    ],
)
def test_broken_tables_report_plain_words(
    reader: object, broken: dict[str, JsonValue], match: str
) -> None:
    """输入：几个读法函数 × 对应的坏表。输出：无。预期行为：都报人话，不静默降级。"""
    with pytest.raises(ValueError, match=match):
        reader(broken)


def test_callout_words_are_sorted_and_unique() -> None:
    """输入：语言字典。输出：无。预期行为：关键词表按字典序（报错信息才稳定），且两两不同。"""
    for lang in locales.available():
        loc = locales.load(lang)
        words = parse.callout_words(loc)
        assert list(words) == sorted(words)
        assert len(set(words)) == len(words)
        assert set(words) == set(parse.callout_index(loc))


def test_parse_state_reads_the_whole_document() -> None:
    """输入：一份七块的手工文档。输出：无。

    预期行为：块类型、每块的行号、提示框种类、列表有序性全部读对；空行不属于任何块。
    """
    doc = parse.parse_state(DOC, locales.load("en"))
    assert doc.kinds() == DOC_KINDS
    assert [block.line_ids for block in doc.blocks] == [
        ("L001",),
        ("L002",),
        ("L004",),
        ("L005",),
        ("L007", "L008"),
        ("L010",),
        ("L012",),
    ]
    assert doc.callouts()[0].callout_kind == "note"
    assert [block.list_ordered for block in doc.list_items()] == [False, False]
    assert len(doc.list_items()) == 2
    assert len(doc.lines) == 12


def test_block_index_and_same_block() -> None:
    """输入：同一份文档。输出：无。

    预期行为：行号能查到所属块；认不出来的行号给 None；`same_block` 只在同一块里为真。
    """
    doc = parse.parse_state(DOC, locales.load("en"))
    assert doc.block_index("L002") == 1
    assert doc.block_index("L003") is None
    assert doc.block_index("L999") is None
    assert doc.same_block("L007", "L008")
    assert not doc.same_block("L001", "L002")
    assert not doc.same_block("L004", "L005")
    assert not doc.same_block("L001", "L999")


def test_asked_pairs_skips_blank_lines_and_command_lines() -> None:
    """输入：同一份文档。输出：无。

    预期行为：只问两个非空、且都不是命令行的相邻对——命令行块上下都被空行包着，所以代码块
    天然不出这一族题。
    """
    loc = locales.load("en")
    doc = parse.parse_state(DOC, loc)
    pairs = parse.asked_pairs(doc.lines, parse.command_tokens(loc))
    assert pairs == [("L001", "L002"), ("L004", "L005")]


@pytest.mark.parametrize(
    ("bodies", "kind"),
    [
        (("Migration memo",), KIND_HEADING),
        (("we agreed--Alex",), KIND_QUOTE),
        (("a" * (SHORT_LIMIT + 1),), KIND_PARAGRAPH),
        (("Migration memo", "a" * (SHORT_LIMIT + 1)), KIND_HEADING),
        (("git status", "git log --oneline -5"), KIND_CODE),
    ],
)
def test_single_region_branches(bodies: tuple[str, ...], kind: str) -> None:
    """输入：几种「一段非空行」。输出：无。

    预期行为：判据顺序固定——全命令行 → 代码块；只有一行短行 → 标题（带署名前缀就是引用）；
    其余 → 开头那行是短行就当标题，剩下的拼成一段散文。
    """
    doc = parse.parse_state(state(*bodies), locales.load("en"))
    assert doc.blocks[0].kind == kind


def test_heading_sits_tight_against_its_paragraph() -> None:
    """输入：一个短行紧跟着一段长行（中间**不空行**）。输出：无。

    预期行为：两块——第一行是标题，后面那些行合成同一段正文。这是「空行才分块」的**第一处例外**，
    读者题面与考生题面里都照抄了这一条（`render.split_rules_block`，同一份）。2026-10-04 真机
    36 条被拒 17 条（47%）就是因为读者手里只有「空行才分块」：把紧贴正文的标题并进了正文，
    而这里的确定性读法是对的。
    """
    loc = locales.load("en")
    doc = parse.parse_state(state("Migration memo", "a" * (SHORT_LIMIT + 1), "b" * 10), loc)
    assert doc.kinds() == (KIND_HEADING, KIND_PARAGRAPH)
    assert doc.blocks[1].line_ids == ("L002", "L003")


def test_two_short_lines_become_two_list_items_and_read_the_step_words() -> None:
    """输入：两行短行，一行带步骤词。输出：无。

    预期行为：两项以上一律**每行一个列表项**（不是标题 + 正文）；有序性由步骤词给。
    """
    loc = locales.load("en")
    doc = parse.parse_state(state("First back up old config", "Then bump the version"), loc)
    assert doc.kinds() == (KIND_LIST_ITEM, KIND_LIST_ITEM)
    assert [block.list_ordered for block in doc.list_items()] == [True, True]
    plain = parse.parse_state(state("platform and release leads", "monitoring and alerts"), loc)
    assert [block.list_ordered for block in plain.list_items()] == [False, False]


def test_code_region_wins_over_the_markers_inside_it() -> None:
    """输入：一个带长选项与提示框关键词的命令行块。输出：无。

    预期行为：整段都是命令行就一定是代码块——读法的判据顺序排在标记词之前，所以英文署名前缀
    `--` 与长选项同形、命令里出现 `Note:` 都不会把它读歪。这条是 `render` / `judges` 给
    命令行块豁免标记词检查的依据。
    """
    loc = locales.load("en")
    doc = parse.parse_state(state("git log --oneline -5", "git show --stat"), loc)
    assert doc.kinds() == (KIND_CODE,)
    assert "--oneline" in doc.blocks[0].text


def test_empty_state_has_no_blocks() -> None:
    """输入：只有空行的 state。输出：无。预期行为：零块、零非空行。"""
    doc = parse.parse_state(state("", "", ""), locales.load("en"))
    assert doc.blocks == ()
    assert [line.text for line in doc.lines] == ["", "", ""]


def test_balanced_pairs_keeps_both_answers_in_equal_numbers() -> None:
    """输入：候选相邻对 + 一个假的「是不是同一块」判定。输出：无。

    预期行为：少的那一边一条不丢、多的那一边等距抽稀到同样条数；某一边一对都没有时另一边最多留
    `LONE_PAIRS` 对；抽稀后**仍是文档顺序**。不配平的话「是」会压倒性地多（真机全量 79%），
    模型永远答「是」就能拿高分。
    """
    pairs = [(f"L{index:03d}", f"L{index + 1:03d}") for index in range(1, 10)]
    same_ids = {"L001", "L002", "L004", "L005", "L006", "L007", "L008"}

    def same_of(left: str, right: str) -> bool:
        """假的「是不是同一块」判定：行标识在 `same_ids` 里就算同一块。"""
        return left in same_ids

    kept = parse.balanced_pairs(pairs, same_of)
    assert [pair for pair in kept if same_of(*pair)] and [pair for pair in kept if not same_of(*pair)]
    assert len([pair for pair in kept if same_of(*pair)]) == len(
        [pair for pair in kept if not same_of(*pair)]
    )
    assert kept == [pair for pair in pairs if pair in kept]
    one_sided = parse.balanced_pairs(pairs, lambda left, right: True)
    assert len(one_sided) == parse.LONE_PAIRS
    assert parse.balanced_pairs([], lambda left, right: True) == []
    assert parse.balanced_pairs(pairs, lambda left, right: False) == one_sided
