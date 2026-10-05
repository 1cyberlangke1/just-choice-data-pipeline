"""改动审查的文本外壳测试：渲染、反解、形状校验与两处禁写检查。"""

from __future__ import annotations

import pytest

from workflows.review import features, locales, render
from workflows.review.features import TagSet

from tests.workflows.review.conftest import body_for


def tags_of(
    secret: str = "no",
    debug: str = "no",
    file_kind: str = "source",
    change_kind: str = "modified",
    size: str = "small",
) -> TagSet:
    """输入：五维取值（都有默认）。输出：一组注入特征。预期行为：只用于构造测试输入。"""
    return TagSet(
        secret=secret, debug=debug, file_kind=file_kind, change_kind=change_kind, size=size
    )


def state_for(tags: TagSet, locale: str = "zh") -> str:
    """输入：注入特征、语言。输出：拼好的 state。预期行为：走渲染期同一套校验。"""
    loc = locales.load(locale)
    body = body_for(tags.as_dict())
    render.check_draft({"body": body}, tags, loc)
    return render.to_state({"body": body}, tags, loc)


def test_round_trip_for_every_combination() -> None:
    """输入：全空间 108 组特征。输出：无。

    预期行为：每一格都能渲染出 state 并**原样反解回来**——清单行逐格相等、改动内容逐行
    相等、三张表等于规范表。这是「渲染必须可反解」的最小落地。
    """
    loc = locales.load("zh")
    canonical = (
        render.rule_rows(loc),
        render.severity_rows(loc),
        render.gate_rows(loc),
    )
    for tags in features.build_cases(2048):
        body = body_for(tags.as_dict())
        render.check_draft({"body": body}, tags, loc)
        state = render.to_state({"body": body}, tags, loc)
        parsed = render.parse_state(state)
        assert parsed.manifest == render.ManifestState(
            file_kind=tags.file_kind, change_kind=tags.change_kind, size=tags.size
        )
        assert list(parsed.body_lines) == body.splitlines()
        assert (parsed.rules, parsed.severity, parsed.gate) == canonical


def test_state_has_the_five_sections_in_order() -> None:
    """输入：一段 state。输出：无。预期行为：段落顺序固定，改动内容独占段落头下面那几行。"""
    tags = tags_of(secret="yes", change_kind="added", size="small")
    state = state_for(tags)
    loc = locales.load("zh")
    heads = [
        str(loc["meta"][key])
        for key in render.SECTION_KEYS
    ]
    positions = [state.index(f"【{title}】") for title in heads]
    assert positions == sorted(positions)
    assert state.splitlines()[0] == "【改动内容】"


def test_context_lines_keep_their_leading_space() -> None:
    """输入：含上下文行（空格前缀）的改动。输出：无。

    预期行为：反解时**只去尾部空白**——行首那一个字符就是 `+` / `-` / 空格，
    去掉它「上下文行」就退化成「没前缀的行」，卫生判别器会当场误判。
    """
    tags = tags_of(size="medium")
    loc = locales.load("zh")
    body = "+ added = 1\n context = 2\n- removed = 3\n+ another = 4"
    render.check_draft({"body": body}, tags, loc)
    parsed = render.parse_state(render.to_state({"body": body}, tags, loc))
    assert list(parsed.body_lines) == ["+ added = 1", " context = 2", "- removed = 3", "+ another = 4"]


def test_to_state_does_not_rewrite_the_model_text() -> None:
    """输入：带奇怪空白的改动。输出：无。预期行为：正文原样进 state（反解靠原文）。"""
    tags = tags_of(size="small")
    loc = locales.load("zh")
    body = '+ name = "x"   \n- name = "y"'
    state = render.to_state({"body": body}, tags, loc)
    parsed = render.parse_state(state)
    assert list(parsed.body_lines) == ['+ name = "x"', '- name = "y"']


def test_diff_mark_reads_the_mark_past_the_indent() -> None:
    """输入：各种行。输出：无。

    预期行为：标记认得出（剥掉行首空白再认），而真实代码里的 `-1` / `-> int` 不算标记
    ——把它们读成删除行会把上下文行算成改动行。
    """
    assert features.diff_mark("+ x = 1") == "+"
    assert features.diff_mark("- x = 1") == "-"
    assert features.diff_mark("  - x = 1") == "-"
    assert features.diff_mark("-") == "-"
    assert features.diff_mark(" x = 1") == " "
    assert features.diff_mark("x = 1") == " "
    assert features.diff_mark("-1") == " "
    assert features.diff_mark("    -> int:") == " "
    assert features.diff_mark("") == " "


def test_indented_and_bare_context_lines_are_normalized() -> None:
    """输入：整段缩进两格、上下文行没写前缀的草稿。输出：无。

    预期行为：两种真机写法都收下（真机各撞过一批），拼进 state 的是「每行都有前缀」的
    规范形态；代码本身一个字符都不改。
    """
    tags = tags_of(size="medium")
    loc = locales.load("zh")
    body = "def process_order(order_id):\n  - status = 1\n  + status = 2\n  return status"
    render.check_draft({"body": body}, tags, loc)
    parsed = render.parse_state(render.to_state({"body": body}, tags, loc))
    assert list(parsed.body_lines) == [
        " def process_order(order_id):",
        "- status = 1",
        "+ status = 2",
        "  return status",
    ]


def test_a_slightly_overlong_body_is_accepted() -> None:
    """输入：行数略超写作指引上界的正文。输出：无。

    预期行为：验收上界是宽容过的（`features.SIZE_LINES_SLACK`）——英文批真机上「大 6~12 行」
    被写到 14 行，清单那一行才是权威，按 12 硬拒是白拒好数据（范式 §9.19）。
    """
    tags = tags_of(size="large")
    loc = locales.load("zh")
    body = "\n".join(f"+ value_{index} = compute({index})" for index in range(14))
    render.check_draft({"body": body}, tags, loc)
    parsed = render.parse_state(render.to_state({"body": body}, tags, loc))
    assert len(parsed.body_lines) == 14


def test_english_state_parses_too() -> None:
    """输入：英文 state。输出：无。预期行为：反解与语言无关（各语言段落名都试一遍）。"""
    tags = tags_of(secret="yes", file_kind="config", size="medium")
    state = state_for(tags, "en")
    parsed = render.parse_state(state)
    assert parsed.manifest is not None
    assert parsed.manifest.file_kind == "config"
    assert parsed.body_lines


def test_same_inputs_give_byte_identical_states() -> None:
    """输入：同一条样本渲染两次。输出：两段 state。预期行为：逐字节一致。"""
    tags = tags_of(debug="yes", size="large")
    assert state_for(tags) == state_for(tags)


def test_label_leak_finds_a_copied_header() -> None:
    """输入：开头抄了段落名的文本。输出：无。预期行为：报出那个段落名。"""
    loc = locales.load("zh")
    assert render.label_leak("改动内容：+ x = 1", loc) == "改动内容"
    assert render.label_leak("【改动清单】+ x = 1", loc) == ""
    assert render.label_leak("+ x = 1", loc) == ""


def test_banned_echo_leaves_code_identifiers_alone() -> None:
    """输入：一段满是 `test` / `size` / `config` 的代码。输出：无。

    预期行为：一个都不报——正文是代码，这些标识符在代码里到处都是，收进禁写清单
    会大面积误伤（宁可漏检也不误判）。
    """
    zh = locales.load("zh")
    code = '+ def test_size(config):\n+     size = config["size"]\n+     return test(size)'
    assert render.banned_echo(code, zh) == []


@pytest.mark.parametrize(
    ("tags", "body", "expect"),
    [
        (tags_of(size="small"), "   ", "空的"),
        (tags_of(size="small"), '```\n+ x = 1\n```', "围栏"),
        (tags_of(size="small"), "x = 1\ny = 2", "一行都没变"),
        (tags_of(change_kind="added", size="small"), "x = 1", r"一行 \+ 都没有"),
        (tags_of(change_kind="deleted", size="small"), "x = 1", r"一行 - 都没有"),
        (tags_of(size="large"), "+ x = 1\n- y = 2", "对不上"),
        (tags_of(change_kind="added", size="medium"), "+ a = 1\n- b = 2\n+ c = 3\n+ d = 4", "新增"),
        (tags_of(change_kind="deleted", size="medium"), "- a = 1\n+ b = 2\n- c = 3\n- d = 4", "删除"),
        (tags_of(size="small"), "改动内容：+ x = 1", "段落名"),
        (tags_of(size="small"), "+ x = 1\n+ y = 2  # 见门控表", "门控表"),
    ],
)
def test_check_draft_rejects_bad_shapes(tags: TagSet, body: str, expect: str) -> None:
    """输入：各种坏形状的草稿。输出：无。预期行为：渲染期就抛 ValueError，理由点出问题。"""
    with pytest.raises(ValueError, match=expect):
        render.check_draft({"body": body}, tags, locales.load("zh"))


def test_parse_state_on_non_text_is_empty() -> None:
    """输入：非文本 state。输出：无。预期行为：给空结构，不抛异常。"""
    parsed = render.parse_state({"not": "text"})
    assert parsed.is_empty
    assert parsed.manifest is None
    assert parsed.rules == ()


def test_table_parsers_skip_malformed_rows() -> None:
    """输入：三张表里混进坏行。输出：无。预期行为：坏行跳过，好行照收。"""
    loc = locales.load("zh")
    state = (
        "【审查规则】（列名）\n"
        "- credential | 条件\n"
        "- 胡说 | 条件\n"
        "- credential | 条件 | 多了一格\n"
        "【严重度表】（列名）\n"
        "- credential | 3 | 2 | 3 | 0\n"
        "- debug | 中 | 0 | 2 | 0\n"
        "【门控表】（列名）\n"
        "- 3 | block | block | block\n"
        "- 5 | allow | allow | allow\n"
    )
    assert render.rules_of(state, loc) == (("credential", "条件"),)
    assert render.severity_of(state, loc) == (("credential", "3", "2", "3", "0"),)
    assert render.gate_of(state, loc) == (("3", "block", "block", "block"),)


def test_severity_rows_mark_out_of_scope_cells_as_not_raised() -> None:
    """输入：规范严重度表。输出：无。

    预期行为：不适用的格子写 0 档（文档上的凭据规则、测试上的调试规则），
    所以那张表自己就把「适用范围」说清楚了。
    """
    loc = locales.load("zh")
    rows = {row[0]: row for row in render.severity_rows(loc)}
    assert rows["credential"] == ("credential", "3", "2", "3", "0")
    assert rows["debug"] == ("debug", "2", "0", "2", "0")
    assert rows["size"] == ("size", "1", "0", "0", "0")
    assert rows["none"] == ("none", "0", "0", "0", "0")


def test_gate_rows_run_from_the_heaviest_level_down() -> None:
    """输入：规范门控表。输出：无。预期行为：行序是从重到轻，最后一行是「不提」。"""
    loc = locales.load("zh")
    assert [row[0] for row in render.gate_rows(loc)] == ["3", "2", "1", "0"]


def test_label_leak_ignores_empty_text() -> None:
    """输入：空文本。输出：无。预期行为：不报段落名。"""
    assert render.label_leak("   \n  ", locales.load("zh")) == ""


def test_cells_returns_none_for_a_bare_prefix() -> None:
    """输入：不是数据行的几行。输出：无。预期行为：只有带前缀且有内容的行才算数据行。"""
    loc = locales.load("zh")
    assert render._cells("- ", loc) is None
    assert render._cells("不是数据行", loc) is None
    assert render._cells("- a | b", loc) == ("a", "b")


def test_accessors_tolerate_non_text_state() -> None:
    """输入：非文本 state。输出：无。预期行为：各访问器给空值，不抛异常。"""
    loc = locales.load("zh")
    assert render.body_lines(123, loc) == []
    assert render.body_of(123, loc) == ""
    assert render.manifest_of(None, loc) is None
    assert render.rules_of(None, loc) == ()
    assert render.severity_of(None, loc) == ()
    assert render.gate_of(None, loc) == ()


def test_severity_and_gate_skip_rows_with_the_wrong_shape() -> None:
    """输入：三张表里混进形状不对的行。输出：无。预期行为：跳过，好行照收。"""
    loc = locales.load("zh")
    state = (
        "【严重度表】（列名）\n"
        "- credential | 3 | 2\n"
        "- 胡说 | 3 | 2 | 3 | 0\n"
        "- credential | 3 | 2 | 3 | 0\n"
        "【门控表】（列名）\n"
        "- x | allow | allow | allow\n"
        "- 3 | allow | allow | nope\n"
        "- 3 | allow | block | block\n"
    )
    assert render.severity_of(state, loc) == (("credential", "3", "2", "3", "0"),)
    assert render.gate_of(state, loc) == (("3", "allow", "block", "block"),)


def test_parse_state_of_an_unrelated_text_is_empty() -> None:
    """输入：一段跟本 workflow 无关的文本。输出：无。预期行为：给空结构。"""
    parsed = render.parse_state("这就是一段普通文本，没有任何段落头。")
    assert parsed.is_empty
    assert parsed.manifest is None
    assert parsed.severity == ()
