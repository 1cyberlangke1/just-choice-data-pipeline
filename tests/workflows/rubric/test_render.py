"""打分与评分的文本外壳测试：渲染、反解、形状校验与两处禁写检查。"""

from __future__ import annotations

import pytest

from workflows.rubric import features, locales, render
from workflows.rubric.features import TagSet

from tests.workflows.rubric.conftest import FILLERS, body_for


def tags_of(
    placeholders: str = "no",
    citations: str = "no",
    length: str = "short",
    form: str = "prose",
    data: str = "none",
) -> TagSet:
    """输入：五维取值（都有默认）。输出：一组注入特征。预期行为：只用于构造测试输入。"""
    return TagSet(
        placeholders=placeholders,
        citations=citations,
        length=length,
        form=form,
        data=data,
    )


def state_for(tags: TagSet, locale: str = "zh") -> str:
    """输入：注入特征、语言。输出：拼好的 state。预期行为：走渲染期同一套校验。"""
    loc = locales.load(locale)
    body = body_for(tags.as_dict())
    render.check_draft({"body": body}, tags, loc)
    return render.to_state({"body": body}, tags, loc)


def test_round_trip_for_every_combination() -> None:
    """输入：全空间 48 组特征。输出：无。

    预期行为：每一格都能渲染出 state 并**原样反解回来**——清单行逐格相等、材料逐行
    相等、三张表等于规范表。这是「渲染必须可反解」的最小落地。
    """
    loc = locales.load("zh")
    canonical = (render.band_rows(), render.gate_rows())
    for tags in features.build_cases(2048):
        body = body_for(tags.as_dict())
        render.check_draft({"body": body}, tags, loc)
        state = render.to_state({"body": body}, tags, loc)
        parsed = render.parse_state(state)
        assert parsed.manifest == render.ManifestState(
            length=tags.length, form=tags.form, data=tags.data
        )
        assert list(parsed.body_lines) == body.splitlines()
        assert parsed.score == render.score_rows(tags, loc)
        assert (parsed.band, parsed.gate) == canonical


def test_state_has_the_five_sections_in_order() -> None:
    """输入：一段 state。输出：无。预期行为：段落顺序固定，材料独占段落头下面那几行。"""
    tags = tags_of(placeholders="yes", length="medium")
    state = state_for(tags)
    loc = locales.load("zh")
    heads = [str(loc["meta"][key]) for key in render.SECTION_KEYS]
    positions = [state.index(f"【{title}】") for title in heads]
    assert positions == sorted(positions)
    assert state.splitlines()[0] == "【待评材料】"


def test_to_state_keeps_the_leading_whitespace_of_the_model_text() -> None:
    """输入：行首带缩进的材料。输出：无。

    预期行为：反解时**只去尾部空白**——行首那几个空格是模型写的原文，而「分条」的判定
    要在 `lstrip()` 之后看，去掉行首会把它读成另一回事。
    """
    tags = tags_of(length="medium")
    loc = locales.load("zh")
    body = "  indented opening line\nplain second line\nplain third line\nplain fourth\nplain fifth\nplain sixth"
    render.check_draft({"body": body}, tags, loc)
    parsed = render.parse_state(render.to_state({"body": body}, tags, loc))
    assert list(parsed.body_lines) == body.splitlines()
    assert parsed.body_lines[0].startswith("  indented")


def test_a_slightly_overlong_body_is_accepted() -> None:
    """输入：行数略超写作指引上界的材料。输出：无。

    预期行为：验收上界是宽容过的（`features.LENGTH_LINES_SLACK`）——清单那一行才是权威，
    按写作指引的上界硬拒是白拒好数据（范式 §9.19）。
    """
    tags = tags_of(length="short")
    loc = locales.load("zh")
    body = "\n".join(list(FILLERS) * 2)
    render.check_draft({"body": body}, tags, loc)
    parsed = render.parse_state(render.to_state({"body": body}, tags, loc))
    assert len(parsed.body_lines) == 12
    with pytest.raises(ValueError, match="句"):
        render.check_draft({"body": f"{body}\n{FILLERS[0]}"}, tags, loc)


def test_a_single_paragraph_body_is_accepted() -> None:
    """输入：一整个自然段（真机第一批的写法）。输出：无。

    预期行为：收下——材料是自然语言，**句子**才是它的自然单位；按「行数」验收会把模型
    写成一整段的正文全部白拒（真机第一批 20 条里 16 条就是这么被拒的）。
    """
    tags = tags_of(length="short", data="numbers")
    loc = locales.load("zh")
    body = "缓存换掉了，平均响应时间从 200 毫秒降到 80 毫秒。吞吐量提升了约 30%。峰值流量也扛住了。"
    render.check_draft({"body": body}, tags, loc)
    parsed = render.parse_state(render.to_state({"body": body}, tags, loc))
    assert len(parsed.body_lines) == 1
    assert features.count_sentences(parsed.body) == 3


def test_english_state_parses_too() -> None:
    """输入：英文 state。输出：无。预期行为：反解与语言无关（各语言段落名都试一遍）。"""
    tags = tags_of(placeholders="yes", citations="yes", length="medium")
    state = state_for(tags, "en")
    parsed = render.parse_state(state)
    assert parsed.manifest is not None
    assert parsed.manifest.length == "medium"
    assert parsed.body_lines


def test_same_inputs_give_byte_identical_states() -> None:
    """输入：同一条样本渲染两次。输出：两段 state。预期行为：逐字节一致。"""
    tags = tags_of(citations="yes", length="long")
    assert state_for(tags) == state_for(tags)


@pytest.mark.parametrize(
    ("body", "hint"),
    [
        ("", "空的"),
        ("   \n  ", "空的"),
        ("```\nline one\nline two\nline three\n```", "围栏"),
        ("待评材料：line one\nline two\nline three", "段落名"),
        ("line one\n见材料清单\nline three", "表名"),
    ],
)
def test_check_draft_rejects_broken_drafts(body: str, hint: str) -> None:
    """输入：各种坏草稿。输出：无。预期行为：渲染期当场抛错，理由点出坏在哪。"""
    tags = tags_of(length="short")
    with pytest.raises(ValueError, match=hint):
        render.check_draft({"body": body}, tags, locales.load("zh"))


def test_check_draft_rejects_shape_mismatches() -> None:
    """输入：与注入特征对不上的材料。输出：无。预期行为：逐条都被拦下。"""
    loc = locales.load("zh")
    three_lines = "alpha one\nbravo two\ncharlie three"
    cases = [
        (three_lines, tags_of(length="long"), "句"),
        (three_lines, tags_of(form="bulleted"), "分条"),
        ("- alpha\n- bravo\n- charlie", tags_of(form="prose"), "行文"),
        (three_lines, tags_of(data="numbers"), "数据"),
        ("alpha 1\nbravo 2\ncharlie 3", tags_of(data="none"), "数据"),
        (three_lines, tags_of(placeholders="yes"), "留空占位"),
        ("TODO: alpha\nbravo\ncharlie", tags_of(placeholders="no"), "留空占位"),
        (three_lines, tags_of(citations="yes"), "出处标记"),
        ("alpha[1]\nbravo\ncharlie", tags_of(citations="no"), "出处标记"),
    ]
    for body, tags, hint in cases:
        with pytest.raises(ValueError, match=hint):
            render.check_draft({"body": body}, tags, loc)


def test_label_leak_finds_a_copied_header() -> None:
    """输入：开头抄了段落名的文本。输出：无。预期行为：报出那个段落名。"""
    loc = locales.load("zh")
    assert render.label_leak("待评材料：line one", loc) == "待评材料"
    assert render.label_leak("【材料清单】line one", loc) == ""
    assert render.label_leak("line one", loc) == ""
    assert render.label_leak("   \n  ", loc) == ""


def test_banned_echo_finds_table_names() -> None:
    """输入：含表名的文本。输出：无。预期行为：命中，且大小写不敏感。"""
    zh = locales.load("zh")
    en = locales.load("en")
    assert render.banned_echo("line one 见评分单", zh) == ["评分单"]
    assert render.banned_echo("line one", zh) == []
    assert render.banned_echo("see the Score Sheet", en) == ["score sheet"]


def test_banned_echo_leaves_prose_alone() -> None:
    """输入：一段满是 length / form / data 的散文。输出：无。

    预期行为：一个都不报——材料是自然语言，这些词在正常叙述里到处都是，收进禁写清单
    会大面积误伤（宁可漏检也不误判）。
    """
    zh = locales.load("zh")
    prose = "The length of the form matters, and so does the data behind it."
    assert render.banned_echo(prose, zh) == []


def test_manifest_of_rejects_unknown_tokens() -> None:
    """输入：词元不在取值表里的清单行。输出：无。预期行为：当解不出来。"""
    loc = locales.load("zh")
    state = "【待评材料】\nline one\n【材料清单】（列名）\n- huge | prose | none"
    assert render.manifest_of(state, loc) is None


def test_manifest_of_rejects_a_row_with_the_wrong_cell_count() -> None:
    """输入：只有两格的清单行。输出：无。预期行为：当解不出来。"""
    loc = locales.load("zh")
    state = "【待评材料】\nline one\n【材料清单】（列名）\n- short | prose"
    assert render.manifest_of(state, loc) is None


def test_table_parsers_skip_malformed_rows() -> None:
    """输入：三张表里混进坏行。输出：无。预期行为：坏行跳过，好行照收。"""
    loc = locales.load("zh")
    state = (
        "【评分单】（列名）\n"
        "- complete | 3 | 3 | 4\n"
        "- 胡说 | 3 | 3 | 4\n"
        "- complete | 3 | 3\n"
        "【档位表】（列名）\n"
        "- low | 0 | 5\n"
        "- low | 一百 | 五\n"
        "【处置表】（列名）\n"
        "- high | accept | revise\n"
        "- high | accept | nope\n"
    )
    assert render.score_of(state, loc) == (("complete", "3", "3", "4"),)
    assert render.band_of(state, loc) == (("low", "0", "5"),)
    assert render.gate_of(state, loc) == (("high", "accept", "revise"),)


def test_score_rows_print_a_mark_for_the_body_dimensions() -> None:
    """输入：规范评分单。输出：无。

    预期行为：正文承载的两条印「按材料数」（读者只能去材料里数），其余三条印观测数字；
    权重与要求两列取自固定表。
    """
    loc = locales.load("zh")
    rows = {row[0]: row for row in render.score_rows(tags_of(placeholders="yes", citations="no"), loc)}
    mark = str(loc["meta"]["count_mark"])
    assert rows["complete"] == ("complete", "3", "3", mark)
    assert rows["grounded"] == ("grounded", "3", "2", mark)
    assert rows["concise"] == ("concise", "2", "2", "4")
    assert rows["formatted"] == ("formatted", "2", "2", "1")
    assert rows["quantified"] == ("quantified", "1", "2", "0")


def test_band_rows_and_gate_rows_are_canonical() -> None:
    """输入：两张规范表。输出：无。预期行为：行序与格子都取自 `features` 的固定表。"""
    assert render.band_rows() == (("low", "0", "5"), ("mid", "6", "7"), ("high", "8", "11"))
    assert render.gate_rows() == (
        ("low", "revise", "reject"),
        ("mid", "accept", "reject"),
        ("high", "accept", "revise"),
    )


def test_obs_from_score_reads_both_forms() -> None:
    """输入：评分单行与材料。输出：无。

    预期行为：观测那一格是数字就直接用、是标记就从材料里数；判据名不认识或行数不齐
    一律给 None——猜一个观测会让审计变成橡皮图章。
    """
    loc = locales.load("zh")
    tags = tags_of(placeholders="yes", citations="yes")
    rows = render.score_rows(tags, loc)
    marks = render.count_marks()
    obs = render.obs_from_score(rows, "TODO: 补上压测数据，比上版快了一倍[1]", marks)
    assert obs == features.obs_map(tags)
    assert render.obs_from_score((), "text", marks) is None
    assert render.obs_from_score((("nope", "1", "1", "1"),), "text", marks) is None
    assert render.obs_from_score((("complete", "1", "1", "??"),), "text", marks) is None


def test_count_obs_only_works_for_body_dimensions() -> None:
    """输入：五个维度名。输出：无。

    预期行为：正文承载的两维数得出来；其余三维的观测印在评分单上，数不出来就该报错
    ——静默返回一个数字会让「谁负责这一维」这件事糊掉。
    """
    assert render.count_obs("placeholders", "TODO: 补上") == 0
    assert render.count_obs("placeholders", "写完了") == 4
    assert render.count_obs("citations", "快了一倍[1]") == 4
    assert render.count_obs("citations", "写完了") == 1
    for dim in ("length", "form", "data"):
        with pytest.raises(KeyError):
            render.count_obs(dim, "写完了")


def test_accessors_tolerate_non_text_state() -> None:
    """输入：非文本 state。输出：无。预期行为：各访问器给空值，不抛异常。"""
    loc = locales.load("zh")
    assert render.body_lines(123, loc) == []
    assert render.body_of(123, loc) == ""
    assert render.manifest_of(None, loc) is None
    assert render.score_of(None, loc) == ()
    assert render.band_of(None, loc) == ()
    assert render.gate_of(None, loc) == ()


def test_cells_returns_none_for_a_bare_prefix() -> None:
    """输入：不是数据行的几行。输出：无。预期行为：只有带前缀且有内容的行才算数据行。"""
    loc = locales.load("zh")
    assert render._cells("- ", loc) is None
    assert render._cells("不是数据行", loc) is None
    assert render._cells("- a | b", loc) == ("a", "b")


def test_parse_state_on_non_text_and_unrelated_text_is_empty() -> None:
    """输入：非文本 state、一段无关文本。输出：无。预期行为：都给空结构，不抛异常。"""
    parsed = render.parse_state({"not": "text"})
    assert parsed.is_empty
    assert parsed.manifest is None
    assert parsed.score == ()
    other = render.parse_state("这就是一段普通文本，没有任何段落头。")
    assert other.is_empty
    assert other.manifest is None
    assert other.band == ()


def test_build_prompt_lists_every_dimension_in_order() -> None:
    """输入：一组特征。输出：提示词。预期行为：五维按固定顺序逐行列出，规则全在。"""
    loc = locales.load("zh")
    tags = tags_of(placeholders="yes", citations="yes", length="long", form="bulleted", data="numbers")
    prompt = render.build_prompt(tags, loc, 0)
    positions = [prompt.index(str(loc["dims"][dim])) for dim in render.DIM_ORDER]
    assert positions == sorted(positions)
    for rule in loc["prompt"]["rules"]:
        assert str(rule).format(language=loc["language_name"]) in prompt


def test_build_task_carries_the_tags_and_the_draft_model() -> None:
    """输入：一组特征。输出：`GenerationTask`。预期行为：题型 / 领域 / 草稿模型与 tags 都对。"""
    loc = locales.load("zh")
    tags = tags_of(citations="yes")
    task = render.build_task(tags, loc, "rubric-train-0000000-render")
    assert task.task_id == "rubric-train-0000000-render"
    assert task.kind == "choice"
    assert task.response_model is render.ScoreDraft
    assert task.spec == {"tags": tags.as_dict()}
    assert "body" in render.ScoreDraft.model_fields
