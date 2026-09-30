"""结构恢复的语言表测试：键与特征层对齐、标记跨语言不重叠、合并读法。"""

from __future__ import annotations

from itertools import combinations

import pytest

from jc.jsontypes import JsonValue
from workflows.autoformat import features, locales, parse, questions


def test_available_lists_every_locale_file() -> None:
    """输入：无。输出：无。预期行为：可用语言就是 `locales/` 下那两个 toml。"""
    assert locales.available() == ["en", "zh"]


def test_load_is_cached_and_reports_missing_locales() -> None:
    """输入：两次读同一语言、一次读不存在的语言。输出：无。

    预期行为：同一语言是**同一个对象**（结果带缓存）；缺语言时报人话并列出可用语言。
    """
    assert locales.load("zh") is locales.load("zh")
    with pytest.raises(FileNotFoundError, match="en, zh"):
        locales.load("ja")


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_tables_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：块类型表、提示框表、主题表、命令族表与特征层的取值表**一一对应**（键集合相等）；
    文案类字段全部非空。
    """
    loc = locales.load(lang)
    assert sorted(loc["type"]) == sorted(features.BLOCK_KINDS)
    assert sorted(loc["type_desc"]) == sorted(features.BLOCK_KINDS)
    assert sorted(loc["callout_kind"]) == sorted(features.CALLOUT_KINDS)
    assert sorted(loc["callout_word"]) == sorted(features.CALLOUT_KINDS)
    assert sorted(loc["topic_name"]) == sorted(features.TOPICS)
    families = sorted(value for value in features.ALL_CODE_FAMILIES if value != "na")
    assert sorted(loc["code_tool"]) == families
    assert sorted(loc["code_family"]) == families
    assert sorted(loc["rule"]) == sorted(
        {
            "heading",
            "paragraph",
            "list_item_ordered",
            "list_item_bullet",
            "code",
            "quote",
            "callout",
        }
    )
    assert sorted(loc["fam"]) == sorted(questions.FAMILIES)
    assert sorted(loc["limit"]) == sorted(parse.LIMIT_KEYS)
    for name in ("type", "type_desc", "callout_kind", "topic_name", "code_family", "rule"):
        assert all(str(value).strip() for value in loc[name].values()), name
    for name in ("doc", "verify"):
        assert all(str(value).strip() for value in loc[name].values()), name


def _limits_of(**overrides: JsonValue) -> dict[str, JsonValue]:
    """输入：要覆盖的节。输出：一份 `[limit]` 表（以中文那份为底）。预期行为：不碰缓存里的原表。"""
    raw = locales.load("zh")["limit"]
    assert isinstance(raw, dict)
    table: dict[str, JsonValue] = {key: value for key, value in raw.items()}
    table.update(overrides)
    return table


def _loc_with_limit(table: JsonValue) -> dict[str, JsonValue]:
    """输入：一份 `[limit]` 表。输出：中文语言字典的**副本**（只换掉 `[limit]`）。"""
    loc: dict[str, JsonValue] = {key: value for key, value in locales.load("zh").items()}
    loc["limit"] = table
    return loc


def test_limits_reads_the_table() -> None:
    """输入：中文那份 `[limit]`。输出：无。预期行为：读成五组区间，且能按块类型取用。"""
    lim = parse.limits(_loc_with_limit(_limits_of()))
    assert lim.heading == (6, 24)
    assert lim.item == (6, 28)
    assert lim.prose == (74, 240)
    assert lim.quote == (24, 120)
    assert lim.callout == (46, 160)
    assert lim.range_of(features.KIND_HEADING) == (6, 24)
    assert lim.range_of(features.KIND_CALLOUT) == (46, 160)
    with pytest.raises(ValueError, match="没有字符预算"):
        lim.range_of(features.KIND_CODE)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_topic_names_are_legal_headings(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每一个主题名本身都是一句**合法的标题**——长度落在 `[limit].heading` 里、不带结尾标点、
    不含步骤词 / 提示框关键词 / 署名前缀。

    为什么这算硬约束：主题名写在提示词的开头（「这份备忘的主题是……」，提示词里就那一句现成的短语），
    模型会顺手把它抄成标题。真机上英文的主题名 37 个字符、标题上限 36，10 条里 6 条就这么废掉了。
    """
    loc = locales.load(lang)
    low, high = parse.limits(loc).heading
    banned = [
        *parse.seq_words(loc),
        *parse.callout_words(loc),
        *parse.attribution_marks(loc),
    ]
    for topic, name in loc["topic_name"].items():
        text = str(name)
        assert low <= len(text) <= high, (topic, text)
        assert not parse.ends_terminal(text), (topic, text)
        assert not any(word in text for word in banned), (topic, text)


def test_limits_rejects_a_missing_or_unknown_section() -> None:
    """输入：缺一节的表、多一节表、根本不是表的表。输出：无。

    预期行为：三种都当场报人话——多出来的那一节没人读，写在那里只会让人以为它生效了。
    """
    short = _limits_of()
    del short["quote"]
    with pytest.raises(ValueError, match="缺了这几节"):
        parse.limits(_loc_with_limit(short))
    with pytest.raises(ValueError, match="多了这几节"):
        parse.limits(_loc_with_limit(_limits_of(wrap=[1, 2])))
    with pytest.raises(ValueError, match="不是一个表"):
        parse.limits(_loc_with_limit(["6", "24"]))
    with pytest.raises(ValueError, match="不是一个表"):
        parse.limits({key: value for key, value in locales.load("zh").items() if key != "limit"})


@pytest.mark.parametrize(
    ("overrides", "want"),
    [
        ({"heading": [6]}, "必须是 \\[下限, 上限\\]"),
        ({"heading": "6-24"}, "必须是 \\[下限, 上限\\]"),
        ({"heading": [6, 24.0]}, "必须是整数"),
        ({"heading": [True, 24]}, "必须是整数"),
        ({"heading": [24, 6]}, "0 < 下限 < 上限"),
        ({"heading": [0, 6]}, "0 < 下限 < 上限"),
        ({"heading": [6, 45]}, "超过了短行阈值"),
        ({"item": [6, 45]}, "超过了短行阈值"),
        ({"prose": [60, 240]}, "不够长"),
        ({"callout": [30, 160]}, "不高于短行阈值"),
    ],
)
def test_limits_rejects_budgets_that_break_the_reading_rules(
    overrides: dict[str, JsonValue], want: str
) -> None:
    """输入：十种写歪的预算。输出：无。

    预期行为：每一处与反解判据挂钩的越界都当场报人话——预算写歪了，判据会跟着悄悄跑偏，
    而跑偏的判据只会把干净数据判成脏数据。
    """
    with pytest.raises(ValueError, match=want):
        parse.limits(_loc_with_limit(_limits_of(**overrides)))


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_callout_words_are_disjoint_across_kinds(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：提示框关键词跨种类互不重复——反解就是靠它认种类的；每格也非空。
    """
    loc = locales.load(lang)
    pools = {kind: [str(word) for word in loc["callout_word"][kind]] for kind in features.CALLOUT_KINDS}
    for kind, words in pools.items():
        assert words, f"{lang}.{kind}"
    flattened = [word for words in pools.values() for word in words]
    assert len(set(flattened)) == len(flattened)
    assert sorted(parse.callout_index(loc)) == sorted(flattened)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_step_words_are_disjoint_from_the_other_pools(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：步骤词非空、两两不同，且与提示框关键词、署名前缀**都不重合**——重合会让
    「这一行是不是列表项」和「这一块是不是提示框」这两条判据互相打架。
    """
    loc = locales.load(lang)
    steps = parse.seq_words(loc)
    assert steps
    assert len(set(steps)) == len(steps)
    assert not (set(steps) & set(parse.callout_words(loc)))
    assert not (set(steps) & set(parse.attribution_marks(loc)))


def test_markers_never_overlap_across_locales() -> None:
    """输入：两种语言。输出：无。

    预期行为：提示框关键词、步骤词、署名前缀**跨语言互不重叠**。这条是合并读法成立的前提：
    `Workflow.reparse` 没有语言参数，只能拿「所有语言的标记一起认」。
    """
    pools = {
        "callout": [set(parse.callout_words(locales.load(lang))) for lang in locales.available()],
        "seq": [set(parse.seq_words(locales.load(lang))) for lang in locales.available()],
        "mark": [set(parse.attribution_marks(locales.load(lang))) for lang in locales.available()],
    }
    for name, groups in pools.items():
        for left, right in combinations(groups, 2):
            assert not (left & right), f"{name} 在两种语言里重叠：{left & right}"


def test_merged_merges_tables_by_subkey_and_lists_flat() -> None:
    """输入：无。输出：无。

    预期行为：`code_tool` / `callout_word` / `seq_word` 按子键合（两边同一套键名），
    `mark.attribution` 直接拼；重复的工具名只留一份。
    """
    merged = locales.merged()
    en, zh = locales.load("en"), locales.load("zh")
    assert sorted(merged["code_tool"]) == sorted(zh["code_tool"])
    for family, pool in merged["code_tool"].items():
        assert pool == zh["code_tool"][family] == en["code_tool"][family]
        assert len(set(pool)) == len(pool)
    for kind in features.CALLOUT_KINDS:
        both = [*zh["callout_word"][kind], *en["callout_word"][kind]]
        assert sorted(merged["callout_word"][kind]) == sorted(both)
    assert sorted(merged["seq_word"]["step"]) == sorted(
        [*zh["seq_word"]["step"], *en["seq_word"]["step"]]
    )
    marks = merged["mark"]["attribution"]
    assert set(marks) == set(zh["mark"]["attribution"]) | set(en["mark"]["attribution"])
    assert len(set(marks)) == len(marks)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_merged_read_agrees_with_the_single_locale_read(lang: str) -> None:
    """输入：一段本语言的 state。输出：无。

    预期行为：合并读法读出来的块类型与单语言读法**逐项相同**——两种语言的标记互不重叠，
    多认一套标记不该把本语言的块读歪。
    """
    from tests.workflows.autoformat.conftest import texts_for
    from workflows.autoformat import render

    case = features.DocCase("build", 64, "mixed", "note", "vcs")
    plan = features.build_plan(case)
    loc = locales.load(lang)
    state = render.render_state(plan, texts_for(plan, loc), loc)
    assert parse.parse_state(state, loc).kinds() == parse.parse_state(state, locales.merged()).kinds()


def test_dig_reports_a_broken_path_in_plain_words() -> None:
    """输入：几条走不通的键路径。输出：无。

    预期行为：走到头就把值取回来；中途缺键、或中途那一节不是表，都报人话并点名路径——
    悄悄跳过会让「某份 toml 少写一张表」变成反解阶段的怪错误。
    """
    table: dict[str, JsonValue] = {"code_tool": {"build": ["make"]}}
    assert locales._dig(table, ("code_tool", "build"), "zh") == ["make"]
    with pytest.raises(ValueError, match="缺了一节"):
        locales._dig({"code_tool": {}}, ("code_tool", "build"), "zh")
    with pytest.raises(ValueError, match="缺了一节"):
        locales._dig({"code_tool": "not a table"}, ("code_tool", "build"), "zh")


def test_merged_reports_a_broken_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：一份把 `code_tool` 写成字符串的语言表。输出：无。

    预期行为：合并时当场报人话、点名是哪张表（不静默跳过，也不拖到反解阶段才炸）。
    """
    monkeypatch.setattr(locales, "load", lambda lang: {"code_tool": "not a table"})
    locales.merged.cache_clear()
    try:
        with pytest.raises(ValueError, match="不是一个表"):
            locales.merged()
    finally:
        monkeypatch.undo()
        locales.merged.cache_clear()
    assert locales.merged()["code_tool"]
