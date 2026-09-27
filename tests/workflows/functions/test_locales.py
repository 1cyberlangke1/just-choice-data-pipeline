"""语言文件：形状一致、短语互不包含、取值短语全局唯一、模板填得满。"""

from __future__ import annotations

from typing import Any

import pytest

from workflows.functions import locales, parse, render, spec as sp
from workflows.functions.workflow import FunctionsWorkflow

LANGS = tuple(locales.available())


def _keys(value: object, prefix: str = "") -> set[str]:
    """输入：语言文件里的一个值。输出：它下面全部键的路径集合（递归）。"""
    if not isinstance(value, dict):
        return set()
    out: set[str] = set()
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        out.add(path)
        out |= _keys(item, path)
    return out


def _val_groups(loc: dict[str, Any]) -> dict[tuple[str, str], tuple[str, ...]]:
    """输入：语言字典。输出：{(短语表键, 取值): 该取值的全部写法}（带别名的表先摊平）。"""
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    for key, table in loc["val"].items():
        if render.is_aliased(table):
            for value in table["ticker"]:
                out[(key, value)] = tuple(
                    str(table[alias][value]) for alias in ("ticker", "name")
                )
        else:
            for value, phrases in table.items():
                out[(key, value)] = render.as_phrases(phrases)
    return out


def _templates(loc: dict[str, Any]) -> list[str]:
    """输入：语言字典。输出：全部句子模板（请求分段 + 题面 + 判据）。"""
    out: list[str] = []
    for table in loc["req"].values():
        for value in table.values():
            out.extend([value] if isinstance(value, str) else value)
    for name, entry in loc["fam"].items():
        out.extend(entry["q"])
        out.extend(entry.get("criteria", {}).values())
    return out


def test_both_locales_have_the_same_shape() -> None:
    """输入：两个语言文件。输出：无。预期行为：键集合完全一致（少一格就会渲染时炸）。"""
    assert len(LANGS) >= 2
    base = _keys(locales.load(LANGS[0]))
    for lang in LANGS[1:]:
        assert _keys(locales.load(lang)) == base, lang


@pytest.mark.parametrize("lang", LANGS)
def test_intent_phrases_are_mutually_exclusive(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：意图短语两两**不互为子串**，且每个都非空——判别器靠「短语在不在文本里」
    认工具，一条是另一条的子串就会出现「一句话命中两个工具」。
    """
    loc = locales.load(lang)
    assert set(loc["intent"]) == set(sp.TOOLS)
    items = list(loc["intent"].items())
    for tool, phrase in items:
        assert phrase.strip(), tool
        assert parse.tool_of(f"前缀{phrase}后缀", loc) == tool
    for left, (tool_a, phrase_a) in enumerate(items):
        for tool_b, phrase_b in items[left + 1 :]:
            assert phrase_a not in phrase_b, (tool_a, tool_b)
            assert phrase_b not in phrase_a, (tool_a, tool_b)


@pytest.mark.parametrize("lang", LANGS)
def test_every_lead_carries_its_own_intent(lang: str) -> None:
    """输入：每种语言的全部 `lead` 模板。输出：无。

    预期行为：每个 `lead` 都含**自己**工具的意图短语——漏一条，那一档问法渲染出来的
    请求就读不出工具（实测踩过：英文问法 1 以大写字母开头，小写的意图短语匹配不上）。
    """
    loc = locales.load(lang)
    for tool, table in loc["req"].items():
        for template in table["lead"]:
            assert str(loc["intent"][tool]) in template, (tool, template)
            assert "{subject}" in template or not sp.SUBJECT_ARGS[tool]


@pytest.mark.parametrize("lang", LANGS)
def test_no_other_template_contains_an_intent_phrase(lang: str) -> None:
    """输入：每种语言里除 `lead` 以外的模板。输出：无。

    预期行为：分段模板、取值短语、flag 短语里**一个意图短语都不许出现**——否则另一族的
    请求会顺带命中它，`tool_of` 直接报「命中两个工具」。
    """
    loc = locales.load(lang)
    intents = [str(phrase) for phrase in loc["intent"].values()]
    pools: list[str] = []
    for table in loc["req"].values():
        pools.extend(value for key, value in table.items() if key != "lead")
    for phrases in _val_groups(loc).values():
        pools.extend(phrases)
    pools.extend(str(phrase) for phrase in loc["flag"].values())
    for template in pools:
        for phrase in intents:
            assert phrase not in template, (lang, phrase, template)


@pytest.mark.parametrize("lang", LANGS)
def test_value_phrases_are_globally_unique(lang: str) -> None:
    """输入：每种语言的 `[val]` 全表。输出：无。

    预期行为：任意两条取值短语**不互为子串**——读法是「短语在不在文本里」，
    一条是另一条的子串（或重复）就会让一个参数反解出两个取值。
    """
    groups = _val_groups(locales.load(lang))
    flat = [(phrase, key, value) for (key, value), phrases in groups.items() for phrase in phrases]
    for index, (phrase, key, value) in enumerate(flat):
        for other, other_key, other_value in flat[index + 1 :]:
            if (key, value) == (other_key, other_value):
                continue
            assert phrase not in other, (lang, phrase, other, key, value, other_key, other_value)
            assert other not in phrase, (lang, other, phrase, other_key, other_value, key, value)


@pytest.mark.parametrize("lang", LANGS)
def test_every_value_is_covered_by_val_and_opt(lang: str) -> None:
    """输入：每种语言的 `[val]` 与 `[opt]`。输出：无。

    预期行为：工具表里每个参数的每个取值都有短语（否则渲染时炸）和选项描述
    （否则出题时炸），且选项描述在同一题里互不相同。
    """
    loc = locales.load(lang)
    groups = _val_groups(loc)
    for arg in sp.ARG_BY_NAME.values():
        if not arg.values:
            continue  # flag 参数没有取值表，自然也没有 `[val]` / `[opt]` 条目
        key = sp.phrase_key(arg)
        for value in arg.values:
            assert (key, value) in groups, (lang, key, value)
            assert value in loc["opt"][key], (lang, key, value)
        assert len(set(loc["opt"][key].values())) == len(loc["opt"][key]), (lang, key)


@pytest.mark.parametrize("lang", LANGS)
def test_alias_spellings_reverse_lookup_to_the_same_value(lang: str) -> None:
    """输入：每种语言的别名表。输出：无。

    预期行为：写代码与写中文名都能反查回同一个取值——两套写法只影响措辞，不影响答案。
    """
    loc = locales.load(lang)
    for key in ("symbol", "benchmark"):
        table = loc["val"][key]
        assert set(table["ticker"]) == set(table["name"]), (lang, key)
        for value in table["ticker"]:
            for alias in ("ticker", "name"):
                text = f"x{render.value_phrase(loc, key, value, alias)}y"
                assert parse.stated_values(text, key, tuple(table["ticker"]), loc) == (value,)


@pytest.mark.parametrize("lang", LANGS)
def test_alias_spellings_actually_differ(lang: str) -> None:
    """输入：每种语言的两种写法。输出：无。

    预期行为：同一个取值的两套写法**不是同一个字符串**——一样的话 `alias` 这一维在文本里
    就看不出来，两个「不同」的 case 会渲染出同一句话，切分之间立刻撞车。
    """
    loc = locales.load(lang)
    for key in ("symbol", "benchmark"):
        table = loc["val"][key]
        for value in table["ticker"]:
            assert str(table["ticker"][value]) != str(table["name"][value]), (lang, key, value)


@pytest.mark.parametrize("lang", LANGS)
def test_lead_phrasings_differ(lang: str) -> None:
    """输入：每种语言每个工具的 `lead` 池。输出：无。

    预期行为：两套句面渲染出来**不一样**——一样的话 `phrasing` 这一维在文本里看不出来，
    留出就不成立，两个 case 还会渲染出同一句话。
    """
    loc = locales.load(lang)
    for tool, table in loc["req"].items():
        leads = [str(template) for template in table["lead"]]
        assert len(leads) == len(set(leads)), tool
        assert len(leads) >= len(sp.PHRASINGS), tool


@pytest.mark.parametrize("lang", LANGS)
def test_flag_phrases_never_collide_with_values_or_templates(lang: str) -> None:
    """输入：每种语言的 flag 短语。输出：无。

    预期行为：出现短语不许出现在任何取值短语或模板里——否则这个开关会被误判为在场。
    """
    loc = locales.load(lang)
    values = [phrase for phrases in _val_groups(loc).values() for phrase in phrases]
    templates = _templates(loc)
    for name, phrase in loc["flag"].items():
        assert str(phrase).strip(), name
        for other in values + templates:
            assert str(phrase) not in other, (lang, name, phrase, other)


@pytest.mark.parametrize("lang", LANGS)
def test_every_tool_and_arg_has_a_description(lang: str) -> None:
    """输入：每种语言的说明表。输出：无。

    预期行为：每个工具有 `[tool_desc]`、每个参数有 `[arg_desc]`（题面里点名它），
    且描述互不相同（选项描述一样就分不出选项）。
    """
    loc = locales.load(lang)
    assert set(loc["tool_desc"]) == set(sp.TOOLS)
    assert len(set(loc["tool_desc"].values())) == len(sp.TOOLS)
    assert set(loc["arg_desc"]) == set(sp.ARG_BY_NAME)
    assert len(set(loc["arg_desc"].values())) == len(sp.ARG_BY_NAME)


@pytest.mark.parametrize("lang", LANGS)
def test_family_prompts_cover_every_phrasing(lang: str) -> None:
    """输入：每种语言的题面池。输出：无。

    预期行为：每个题族都有题面、每个问法编号都取得到一条，`stated` / `member` 的判据
    真值两行都写了。
    """
    loc = locales.load(lang)
    assert set(loc["fam"]) == {"tool", "arg", "stated", "member"}
    for name, entry in loc["fam"].items():
        assert entry["q"], name
        for phrasing in sp.PHRASINGS:
            assert locales.pick(entry["q"], phrasing), (name, phrasing)
        if name in ("stated", "member"):
            assert set(entry["criteria"]) == {"true", "false"}, name


def test_marks_are_declared() -> None:
    """输入：标点表。输出：无。预期行为：收尾标点与列举连接词都在，且两种语言不同。"""
    for lang in LANGS:
        marks = locales.load(lang)["mark"]
        assert set(marks) == {"close", "list_sep", "list_last"}
        assert all(str(value) for value in marks.values())


def test_merged_phrase_groups_stay_unambiguous() -> None:
    """输入：合并表。输出：无。

    预期行为：两种语言的取值短语并到一起之后，同一参数的不同取值仍然互不包含——
    否则 `reparse`（没有语言参数）会反解出两个取值。
    """
    merged = locales.merged()
    for key, table in merged["val"].items():
        flat = [(value, phrase) for value, phrases in table.items() for phrase in phrases]
        for index, (value, phrase) in enumerate(flat):
            for other_value, other in flat[index + 1 :]:
                if value == other_value:
                    continue
                assert phrase not in other, (key, value, phrase, other_value, other)
                assert other not in phrase, (key, other_value, other, value, phrase)


def test_merged_table_reads_both_languages(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：合并表。输出：无。

    预期行为：两种语言渲染出来的文本都能被同一张合并表读回来，且意图短语是并集。
    """
    merged = locales.merged()
    for lang in LANGS:
        loc = locales.load(lang)
        state = render.render_state(sp.sample_spec(make_case(), "train", 0), loc)
        assert parse.read_request(state, merged).tool == "plot_price"
    for tool, phrases in merged["intent"].items():
        assert len(phrases) == len(LANGS), tool


def test_missing_locale_lists_the_available_ones() -> None:
    """输入：不存在的语言。输出：无。预期行为：报错并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="可用"):
        locales.load("fr")


def test_available_locales_match_the_workflow() -> None:
    """输入：无。输出：无。预期行为：workflow 报的语言列表就是磁盘上的语言文件。"""
    assert FunctionsWorkflow().available_locales() == list(LANGS)


def test_pick_wraps_around() -> None:
    """输入：越界编号。输出：无。预期行为：回绕而不是报错（题面措辞不是判据）。"""
    assert locales.pick(["a", "b"], 3) == "b"
    assert locales.pick(["a", "b"], 0) == "a"
