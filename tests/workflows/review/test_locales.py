"""文案表的不变量：这些不是「文案好不好」，而是「答案对不对」的前提。"""

from __future__ import annotations

import pytest

from workflows.review import locales, parse
from workflows.review import spec as sp

LANGS = locales.available()


def _used_body_keys() -> dict[str, str | None]:
    """输入：无。输出：{正文键: 它该命中的规则}（干净槽给 None）。"""
    out: dict[str, str | None] = {}
    for case in sp.build_cases(sp.ReviewConfig()):
        for row in sp.rows_of(case):
            out[row.body] = row.rule
    return out


def test_available_locales_are_sorted_and_nonempty() -> None:
    """输入：无。输出：无。预期行为：至少两种语言，顺序稳定。"""
    assert LANGS == sorted(LANGS)
    assert {"zh", "en"} <= set(LANGS)


def test_unknown_locale_reports_available_ones() -> None:
    """输入：不存在的语言。输出：无。预期行为：报错并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="没有语言文件"):
        locales.load("xx")


def test_pick_wraps_around() -> None:
    """输入：长度为 2 的池、编号 3。输出：无。预期行为：回绕到第 1 个（不报错）。"""
    assert locales.pick(["a", "b"], 3) == "b"
    assert locales.pick(["a", "b"], 4) == "a"


@pytest.mark.parametrize("lang", LANGS)
def test_locales_share_the_same_shape(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：语言文件的**键集合完全一致**——少一格会让那种语言直接跑不动，少一道问法
    会让 `pick` 回绕到别的句面上（答案照样发出去，只是题变得莫名其妙）。
    """
    base = locales.load("zh")
    other = locales.load(lang)
    assert set(other["meta"]) == set(base["meta"])
    for table in ("file", "change", "size", "body"):
        assert set(other[table]) == set(base[table])
    assert set(other["fam"]) == set(base["fam"])
    for family in base["fam"]:
        assert len(other["fam"][family]["q"]) == len(base["fam"][family]["q"])
        assert set(other["fam"][family]) == set(base["fam"][family])
    assert set(other["option"]) == set(base["option"])
    for table in base["option"]:
        assert set(other["option"][table]) == set(base["option"][table])
    assert set(other["policy"]) == set(base["policy"])


@pytest.mark.parametrize("lang", LANGS)
def test_vocabularies_cover_the_spec_constants(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：每张词表的键就是 spec 里的常量——少一个取值会让渲染 `KeyError`，
    多一个取值会让反解出来的键落进规则函数不认识的洞里。
    """
    loc = locales.load(lang)
    assert set(loc["file"]) == set(sp.FILES)
    assert set(loc["change"]) == set(sp.CHANGES)
    assert set(loc["size"]) == set(sp.SIZES)
    assert set(loc["option"]["gate"]) == set(sp.GATES)
    assert len(loc["fam"]["severity"]["levels"]) == 3


@pytest.mark.parametrize("lang", LANGS)
def test_body_templates_hit_exactly_their_rule(lang: str) -> None:
    """输入：每种语言的每一条正文模板。输出：无。

    预期行为：**每条模板真的命中它标注的那一条规则、且只命中那一条**（干净槽一条都不命中）
    ——这是「flag 那一族答案对不对」的前提：题面里的判据是文本匹配，构造侧用的是键，
    两者不一致时产出的答案就是错的。
    """
    loc = locales.load(lang)
    used = _used_body_keys()
    assert set(loc["body"]) == set(used)
    for key, rule in used.items():
        size = "big" if rule == sp.RULE_BULK else "small"
        assert sp.rule_from(size, str(loc["body"][key])) == rule, (lang, key)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_every_table(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：三张表用到的标签与三条规则的触发词都在判据里印出来——考生只能从题面读规则，
    漏印一项就等于把那一档的答案变成不可解。
    """
    loc = locales.load(lang)
    policy = loc["policy"]["text"]
    for table in ("file", "change"):
        for text in loc[table].values():
            assert text in policy, (lang, table, text)
    assert loc["size"]["big"] in policy
    assert "password" in policy and "print(" in policy


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_every_gate_word(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：门控三个选项的描述开头那个词都出现在判据里——选项名与判据里的措辞对不上时，
    考生只能靠猜。
    """
    loc = locales.load(lang)
    policy = loc["policy"]["text"]
    for text in loc["option"]["gate"].values():
        head = str(text).split("：")[0].split(":")[0].strip()
        assert head and head in policy, (lang, head)


@pytest.mark.parametrize("lang", LANGS)
def test_flag_criteria_are_distinct_and_nonempty(lang: str) -> None:
    """输入：每种语言的 flag 判据。输出：无。预期行为：真 / 假两行都不空且不同。"""
    criteria = locales.load(lang)["fam"]["flag"]["criteria"]
    assert set(criteria) == {"true", "false"}
    assert all(criteria.values())
    assert criteria["true"] != criteria["false"]


@pytest.mark.parametrize("lang", LANGS)
def test_cell_separator_is_shared_across_languages(lang: str) -> None:
    """输入：每种语言。输出：无。预期行为：抬头分隔符一致（合并表只按一个分隔符切行）。"""
    assert locales.load(lang)["meta"]["cell_sep"] == locales.load("zh")["meta"]["cell_sep"]


def test_merged_table_covers_every_key() -> None:
    """输入：合并表。输出：无。

    预期行为：三张词表把 spec 里的每一个键都覆盖到，而且语言之间不重叠（重叠会直接
    报错，这里验的是「合并之后仍然认得出所有键」）。
    """
    merged = parse.merged()
    assert set(merged.file.values()) == set(sp.FILES)
    assert set(merged.change.values()) == set(sp.CHANGES)
    assert set(merged.size.values()) == set(sp.SIZES)
