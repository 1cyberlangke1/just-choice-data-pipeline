"""文案表的不变量：这些不是「文案好不好」，而是「答案对不对」的前提。"""

from __future__ import annotations

import pytest

from workflows.elementpick import locales, parse, questions, render
from workflows.elementpick import spec as sp

LANGS = locales.available()
#: 判据里必须印出来的三张词表（标签那六段文本印在清单里，不印在判据里）。
VOCAB_TABLES = ("role", "visibility", "clickable")


@pytest.mark.parametrize("lang", LANGS)
def test_locales_share_the_same_shape(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：语言文件的**键集合完全一致**——少一格会让那种语言直接跑不动，少一道问法
    会让 `pick` 回绕到别的句面上（答案照样发出去，只是题变得莫名其妙）。
    """
    base = locales.load("zh")
    other = locales.load(lang)
    assert set(other["meta"]) == set(base["meta"])
    for table in ("role", "visibility", "clickable", "label"):
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
    assert set(loc["role"]) == set(sp.ROLES)
    assert set(loc["visibility"]) == set(sp.VISIBILITIES)
    assert set(loc["clickable"]) == set(sp.CLICKABILITIES)
    assert set(loc["label"]) == set(sp.LABELS)
    assert set(loc["option"]["role_class"]) == set(sp.ROLE_CLASSES)
    assert set(loc["fam"]) == set(questions.FAMILIES)
    assert len(loc["fam"]["decoy"]["levels"]) == len(sp.EXTRA_COUNTS)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_every_table(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：六个角色、两档可见性、两档可点性都在判据里印出来——考生只能从题面读规则，
    漏印一项就等于把那一族的答案变成不可解。
    """
    loc = locales.load(lang)
    policy = str(loc["policy"]["text"])
    for table in VOCAB_TABLES:
        for text in loc[table].values():
            assert str(text) in policy, (lang, table, text)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_every_role_class_word(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：角色三分类的选项描述开头那个词都出现在判据里——选项名与判据里的措辞对不上
    时，考生只能靠猜。
    """
    loc = locales.load(lang)
    policy = str(loc["policy"]["text"])
    for text in loc["option"]["role_class"].values():
        head = str(text).split("：")[0].split(":")[0].strip()
        assert head and head in policy, (lang, head)


@pytest.mark.parametrize("lang", LANGS)
def test_noul_criteria_are_distinct_and_nonempty(lang: str) -> None:
    """输入：每种语言的两个 noul 题族判据。输出：无。预期行为：真 / 假两行都不空且不同。"""
    fam = locales.load(lang)["fam"]
    for family in questions.NOUL_FAMILIES:
        criteria = fam[family]["criteria"]
        assert set(criteria) == {"true", "false"}
        assert all(criteria.values())
        assert criteria["true"] != criteria["false"]


@pytest.mark.parametrize("lang", LANGS)
def test_cell_separator_is_shared_across_languages(lang: str) -> None:
    """输入：每种语言。输出：无。预期行为：四格分隔符一致（合并表只按一个分隔符切行）。"""
    assert locales.load(lang)["meta"]["cell_sep"] == locales.load("zh")["meta"]["cell_sep"]


@pytest.mark.parametrize("lang", LANGS)
def test_shadow_suffix_is_not_part_of_any_label(lang: str) -> None:
    """输入：每种语言的后缀与标签。输出：无。

    预期行为：没有哪个标签原文**本身**就以后缀结尾——否则「先查完全一致」这一步会被
    剥后缀那一步抢走，反解出来的键会从标签变成近似键。
    """
    loc = locales.load(lang)
    suffix = str(loc["meta"]["shadow_suffix"])
    assert suffix
    for text in loc["label"].values():
        assert not str(text).endswith(suffix), (lang, text)


@pytest.mark.parametrize("lang", LANGS)
def test_near_texts_round_trip_through_the_single_language_table(lang: str) -> None:
    """输入：每种语言的六段近似文本。输出：无。

    预期行为：单语言表（不是合并表）也能把近似文本剥回近似键——渲染与反解是镜像。
    """
    loc = locales.load(lang)
    labels = parse.labels_of(loc)
    for label in sp.LABELS:
        raw = render.text_of(f"{sp.SHADOW_PREFIX}{label}", loc)
        assert parse.decode_text(raw, labels) == f"{sp.SHADOW_PREFIX}{label}"


def test_merged_table_covers_every_key() -> None:
    """输入：合并表。输出：无。

    预期行为：四张词表把 spec 里的每一个键都覆盖到，而且语言之间不重叠（重叠会直接
    报错，这里验的是「合并之后仍然认得出所有键」）。
    """
    merged = parse.merged()
    assert set(merged.role.values()) == set(sp.ROLES)
    assert set(merged.visibility.values()) == set(sp.VISIBILITIES)
    assert set(merged.clickable.values()) == set(sp.CLICKABILITIES)
    assert set(merged.label.values()) == set(sp.LABELS)
