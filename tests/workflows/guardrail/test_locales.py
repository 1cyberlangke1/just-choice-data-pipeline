"""语言文件的硬约束：标签不打架、句子模板互不包含、题面与选项齐全。"""

from __future__ import annotations

import pytest

from workflows.guardrail import judges, locales, parse, questions
from workflows.guardrail import spec as sp

LANGS = ("zh", "en")


@pytest.mark.parametrize("lang", LANGS)
def test_every_required_key_is_present(lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：字段值、条款名、句模板、题面全齐。"""
    loc = locales.load(lang)
    for dim, values in (
        ("reach", sp.REACHES),
        ("direction", sp.DIRECTIONS),
        ("audience", sp.AUDIENCES),
        ("history", sp.HISTORIES),
        ("clause", sp.CLAUSES),
    ):
        assert set(loc[dim]) == set(values), dim
    assert len(loc["sentence"]["filler"]) == sp.FILLER_SLOTS
    assert set(loc["sentence"]["violation"]) == set(sp.CLAUSES)
    assert set(loc["sentence"]["mention"]) == set(sp.CLAUSES)
    assert set(loc["sentence"]["wrapper"]) == set(sp.WORDINGS)
    assert set(loc["fam"]) == {
        "action",
        "top_clause",
        "violates",
        "minor_exposure",
        "repeat_offender",
        "severity",
    }
    assert set(loc["option"]["top_clause"]) == set(sp.CLAUSES) | {"none"}
    assert set(loc["option"]["action"]) == set(sp.ACTIONS)


@pytest.mark.parametrize("lang", LANGS)
def test_labels_do_not_collide(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：标签两两之间没有前缀关系（否则反解会把一格认成另一格），
    而且抬头不以任何标签开头（否则第一行会被当成字段）。
    """
    loc = locales.load(lang)
    labels = list(parse.label_index(loc))
    for left in labels:
        for right in labels:
            if left != right:
                assert not right.startswith(left), (left, right)
    assert not loc["meta"]["title"].startswith(tuple(labels))


@pytest.mark.parametrize("lang", LANGS)
def test_sentence_templates_are_pairwise_distinct(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`sentence_lookup` 不报重复——违规句（三种措辞）、擦边句、填充句彼此都不重合；
    重合了判别器认句就有歧义。
    """
    table = judges.sentence_lookup(locales.load(lang))
    assert len(table) == len(sp.CLAUSES) * (len(sp.WORDINGS) + 1) + sp.FILLER_SLOTS


@pytest.mark.parametrize("lang", LANGS)
def test_no_violation_sentence_is_contained_in_a_mention(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：任何一条违规句（含包裹）都不是任何擦边句的子串——判别器按整行精确匹配，
    但**人**读题时若擦边句里嵌着违规句，题面就自相矛盾了。
    """
    loc = locales.load(lang)
    violations = [
        loc["sentence"]["wrapper"][wording].format(s=loc["sentence"]["violation"][clause])
        for clause in sp.CLAUSES
        for wording in sp.WORDINGS
    ]
    for clause in sp.CLAUSES:
        mention = loc["sentence"]["mention"][clause]
        for violation in violations:
            assert violation not in mention, (clause, violation)


@pytest.mark.parametrize("lang", LANGS)
def test_wrappers_produce_distinct_text(lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：三种措辞下同一句文本互不相同。"""
    loc = locales.load(lang)
    for clause in sp.CLAUSES:
        raw = loc["sentence"]["violation"][clause]
        texts = {loc["sentence"]["wrapper"][wording].format(s=raw) for wording in sp.WORDINGS}
        assert len(texts) == len(sp.WORDINGS), clause


@pytest.mark.parametrize("lang", LANGS)
def test_families_have_phrasings_and_distinct_options(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个题族至少一条问法；choice 的选项描述两两不同；noul 的判据恰好
    true / false 两行；score 的档位五档。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        entry = loc["fam"]["violates" if family.startswith("violates_") else family]
        assert entry["q"], family
    for family in ("action", "top_clause"):
        options = list(loc["option"][family].values())
        assert len(set(options)) == len(options), family
    for family in ("violates", "minor_exposure", "repeat_offender"):
        assert set(loc["fam"][family]["criteria"]) == {"true", "false"}
    assert len(loc["fam"]["severity"]["levels"]) == 5


def test_languages_do_not_share_labels() -> None:
    """输入：两份语言文件。输出：无。预期行为：标签原文互不重叠（`merged_labels` 才不炸）。"""
    zh = set(parse.label_index(locales.load("zh")))
    en = set(parse.label_index(locales.load("en")))
    assert not zh & en
    assert len(parse.merged_labels()) == len(zh) + len(en)
