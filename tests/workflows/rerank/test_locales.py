"""语言文件：键齐全、标签不撞车、模板占位符、词表两两不互为子串。"""

from __future__ import annotations

import pytest

from workflows.rerank import locales, parse, questions
from workflows.rerank import spec as sp

LANGS = ("zh", "en")


@pytest.mark.parametrize("lang", LANGS)
def test_every_required_key_is_present(lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：词表、句模板、判据、题面全齐。"""
    loc = locales.load(lang)
    assert set(loc["topic"]) == set(sp.TOPICS)
    assert set(loc["aspect"]) == set(sp.ASPECTS)
    assert set(loc["source"]) == set(sp.SOURCES)
    assert set(loc["sentence"]["body"]) == set(sp.WORDINGS)
    assert set(loc["sentence"]["example"]) == set(sp.WORDINGS)
    assert set(loc["sentence"]["trap"]) == set(sp.TRAPS_KEYS)
    for wording in sp.WORDINGS:
        assert set(loc["sentence"]["example"][wording]) == {"yes", "no"}, wording
    assert set(loc["fam"]) == set(questions.FAMILIES)
    assert set(loc["rules"]) == {
        "conditions_head",
        "condition_topic",
        "condition_aspect",
        "condition_fresh",
        "condition_example",
        "order_head",
        "order_relevance",
        "order_source",
        "order_day",
        "order_index",
    }
    assert {"title", "colon", "sep", "query_label", "cand_label", "source_label", "day_label"} <= set(
        loc["meta"]
    )
    assert "{cid}" in loc["option"]["cand"]


@pytest.mark.parametrize("lang", LANGS)
def test_body_templates_have_exactly_one_of_each_placeholder(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个正文模板恰好含 `{topic}` 与 `{aspect}` 各一次——判别器是从模板生成
    正则来抓这两个词的，少一个占位符就抓不回来。
    """
    loc = locales.load(lang)
    for wording in sp.WORDINGS:
        template = loc["sentence"]["body"][wording]
        assert template.count("{topic}") == 1, template
        assert template.count("{aspect}") == 1, template
    assert loc["query"]["text"].count("{topic}") == 1
    assert loc["query"]["text"].count("{aspect}") == 1


@pytest.mark.parametrize("lang", LANGS)
def test_labels_do_not_collide(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：标签两两之间没有前缀关系（否则反解会把一格认成另一格），而且抬头不以
    任何标签开头（否则第一行会被当成字段）。
    """
    loc = locales.load(lang)
    labels = list(parse.label_index(loc))
    for left in labels:
        for right in labels:
            if left != right:
                assert not right.startswith(left), (left, right)
    assert not loc["meta"]["title"].startswith(tuple(labels))


@pytest.mark.parametrize("lang", LANGS)
def test_words_are_distinct_and_not_substrings(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：主题词 / 方面词两两不互为子串——「这条讲的正是查询的主题吗」靠整词比对
    读出来；来源词两两不同——反解 `source` 表时不能撞车。
    """
    loc = locales.load(lang)
    for table in ("topic", "aspect", "source"):
        words = list(loc[table].values())
        assert len(set(words)) == len(words), table
        for left in words:
            for right in words:
                if left != right:
                    assert left not in right, (table, left, right)


@pytest.mark.parametrize("lang", LANGS)
def test_wordings_produce_distinct_text(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：同一 (主题, 方面) 下三种措辞的正文句互不相同，两种示例句也互不相同——
    措辞这一维要真的换说法，否则它只是「同一个模板重复三遍」。
    """
    loc = locales.load(lang)
    for topic in sp.TOPICS:
        for aspect in sp.ASPECTS:
            texts = {
                loc["sentence"]["body"][wording].format(
                    topic=loc["topic"][topic], aspect=loc["aspect"][aspect]
                )
                for wording in sp.WORDINGS
            }
            assert len(texts) == len(sp.WORDINGS), (topic, aspect)
    for kind in ("yes", "no"):
        texts = {loc["sentence"]["example"][wording][kind] for wording in sp.WORDINGS}
        assert len(texts) == len(sp.WORDINGS), kind


@pytest.mark.parametrize("lang", LANGS)
def test_trap_sentences_are_pairwise_distinct(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：噪声句两两不同——判别器把它们收成一个集合，重复会让「认不出的行」漏检。
    """
    words = list(locales.load(lang)["sentence"]["trap"].values())
    assert len(set(words)) == len(words)


@pytest.mark.parametrize("lang", LANGS)
def test_families_have_phrasings_and_valid_criteria(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个题族至少一条问法；noul 的判据恰好 true / false 两行；score 的档位
    五档；choice 的选项描述模板带着编号。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        assert loc["fam"][family]["q"], family
    for family in questions.NOUL_FAMILIES:
        assert set(loc["fam"][family]["criteria"]) == {"true", "false"}, family
    assert len(loc["fam"]["relevance"]["levels"]) == 5
    for family in questions.CHOICE_FAMILIES:
        assert "q" in loc["fam"][family]


def test_languages_do_not_share_labels() -> None:
    """输入：两份语言文件。输出：无。预期行为：标签原文互不重叠（`merged_labels` 才不炸）。"""
    zh = set(parse.label_index(locales.load("zh")))
    en = set(parse.label_index(locales.load("en")))
    assert not zh & en
    assert len(parse.merged_labels()) == len(zh) + len(en)
