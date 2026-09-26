"""语言文件的契约：文本两两互不为子串、标签互不重叠、题面与选项齐全。

这套不变量不是洁癖——判别器**只看文本**：句子靠整句匹配反查回事实、主张靠子串匹配认出来、
引文靠「是不是某个整句」定在不在文档里。任何一条包含关系都会同时污染这三条判据。
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from workflows.citation import locales, parse, questions
from workflows.citation import spec as sp

#: 一句原子陈述在语言文件里的四种写法。
TEXT_KEYS: tuple[str, ...] = ("doc", "neg", "claim", "claim_neg")


def _all_texts() -> list[tuple[str, str, str, str, str]]:
    """输入：无。输出：[(语言, 主题, 句号, 写法, 文本)]——两种语言、四个主题、全部句子。"""
    out: list[tuple[str, str, str, str, str]] = []
    for lang in locales.available():
        loc = locales.load(lang)
        for topic in sp.TOPIC_NAMES:
            entry = loc["topic"][topic]
            for sentence in sp.SENTENCE_IDS[topic]:
                item = entry["sentence"][sentence]
                for key in TEXT_KEYS:
                    out.append((lang, topic, sentence, key, item[key]))
    return out


def test_labels_are_disjoint_across_languages() -> None:
    """输入：全部语言。输出：无。

    预期行为：合并表能覆盖每种语言的全部标签，且同一个标签文本不出现在两种语言里——
    重叠会让语言无关的反解把两种语言的字段串成一行（`merged_labels` 会当场报错）。
    """
    merged = parse.merged_labels()
    for lang in locales.available():
        for text, slot in parse.label_index(locales.load(lang)).items():
            assert merged[text] == slot
    assert len(merged) == len(parse.META_KEYS) * len(locales.available())


def test_merged_labels_rejects_two_languages_sharing_a_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """输入：两种语言的标签表里出现同一个标签文本。输出：异常。

    预期行为：当场报错——语言无关的反解靠「这个标签文本属于哪个槽」认字段，
    重叠会让两种语言的字段串成一行，读出来的值是错的。
    """
    shared = {"meta": {key: "同一个标签" for key in parse.META_KEYS}}
    monkeypatch.setattr(locales, "available", lambda: ["zh", "en"])
    monkeypatch.setattr(locales, "load", lambda lang: shared)
    with pytest.raises(ValueError, match="重复"):
        parse.merged_labels()


def test_no_text_is_a_substring_of_another() -> None:
    """输入：两种语言的全部句子与转述。输出：无。

    预期行为：192 条文本**两两互不为子串**。这条撑起三件事：
    - 句子表不会把一个文本映射到两个事实（`_put` 会炸）；
    - 主张里的转述只认得出**一条**（`claim_fact` 会炸）；
    - 截断出来的前半句不可能等于文档里的任何整句——于是「编造」这一档站得住。
    """
    texts = _all_texts()
    assert len(texts) == len(sp.TOPIC_NAMES) * 6 * len(TEXT_KEYS) * len(locales.available())
    for lang, topic, sentence, key, text in texts:
        assert text.strip() == text and text, (lang, topic, sentence, key)
        for other_lang, other_topic, other_sentence, other_key, other in texts:
            if (lang, topic, sentence, key) == (other_lang, other_topic, other_sentence, other_key):
                continue
            assert text not in other, f"{text!r}（{lang}/{topic}/{sentence}/{key}）出现在 {other!r} 里"


def test_the_four_forms_of_one_sentence_are_pairwise_distinct() -> None:
    """输入：每一句话的四种写法。输出：无。

    预期行为：原文、反面、两种转述四个文本互不相同——`claim` 与同句的 `doc` 要是撞了，
    「引文是否支持主张」就退化成字符串匹配了。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for topic in sp.TOPIC_NAMES:
            for sentence in sp.SENTENCE_IDS[topic]:
                item = loc["topic"][topic]["sentence"][sentence]
                values = [item[key] for key in TEXT_KEYS]
                assert len(set(values)) == len(TEXT_KEYS), (lang, topic, sentence)


def test_sentence_lines_never_look_like_a_section_head() -> None:
    """输入：两种语言的句子与转述。输出：无。

    预期行为：句子行不以数字开头。反解靠 `^\\d+\\.` 认小节行，句子要是以「1. 」开头，
    它就会被吃成小节行，整页结构全乱。
    """
    for lang, topic, sentence, key, text in _all_texts():
        assert not re.match(r"^\d", text), (lang, topic, sentence, key, text)
        assert not text.startswith("["), (lang, topic, sentence, key, text)


def test_headings_are_complete_and_distinct() -> None:
    """输入：每种语言的主题。输出：无。预期行为：每主题 6 个标题、互不重复。"""
    for lang in locales.available():
        loc = locales.load(lang)
        for topic in sp.TOPIC_NAMES:
            headings = loc["topic"][topic]["headings"]
            assert len(headings) == sp.HEADING_COUNT, (lang, topic)
            assert len(set(headings)) == len(headings), (lang, topic)


def test_topic_labels_are_distinct_and_never_a_prefix_of_each_other() -> None:
    """输入：每种语言的主题名。输出：无。

    预期行为：主题名互不为前缀——产物审计靠「来源文档那一行以哪个主题名开头」认主题，
    前缀关系会让它把两个主题算成一个。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        labels = [loc["topic"][name]["label"] for name in sp.TOPIC_NAMES]
        assert len(set(labels)) == len(labels), lang
        for one in labels:
            for other in labels:
                if one != other:
                    assert not other.startswith(one), (lang, one, other)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_meta_fields_are_present(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：抬头、四个标签、小节词、冒号、文档后缀模板、两种主张框架都在；
    后缀模板必须带 `{n}`，框架必须带 `{claim}`——少一个占位符就会渲染出写死的数字。
    """
    meta = locales.load(lang)["meta"]
    for key in ("title", "doc_label", "claim_label", "quote_label", "source_label", "section_word", "colon"):
        assert meta[key], key
    assert "{n}" in meta["doc_suffix"]
    assert len(meta["claim_frames"]) >= 2
    for frame in meta["claim_frames"]:
        assert "{claim}" in frame


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_every_question_family_has_phrasings_and_the_right_criteria(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：七个题族都在、每族至少两种问法；noul 族有 true/false 描述；
    score 族的等级表宽度就是答案上限（证据强度 0~3，四档）。
    """
    loc = locales.load(lang)
    assert set(loc["ask"]) == set(questions.FAMILIES)
    for family in questions.FAMILIES:
        entry = loc["ask"][family]
        assert len(entry["q"]) >= 2, family
    for family in questions.NOUL_FAMILIES:
        assert loc["ask"][family]["criteria"]["true"] and loc["ask"][family]["criteria"]["false"]
    assert len(loc["ask"]["evidence_strength"]["levels"]) == 4


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_option_maps_cover_exactly_the_reachable_answers(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四个 choice 题族的选项名**恰好**等于规则引擎能算出来的答案集合——
    少一个会让某条数据组装不出来，多一个则是永远选不到的死选项。
    """
    loc = locales.load(lang)
    reachable: dict[str, set[str]] = {"verdict": set(), "relation": set(), "quote_source": set(), "action": set()}
    for found in (False, True):
        for located in (False, True):
            for relation in ("supports", "contradicts", "says_nothing"):
                answers = sp.answers_of(found, located, relation)
                reachable["verdict"].add(answers["verdict"])
                reachable["relation"].add(answers["relation"])
                reachable["quote_source"].add(answers["quote_source"])
                reachable["action"].add(answers["action"])
    for family, wanted in reachable.items():
        assert set(loc["option"][family]) == wanted, (family, set(loc["option"][family]) ^ wanted)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_choice_option_descriptions_are_distinct(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：同一题族里没有两个选项描述一模一样——描述重复意味着两个选项都说得通，
    是构造 bug（组装器会当场报错）。
    """
    loc = locales.load(lang)
    for family in questions.CHOICE_FAMILIES:
        options: dict[str, Any] = loc["option"][family]
        assert len(set(options.values())) == len(options), family


def test_unknown_locale_lists_the_available_ones() -> None:
    """输入：不存在的语言。输出：异常。预期行为：报错列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="zh"):
        locales.load("nope")
