"""语言文件的契约：标签互不重叠、主题词互不包含、模板齐全、题面编号干净。"""

from __future__ import annotations

import re

import pytest

from workflows.search import locales, parse, questions
from workflows.search import spec as sp


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


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_topic_words_do_not_contain_each_other(lang: str) -> None:
    """输入：一种语言的主题词表。输出：无。

    预期行为：主题名之间、要点词之间、主题名与要点词之间都**互不包含**——
    判别器靠「问题里出现了哪个主题名」认主题、靠「摘要里出现了几个要点词」算覆盖度，
    任何包含关系都会同时污染这两条判据（跑题项的摘要会假装覆盖了要点）。
    """
    loc = locales.load(lang)
    labels = [loc["topic"][name]["label"] for name in sp.TOPIC_NAMES]
    points = [point for name in sp.TOPIC_NAMES for point in loc["topic"][name]["points"]]
    for group in (labels, points, [*labels, *points]):
        for one in group:
            for other in group:
                if one is other or one == other:
                    continue
                assert one not in other, f"{one!r} 出现在 {other!r} 里"


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_every_template_and_key_exists(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每个角色的标题与摘要模板都在；每个文档类型、来源权威、动作、主题都有文案。
    少一条就会在生成期抛 KeyError，与其等它炸，不如在这里点名。
    """
    loc = locales.load(lang)
    for flavor in sp.FLAVORS:
        assert flavor.snippet in loc["snippet"], flavor.name
        assert flavor.name in loc["title"], flavor.name
    for _snippet_a, _snippet_b, title_a, title_b, _, _ in sp.PAIR_PLANS.values():
        assert title_a in loc["title"] and title_b in loc["title"]
    for name in sp.TOPIC_NAMES:
        assert name in loc["topic"]
    for kind in sp.KINDS:
        assert kind in loc["doc_kind"]
    for authority in sp.AUTHORITIES:
        assert authority in loc["authority"]
    for action in ("answer", "search_more", "switch_source"):
        assert action in loc["option"]["next_action"]


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_only_the_conflict_snippets_mention_the_metric(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：整页里**只有** `conflict_a` / `conflict_b` 提到指标名——于是「两条是否矛盾」
    这件事在页面上只有一个来源，判别器的 `conflict_of` 才站得住。
    """
    loc = locales.load(lang)
    for name in sp.TOPIC_NAMES:
        entry = loc["topic"][name]
        fields = {
            "label": entry["label"],
            "p1": entry["points"][0],
            "p2": entry["points"][1],
            "metric": entry["metric"],
            "v1": entry["values"][0],
            "v2": entry["values"][1],
            "other": entry["label"],
            "op1": entry["points"][0],
            "op2": entry["points"][1],
        }
        for key, block in loc["snippet"].items():
            text = block["text"].format(**fields)
            mentions = entry["metric"] in text
            assert mentions == (key in ("conflict_a", "conflict_b")), (name, key, text)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_query_templates_carry_the_topic_and_both_points(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每一条问法里都嵌了主题名与两个要点词——判别器靠这两样认主题、算覆盖度，
    漏一个就会让某些样本反解不出来。
    """
    loc = locales.load(lang)
    for name in sp.TOPIC_NAMES:
        entry = loc["topic"][name]
        fields = {
            "label": entry["label"],
            "p1": entry["points"][0],
            "p2": entry["points"][1],
        }
        for template in loc["query"]["tpl"]:
            text = template.format(**fields)
            assert entry["label"] in text
            assert entry["points"][0] in text and entry["points"][1] in text


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_probe_and_pair_question_texts_carry_exactly_the_indices(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：问「结果 {n}」与「结果 {i} 与结果 {j}」的题面里，数字**只允许**是编号本身——
    判别器从题面读编号重算，多一个数字就会读错。
    """
    loc = locales.load(lang)
    for family in questions.PROBE_FAMILIES:
        for template in loc["fam"][family]["q"]:
            text = template.format(n=7)
            assert re.findall(r"\d+", text) == ["7"], (family, text)
    for family in questions.PAIR_FAMILIES:
        for template in loc["fam"][family]["q"]:
            text = template.format(i=7, j=8)
            assert re.findall(r"\d+", text) == ["7", "8"], (family, text)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_score_levels_match_the_scale_width(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：相关度 4 档（0~3）、要点覆盖 3 档（0~2）——等级表的长度就是答案的取值范围，
    短了会越界，长了会有永远取不到的等级。
    """
    loc = locales.load(lang)
    assert len(loc["fam"]["relevance"]["levels"]) == 4
    assert len(loc["fam"]["covered_points"]["levels"]) == 3


def test_unknown_locale_lists_the_available_ones() -> None:
    """输入：不存在的语言。输出：异常。预期行为：报错列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="zh"):
        locales.load("nope")
