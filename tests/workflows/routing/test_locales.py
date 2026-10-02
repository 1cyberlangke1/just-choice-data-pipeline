"""工单分诊的文案测试：选项名与特征表对齐、三张表顺序、禁用语不误伤。"""

from __future__ import annotations

import pytest

from workflows.routing import features, locales, render

from tests.workflows.routing.conftest import body_for


def test_criteria_keys_match_the_feature_tables() -> None:
    """输入：两语言文案。输出：无。

    预期行为：每个 choice / noul 题族的选项名必须与特征层那一维的取值完全一致——
    对不上就会出现「选项里没有的答案」或「答不出选项里的答案」。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["fam"]["issue"]["criteria"]) == list(features.ISSUES)
        assert list(loc["fam"]["urgency"]["criteria"]) == list(features.URGENCIES)
        assert list(loc["fam"]["rule"]["criteria"]) == list(features.QUEUE_RULES)
        assert list(loc["fam"]["queue"]["criteria"]) == list(features.QUEUES)
        assert list(loc["fam"]["needs_human"]["criteria"]) == ["true", "false"]
        assert len(loc["fam"]["priority"]["levels"]) == features.PRIORITY_LEVELS


def test_rule_table_order_matches_the_code() -> None:
    """输入：两语言派单表。输出：无。

    预期行为：locale 里 `rules` 的书写顺序必须与 `features.QUEUE_RULES` 一致——
    派单表是「从上往下第一条命中」，顺序变了语义就变了。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["rules"]) == list(features.QUEUE_RULES)
        for name in features.QUEUE_RULES:
            assert str(loc["rules"][name]["condition"]).strip()


def test_prompt_only_covers_the_text_dimensions() -> None:
    """输入：两语言 `dims` 表。输出：无。

    预期行为：`dims` 里只有由正文承载的两维——多一维就会让提示词把「工单属性」那一行
    管的事也写进正文，两边一冲突这道题就有歧义了（范式 §9.18）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["dims"]) == list(render.DIM_ORDER)


def test_attribute_vocabulary_covers_every_value() -> None:
    """输入：两语言属性词表。输出：无。

    预期行为：产品域 / 客户等级 / 合规标记三张词表覆盖各自取值；关键域清单由代码填进去。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["area"]) == list(features.AREAS)
        assert list(loc["tier"]) == list(features.TIERS)
        assert list(loc["compliance"]) == list(features.COMPLIANCES)
        rows = dict(render.flag_rows(loc))
        assert rows["critical_areas"].strip()
        for name in features.CRITICAL_AREAS:
            assert str(loc["area"][name]) in rows["critical_areas"]


def test_feature_line_raises_on_a_missing_phrase() -> None:
    """输入：一个没写进 locale 的取值。输出：无。预期行为：抛 KeyError，不静默给空串。"""
    loc = locales.load("zh")
    with pytest.raises(KeyError):
        locales.feature_line(loc, "issue", "bogus")


def test_banned_phrases_do_not_fire_on_the_fixture_bodies() -> None:
    """输入：夹具里那几段正文。输出：无。

    预期行为：一条都不许撞禁用语——撞了就说明禁用语写得过宽，真跑时会把本来能用的
    数据在渲染期白拒掉。
    """
    cases = features.build_cases(2048)
    for lang in locales.available():
        loc = locales.load(lang)
        for tags in cases:
            body = body_for(tags.as_dict())
            assert render.banned_echo(body, loc) == [], body


def test_issue_labels_are_declared_and_do_not_fire_on_the_fixture_bodies() -> None:
    """输入：两语言的档名名单与夹具正文。输出：无。

    预期行为：`meta.issue_labels` 非空、每条都不是空白；夹具那几段正文一条都不许撞——
    撞了就说明名单收得太宽，真跑时会把本来能用的数据在渲染期白拒掉。
    """
    cases = features.build_cases(2048)
    for lang in locales.available():
        loc = locales.load(lang)
        labels = loc["meta"]["issue_labels"]
        assert isinstance(labels, list) and labels
        assert all(str(item).strip() for item in labels)
        for tags in cases:
            body = body_for(tags.as_dict())
            assert render.label_leak(body, loc) == "", (lang, body)
