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
