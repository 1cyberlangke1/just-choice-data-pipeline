"""题族：四道题、题型、答案、判据写进题面、选项表与等级表。"""

from __future__ import annotations

import random

import pytest

from tests.workflows.quoteread.conftest import _make_spec
from workflows.quoteread import features as ft
from workflows.quoteread import locales, questions
from workflows.quoteread import spec as sp

LOC = locales.load("zh")


def test_answers_match_the_spec_for_every_case() -> None:
    """输入：全空间 case。输出：无。

    预期行为：四道题的答案都与真值规格算出来的一致——choice 是选项名（组装器再换成编号）、
    noul 是布尔、score 是档位下标。这里是「生成器把答案算错了」那条路的第一道闸。
    """
    for case in ft.build_cases(512):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        answers = {
            item.family: item.answer for item in questions.build(spec, LOC, 0, random.Random(0))
        }
        assert answers["tier"] == spec.tier, case.describe()
        assert answers["leader"] == spec.leader, case.describe()
        assert answers["alert"] is spec.alert, case.describe()
        assert answers["divergent"] is spec.divergent, case.describe()


def test_leader_options_are_this_case_three_instruments() -> None:
    """输入：同分布与留出侧各一条规格。输出：无。

    预期行为：选项键就是这一条 case 的三个名键（同分布侧是 `I*`、留出侧是 `A*`），描述互不
    相同——两个选项看起来一样的话考生没法分辨，是构造 bug。
    """
    id_options = questions.options_of("leader", _make_spec(3, names=0), LOC)
    assert set(id_options) == set(ft.NAME_IDS)
    assert len(set(id_options.values())) == ft.ROWS
    ood_options = questions.options_of("leader", _make_spec(4, names=1), LOC)
    assert set(ood_options) == set(ft.OOD_NAME_IDS)
    assert not set(id_options.values()) & set(ood_options.values())


def test_score_criteria_is_the_five_level_table() -> None:
    """输入：一条规格。输出：无。

    预期行为：`criteria` 是五档描述（原序，不打乱），答案就是这个数组里的下标——
    下标与 `features.TIER_FLOORS` 的档数一一对应。
    """
    spec = _make_spec(6, values=4)
    item = next(
        item for item in questions.build(spec, LOC, 0, random.Random(0)) if item.family == "tier"
    )
    assert item.question["criteria"] == list(LOC["fam"]["tier"]["levels"])
    assert len(item.question["criteria"]) == len(ft.TIER_FLOORS)
    assert item.answer == spec.tier
    assert isinstance(item.answer, int) and 0 <= item.answer < len(item.question["criteria"])


def test_rules_state_the_conversion_the_tiers_the_alert_and_the_tie_break() -> None:
    """输入：一份语言文件。输出：无。

    预期行为：判据原文里同时出现「换算」（1% = 100 基准点）、「档位表」、「警戒线」与
    「并列」四条——少写一条，读者就没法自己把四个答案算出来。
    """
    for lang in ("zh", "en"):
        rules = questions.rules_block(locales.load(lang))
        lowered = rules.lower()
        assert "100" in rules, lang
        assert "基准点" in rules or "basis point" in lowered, lang
        assert "档" in rules or "tier" in lowered, lang
        assert "警戒线" in rules or "alert line" in lowered, lang
        assert "并列" in rules or "tie" in lowered, lang
