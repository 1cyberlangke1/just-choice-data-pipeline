"""题族：四道题、题型、答案、判据写进题面、选项表与等级表。"""

from __future__ import annotations

import random

import pytest

from tests.workflows.quoteread.conftest import _make_spec
from workflows.quoteread import features as ft
from workflows.quoteread import locales, questions
from workflows.quoteread import spec as sp

LOC = locales.load("zh")


def test_families_are_the_four_known_ones() -> None:
    """输入：无。输出：题族清单。

    预期行为：四个题族、顺序稳定（先档位、再标的、最后两个布尔），三张分类表正好把四族
    分完——多一族少一族都会让下面的「答案在选项里」漏查。
    """
    assert questions.FAMILIES == ("tier", "leader", "alert", "divergent")
    assert questions.SCORE_FAMILIES == ("tier",)
    assert questions.CHOICE_FAMILIES == ("leader",)
    assert questions.NOUL_FAMILIES == ("alert", "divergent")
    assert set(questions.FAMILIES) == (
        set(questions.CHOICE_FAMILIES)
        | set(questions.SCORE_FAMILIES)
        | set(questions.NOUL_FAMILIES)
    )


def test_build_returns_one_question_per_family() -> None:
    """输入：一条规格。输出：无。预期行为：四道题、族名一一对应、题型正确。"""
    built = questions.build(_make_spec(1), LOC, 0, random.Random(0))
    assert [item.family for item in built] == list(questions.FAMILIES)
    types = {item.family: item.question["type"] for item in built}
    assert types["tier"] == "score"
    assert types["leader"] == "choice"
    assert types["alert"] == "noul"
    assert types["divergent"] == "noul"


def test_only_choice_options_may_be_shuffled() -> None:
    """输入：一条规格。输出：无。

    预期行为：只有 choice 允许打乱——`tier` 的五档有自然顺序（波动从平盘往剧烈涨），
    打乱它会把「档位」读成「随便一个等级」。
    """
    for item in questions.build(_make_spec(2), LOC, 0, random.Random(0)):
        assert item.shuffle is (item.question["type"] == "choice"), item.family


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


def test_unknown_family_has_no_options() -> None:
    """输入：表外的题族。输出：无。预期行为：报人话，不返回空选项表。"""
    with pytest.raises(ValueError, match="未知题族"):
        questions.options_of("tier", _make_spec(5), LOC)


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


def test_noul_criteria_has_true_and_false() -> None:
    """输入：一条规格。输出：无。

    预期行为：两个 noul 族都带 `criteria`（官方支持描述，那我们就带上）——真 / 假两行都在，
    答案必须是布尔而不是字符串 `"true"`。
    """
    built = questions.build(_make_spec(7), LOC, 0, random.Random(0))
    for item in built:
        if item.question["type"] != "noul":
            continue
        assert set(item.question["criteria"]) == {"true", "false"}, item.family
        assert isinstance(item.answer, bool), item.family


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_no_question_text_keeps_a_placeholder(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：题面、选项描述、等级描述里都不许留下 `{...}` 占位符——留一个就是一道
    问不出来的题，而契约校验只管形状，管不到这个。
    """
    loc = locales.load(lang)
    for item in questions.build(_make_spec(8), loc, 0, random.Random(0)):
        texts = [item.question["instructions"]]
        criteria = item.question.get("criteria")
        if isinstance(criteria, dict):
            texts.extend(str(value) for value in criteria.values())
        elif isinstance(criteria, list):
            texts.extend(str(value) for value in criteria)
        for text in texts:
            assert "{" not in text and "}" not in text, f"{lang}/{item.family}: {text}"


def test_instructions_end_with_the_rules_block() -> None:
    """输入：一条规格。输出：无。

    预期行为：每道题的题面都以判据原文结尾——判据必须随题面一起给出去（范式 §9.8），
    藏在判别器里等于只读 state 的读者算不出来。
    """
    rules = questions.rules_block(LOC)
    assert rules
    for item in questions.build(_make_spec(9), LOC, 0, random.Random(0)):
        assert item.question["instructions"].endswith(rules), item.family


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


def test_build_is_a_pure_function() -> None:
    """输入：同一规格、两个不同 RNG。输出：无。预期行为：题目逐字段一致（不掷骰子）。"""
    spec = _make_spec(10)
    first = questions.build(spec, LOC, 0, random.Random(0))
    second = questions.build(spec, LOC, 0, random.Random(999))
    assert first == second


def test_phrasing_index_cycles_through_the_pool() -> None:
    """输入：两种问法编号。输出：无。

    预期行为：问法按编号取，取遍池子里的每一条——同一 case 的多条样本靠它换说法，
    两种编号给同一句话等于这一维白设。
    """
    pool = LOC["fam"]["tier"]["q"]
    seen = {
        questions.build(_make_spec(11), LOC, index, random.Random(0))[0].question["instructions"]
        for index in range(len(pool))
    }
    assert len(seen) == len(pool)
