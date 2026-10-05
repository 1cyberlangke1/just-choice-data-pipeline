"""题族：四道题、题型、答案、判据写进题面、选项表与等级表。"""

from __future__ import annotations

import random

import pytest

from tests.workflows.clipcut.conftest import _make_spec
from workflows.clipcut import features as ft
from workflows.clipcut import locales, questions
from workflows.clipcut import spec as sp

LOC = locales.load("zh")


def test_answers_match_the_spec_for_every_case() -> None:
    """输入：全空间 case。输出：无。

    预期行为：四道题的答案都与真值规格算出来的一致——choice 是选项名（组装器再换成编号）、
    noul 是布尔、score 是等级下标。这里是「生成器把答案算错了」那条路的第一道闸。
    """
    for case in ft.build_cases(512):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        answers = {
            item.family: item.answer for item in questions.build(spec, LOC, 0, random.Random(0))
        }
        assert answers["keep_count"] == spec.keep_count, case.describe()
        assert answers["first_kept"] == spec.first_kept, case.describe()
        assert answers["blocked_any"] is spec.blocked_any, case.describe()
        assert answers["dup_any"] is spec.dup_any, case.describe()


def test_choice_answers_are_option_names_inside_the_criteria() -> None:
    """输入：全空间 case。输出：无。

    预期行为：choice 的答案必须是 `criteria` 里的**键**——组装器靠 `list(criteria).index`
    取编号，答案不在选项里会在组装期直接报错（那是构造 bug，不能进产物）。
    """
    for case in ft.build_cases(512):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        for item in questions.build(spec, LOC, 0, random.Random(0)):
            if item.question["type"] != "choice":
                continue
            assert item.answer in item.question["criteria"], item.family


def test_first_kept_options_are_the_five_slots() -> None:
    """输入：一条规格。输出：无。

    预期行为：五个选项键是 `s1`~`s4` + `none`，描述互不相同——两个选项看起来一样的话
    考生没法分辨，是构造 bug。
    """
    spec = _make_spec(3, topics=2, flags=0)
    options = questions.options_of("first_kept", spec, LOC)
    assert set(options) == {"s1", "s2", "s3", "s4", "none"}
    assert len(set(options.values())) == 5


def test_score_criteria_is_the_five_level_table() -> None:
    """输入：一条规格。输出：无。

    预期行为：`criteria` 是五档描述（原序，不打乱），答案就是这个数组里的下标——
    下标与 `features.SEGMENTS` 的段数一一对应。
    """
    spec = _make_spec(5, topics=0, flags=0)
    item = next(
        item for item in questions.build(spec, LOC, 0, random.Random(0))
        if item.family == "keep_count"
    )
    assert item.question["criteria"] == list(LOC["fam"]["keep_count"]["levels"])
    assert len(item.question["criteria"]) == ft.SEGMENTS + 1
    assert item.answer == spec.keep_count
    assert 0 <= item.answer < len(item.question["criteria"])


def test_rules_state_both_retention_conditions() -> None:
    """输入：一份语言文件。输出：无。

    预期行为：判据原文里同时出现「含高亮点」与「第一次出现」两个条件、以及「重复」这条
    去冗余规则——少写一条，读者就没法自己把四个答案算出来。
    """
    for lang in ("zh", "en"):
        rules = questions.rules_block(locales.load(lang))
        assert "高亮" in rules or "highlight" in rules.lower(), lang
        assert "第一次" in rules or "first" in rules.lower(), lang
        assert "重复" in rules or "repeat" in rules.lower(), lang
