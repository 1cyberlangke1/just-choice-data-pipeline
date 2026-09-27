"""文案表的不变量：这些不是「文案好不好」，而是「答案对不对」的前提。"""

from __future__ import annotations

import pytest

from workflows.modelroute import locales, parse
from workflows.modelroute.spec import (
    COMPLEXITIES,
    COSTS,
    DATAS,
    LATENCIES,
    RETRIES,
    STEPS,
    TASKS,
    TIERS,
)

LANGS = locales.available()


def test_available_locales_are_sorted_and_nonempty() -> None:
    """输入：无。输出：无。预期行为：至少两种语言，顺序稳定。"""
    assert LANGS == sorted(LANGS)
    assert {"zh", "en"} <= set(LANGS)


def test_unknown_locale_reports_available_ones() -> None:
    """输入：不存在的语言。输出：无。预期行为：报错并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="没有语言文件"):
        locales.load("xx")


def test_pick_wraps_around() -> None:
    """输入：长度为 2 的池、编号 3。输出：无。预期行为：回绕到第 1 个（不报错）。"""
    assert locales.pick(["a", "b"], 3) == "b"
    assert locales.pick(["a", "b"], 4) == "a"


@pytest.mark.parametrize("lang", LANGS)
def test_locales_share_the_same_shape(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：语言文件的**键集合完全一致**——少一个任务的正文模板会让那种语言直接跑不动，
    少一格题面会让 `pick` 回绕到别的问法上（答案照样发出去，只是题变得莫名其妙）。
    """
    base = locales.load("zh")
    other = locales.load(lang)
    assert set(other["meta"]) == set(base["meta"])
    for table in ("complexity", "latency", "cost", "retry", "data"):
        assert set(other[table]) == set(base[table])
    assert set(other["body"]) == set(base["body"]) == set(TASKS)
    assert set(other["task_phrase"]) == set(base["task_phrase"]) == set(TASKS)
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

    预期行为：每个词表的键就是 spec 里的常量——少一个取值会让渲染 `KeyError`，
    多一个取值会让 `reverse` 出来的键落进规则函数不认识的洞里。
    """
    loc = locales.load(lang)
    assert set(loc["complexity"]) == set(COMPLEXITIES)
    assert set(loc["latency"]) == set(LATENCIES)
    assert set(loc["cost"]) == set(COSTS)
    assert set(loc["retry"]) == set(RETRIES)
    assert set(loc["data"]) == set(DATAS)
    assert set(loc["option"]["tier"]) == set(TIERS)
    assert set(loc["option"]["next"]) == set(TIERS) - {TIERS[0]}
    assert set(loc["option"]["effort"]) == set(STEPS)


@pytest.mark.parametrize("lang", LANGS)
def test_each_task_phrase_lives_in_exactly_one_body(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：每个任务短语**恰好**出现在它自己那句正文里——判别器靠它把任务认回来，
    串味会让 `task_of` 报「认不出唯一的任务类型」，整个 workflow 一条样本都出不来。
    """
    loc = locales.load(lang)
    for task in TASKS:
        phrase = loc["task_phrase"][task]
        holders = [key for key in TASKS if phrase in loc["body"][key]]
        assert holders == [task], f"{lang}/{task}: {holders}"


@pytest.mark.parametrize("lang", LANGS)
def test_policy_spells_out_every_tier_and_task(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：判据原文里点名了每个档位与每个任务短语——判据写在题面里（范式 §9.8），
    少写一个档，考生就只能靠猜。
    """
    loc = locales.load(lang)
    policy = loc["policy"]["text"]
    for tier in TIERS:
        assert tier in policy, f"{lang}: 判据里没有档位 {tier}"
    for task in TASKS:
        assert loc["task_phrase"][task] in policy, f"{lang}: 判据里没有任务 {task}"


@pytest.mark.parametrize("lang", LANGS)
def test_labels_are_distinct_within_a_locale(lang: str) -> None:
    """输入：每种语言。输出：无。预期行为：同一语言里没有两格标签文案相同。"""
    index = parse.label_index(locales.load(lang))
    assert len(index) == len(parse.META_KEYS)


def test_merged_labels_covers_both_locales_without_collisions() -> None:
    """输入：无。输出：无。

    预期行为：各语言的标签互不重叠——`reparse` 没有语言参数，重叠了只能报错。
    """
    merged = parse.merged_labels()
    assert len(merged) == len(parse.META_KEYS) * len(LANGS)


@pytest.mark.parametrize("lang", LANGS)
def test_option_descriptions_are_distinct(lang: str) -> None:
    """输入：每种语言的选项表。输出：无。预期行为：同一张表里没有两个一样的描述。"""
    loc = locales.load(lang)
    for table in loc["option"].values():
        assert len(set(table.values())) == len(table)


@pytest.mark.parametrize("lang", LANGS)
def test_noul_criteria_are_distinct(lang: str) -> None:
    """输入：每种语言的 noul 判据。输出：无。预期行为：true / false 两行文案不一样。"""
    loc = locales.load(lang)
    for family in ("is_first", "degrade", "cache"):
        criteria = loc["fam"][family]["criteria"]
        assert set(criteria) == {"true", "false"}
        assert criteria["true"] != criteria["false"]
