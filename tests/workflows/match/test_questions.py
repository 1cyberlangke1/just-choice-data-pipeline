"""题族：八道题、题型、答案、判据写进题面、问法轮换。"""

from __future__ import annotations

import random

import pytest

from tests.workflows.match.conftest import _make_case, _make_spec
from workflows.match import locales, questions
from workflows.match import spec as sp

LOC = locales.load("zh")


def test_families_cover_two_choices_five_nouls_and_one_score() -> None:
    """输入：无。输出：题族清单。

    预期行为：八个题族、顺序稳定、题型分配正确，且五个 `same_<栏>` 正好覆盖五栏。
    """
    assert questions.FAMILIES == (
        "verdict",
        "variant_count",
        "same_name",
        "same_brand",
        "same_size",
        "same_abv",
        "same_pack",
        "band",
    )
    assert set(questions.NOUL_FIELDS.values()) == set(sp.FIELDS)
    assert set(questions.FAMILIES) == (
        set(questions.CHOICE_FAMILIES)
        | set(questions.NOUL_FIELDS)
        | set(questions.SCORE_FAMILIES)
    )


def test_variant_choices_come_from_the_pattern_table() -> None:
    """输入：十二种关系。输出：无。

    预期行为：`variant_count` 的选项就是关系表实际会产生的栏数（0~3），不手写第二份。
    """
    assert questions.VARIANT_CHOICES == (0, 1, 2, 3)


def test_build_returns_one_question_per_family() -> None:
    """输入：一条规格。输出：无。预期行为：八道题、族名一一对应、题型正确。"""
    built = questions.build(_make_spec(1), LOC, 0, random.Random(0))
    assert [item.family for item in built] == list(questions.FAMILIES)
    types = {item.family: item.question["type"] for item in built}
    for family in questions.CHOICE_FAMILIES:
        assert types[family] == "choice"
    for family in questions.NOUL_FIELDS:
        assert types[family] == "noul"
    assert types["band"] == "score"


def test_answers_match_the_spec() -> None:
    """输入：全部十二种关系。输出：无。

    预期行为：choice 的答案是选项名（组装器再换成编号）、noul 是布尔、score 是档位下标，
    且都与规格算出来的一致。
    """
    for pattern in sp.PATTERNS:
        spec = sp.sample_spec(None, _make_case(pattern=pattern), "train", 0, 0)
        answers = {
            item.family: item.answer
            for item in questions.build(spec, LOC, 0, random.Random(0))
        }
        assert answers["verdict"] == spec.verdict
        assert answers["variant_count"] == f"v{spec.variant_count}"
        assert answers["band"] == spec.band
        for family, field in questions.NOUL_FIELDS.items():
            assert answers[family] is spec.same[sp.FIELDS.index(field)]


def test_choice_options_are_complete_and_distinct() -> None:
    """输入：一条规格。输出：无。

    预期行为：`verdict` 三个选项、`variant_count` 四个选项，选项名固定、描述互不相同
    （描述重复的话考生没法分辨，构造期就报错）。
    """
    spec = _make_spec(2)
    verdict = questions.options_of("verdict", spec, LOC)
    assert set(verdict) == {"same", "different", "review"}
    assert len(set(verdict.values())) == 3
    variants = questions.options_of("variant_count", spec, LOC)
    assert set(variants) == {"v0", "v1", "v2", "v3"}
    assert len(set(variants.values())) == 4


def test_every_instruction_carries_the_rules_text() -> None:
    """输入：一条规格。输出：无。

    预期行为：八道题的题面都以判据原文结尾——判据不写进题面，读者就没法自己算（范式 §9.8）。
    """
    rules = questions.rules_block(LOC)
    for item in questions.build(_make_spec(4), LOC, 0, random.Random(0)):
        assert item.question["instructions"].endswith(rules)


def test_noul_criteria_are_filled_with_the_field_label() -> None:
    """输入：一条规格。输出：无。预期行为：判据里的 `{field}` 换成这一族的栏名，不留占位符。"""
    for item in questions.build(_make_spec(5), LOC, 0, random.Random(0)):
        if item.question["type"] != "noul":
            continue
        field = questions.NOUL_FIELDS[item.family]
        for text in item.question["criteria"].values():
            assert "{" not in text
            assert LOC["field"][field] in text


def test_score_criteria_are_the_three_bands() -> None:
    """输入：一条规格。输出：无。预期行为：档位描述三行，下标与规格的档位一致。"""
    spec = _make_spec(6)
    band = next(item for item in questions.build(spec, LOC, 0, random.Random(0)) if item.family == "band")
    assert band.question["criteria"] == list(LOC["fam"]["band"]["levels"])
    assert band.answer == spec.band
    assert 0 <= band.answer < len(band.question["criteria"])


def test_phrasings_rotate_the_head_line() -> None:
    """输入：三种问法编号。输出：无。

    预期行为：问法编号换一条，题面第一行就换一句（判据那段不变）——问法是文本维，不参与判据。
    """
    spec = _make_spec(7)
    heads = {
        phrasing: questions.build(spec, LOC, phrasing, random.Random(0))[0]
        .question["instructions"]
        .splitlines()[0]
        for phrasing in (0, 1, 2)
    }
    assert len(set(heads.values())) == 3
    for phrasing, head in heads.items():
        assert head == LOC["fam"]["verdict"]["q"][phrasing]


def test_build_ignores_the_rng() -> None:
    """输入：两个不同种子的 RNG。输出：无。预期行为：题目完全由规格与语言文件决定。"""
    spec = _make_spec(8)
    first = questions.build(spec, LOC, 0, random.Random(1))
    second = questions.build(spec, LOC, 0, random.Random(2))
    assert [item.question for item in first] == [item.question for item in second]
    assert [item.answer for item in first] == [item.answer for item in second]


def test_rules_block_comes_from_the_locale() -> None:
    """输入：语言字典。输出：无。预期行为：判据原文逐行来自语言文件，不在这里另写一份。"""
    assert questions.rules_block(LOC) == "\n".join(LOC["rules"]["lines"])
