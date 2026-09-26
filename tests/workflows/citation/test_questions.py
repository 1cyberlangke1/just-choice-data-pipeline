"""题族：七个题族、三种题型、选项名稳定、答案由规则引擎算出来、组装成契约行。"""

from __future__ import annotations

import random

import pytest

from jc import assemble
from workflows.citation import locales, questions, render
from workflows.citation import spec as sp


def _spec(**overrides: object) -> sp.CitationSpec:
    """输入：维度覆盖项。输出：真值规格（默认「鉴权 / 通过 / 照抄 / 三节」）。"""
    fields: dict[str, object] = {
        "topic": "auth",
        "status": "verified",
        "quote_form": "verbatim",
        "sections": 3,
        "noise": "plain",
        "claim_style": 1,
        "where": "middle",
        "doc_style": "bracketed",
    }
    fields.update(overrides)
    return sp.sample_spec(random.Random(3), sp.CitationCase(**fields), "train", 0, 0)  # type: ignore[arg-type]


def _build(spec: sp.CitationSpec, phrasing: int = 0) -> list[object]:
    """输入：规格、问法编号。输出：题目列表。预期行为：走真的题族构造器。"""
    return questions.build(spec, locales.load("zh"), phrasing, random.Random(1))


def test_families_are_the_seven_declared_ones() -> None:
    """输入：无。输出：无。预期行为：四个 choice + 两个 noul + 一个 score，顺序固定。"""
    assert questions.FAMILIES == (
        "verdict",
        "quote_source",
        "relation",
        "action",
        "found",
        "located",
        "evidence_strength",
    )
    assert set(questions.FAMILIES) == set(sp.answers_of(True, True, "supports"))


def test_build_returns_one_question_per_family_in_order() -> None:
    """输入：一份规格。输出：题目列表。预期行为：题号与题族一一对应、顺序与声明一致。"""
    built = _build(_spec())
    assert [item.family for item in built] == list(questions.FAMILIES)
    assert [item.question["type"] for item in built] == [
        "choice",
        "choice",
        "choice",
        "choice",
        "noul",
        "noul",
        "score",
    ]


@pytest.mark.parametrize(
    ("status", "form", "expected"),
    [
        ("verified", "verbatim", "verified"),
        ("contradicted", "verbatim", "contradicted"),
        ("unsupported", "verbatim", "unsupported"),
        ("mislocated", "verbatim", "unsupported"),
        ("fabricated", "reworded", "fabricated"),
        ("fabricated", "truncated", "fabricated"),
    ],
)
def test_answers_come_from_the_rule_engine(status: str, form: str, expected: str) -> None:
    """输入：每个结论档。输出：无。

    预期行为：七个答案全部等于 `spec.answers_of` 的结果——题族构造器不另起一套判断。
    """
    spec = _spec(status=status, quote_form=form)
    want = sp.answers_of(*spec.truth())
    answers = {item.family: item.answer for item in _build(spec)}
    assert answers == {family: want[family] for family in questions.FAMILIES}
    assert answers["verdict"] == expected


def test_choice_options_are_the_stable_identifiers() -> None:
    """输入：一份规格。输出：无。

    预期行为：choice 的 `criteria` 键就是选项名（`verified` 等稳定标识），金答案也是选项名——
    判别器靠它反查，不靠描述文字比对；描述文字只在选项的值里。
    """
    loc = locales.load("zh")
    for item in _build(_spec()):
        if item.question["type"] != "choice":
            continue
        assert set(item.question["criteria"]) == set(loc["option"][item.family])
        assert item.answer in item.question["criteria"]
        assert item.question["criteria"][item.answer] == loc["option"][item.family][item.answer]


def test_noul_and_score_carry_their_own_criteria() -> None:
    """输入：一份规格。输出：无。

    预期行为：noul 带 true/false 描述、score 的 `criteria` 就是等级表（4 档）；
    两者都**不打乱**选项（有自然顺序的集合不能乱）。
    """
    loc = locales.load("zh")
    built = {item.family: item for item in _build(_spec())}
    for family in questions.NOUL_FAMILIES:
        item = built[family]
        assert item.question["criteria"] == dict(loc["ask"][family]["criteria"])
        assert item.shuffle is False
    strength = built["evidence_strength"]
    assert strength.question["criteria"] == list(loc["ask"]["evidence_strength"]["levels"])
    assert strength.shuffle is False
    assert isinstance(strength.answer, int) and 0 <= strength.answer < 4
    for family in questions.CHOICE_FAMILIES:
        assert built[family].shuffle is True


def test_phrasing_picks_a_different_wording() -> None:
    """输入：同一份规格、两个问法编号。输出：无。预期行为：题面换了，答案不动。"""
    first = _build(_spec(), phrasing=0)
    second = _build(_spec(), phrasing=1)
    assert [item.question["instructions"] for item in first] != [
        item.question["instructions"] for item in second
    ]
    assert [item.answer for item in first] == [item.answer for item in second]


def test_build_does_not_roll_dice() -> None:
    """输入：同一份规格、两个不同的 RNG。输出：无。预期行为：题目逐字段一致。"""
    spec = _spec(status="mislocated", noise="decoy")
    first = questions.build(spec, locales.load("zh"), 0, random.Random(1))
    second = questions.build(spec, locales.load("zh"), 0, random.Random(999))
    assert first == second


def test_choice_helper_rejects_duplicate_descriptions_and_unknown_answers() -> None:
    """输入：坏的选项表。输出：异常。预期行为：构造 bug 在生成时当场炸，不产出坏数据。"""
    with pytest.raises(ValueError, match="描述重复"):
        questions._choice("verdict", "x", {"a": "同一个", "b": "同一个"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("verdict", "x", {"a": "1", "b": "2"}, "c")


def test_assemble_maps_the_answer_name_to_an_index() -> None:
    """输入：一份规格。输出：契约样本行。预期行为：choice 的答案存的是**选项下标**，
    指向的选项名就是规则引擎算出来的那个；noul 存布尔、score 存等级下标。
    """
    loc = locales.load("zh")
    rng = random.Random(7)
    spec = _spec(status="contradicted")
    labeled = questions.build(spec, loc, spec.phrasing, rng)
    row = assemble.assemble_sample("citation-train-0000000", render.build(spec, loc), labeled, rng)
    want = sp.answers_of(*spec.truth())
    assert set(row["questions"]) == set(questions.FAMILIES)
    assert set(row["answers"]) == set(questions.FAMILIES)
    for family in questions.CHOICE_FAMILIES:
        question = row["questions"][family]
        options = list(question["criteria"])
        assert options[row["answers"][family]["choice"]] == want[family]
    assert row["answers"]["found"]["noul"] is want["found"]
    assert row["answers"]["located"]["noul"] is want["located"]
    assert row["answers"]["evidence_strength"]["score"] == want["evidence_strength"]
