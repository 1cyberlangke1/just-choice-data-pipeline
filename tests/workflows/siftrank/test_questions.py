"""题族：四道题、题型、答案、判据写进题面、选项表与等级表。"""

from __future__ import annotations

import random


from tests.workflows.siftrank.conftest import _make_spec
from workflows.siftrank import locales, questions
from workflows.siftrank import spec as sp

LOC = locales.load("zh")


def test_families_are_the_four_known_ones() -> None:
    """输入：无。输出：题族清单。

    预期行为：四个题族、顺序稳定（先两个 choice、再 score、最后那个布尔），三张分类表正好
    把四族分完——多一族少一族都会让下面的「答案在选项里」漏查。
    """
    assert questions.FAMILIES == ("agg_top", "pair_order", "cut_size", "stable")
    assert questions.CHOICE_FAMILIES == ("agg_top", "pair_order")
    assert questions.SCORE_FAMILIES == ("cut_size",)
    assert questions.NOUL_FAMILIES == ("stable",)
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
    for family in questions.CHOICE_FAMILIES:
        assert types[family] == "choice"
    assert types["cut_size"] == "score"
    assert types["stable"] == "noul"


def test_only_choice_options_may_be_shuffled() -> None:
    """输入：一条规格。输出：无。

    预期行为：只有 choice 允许打乱——`cut_size` 的三档有自然顺序（落差从第 1 名往后挪），
    打乱它会把「档位」读成「随便一个等级」。
    """
    for item in questions.build(_make_spec(2), LOC, 0, random.Random(0)):
        assert item.shuffle is (item.question["type"] == "choice"), item.family


def test_answers_match_the_spec_for_every_case() -> None:
    """输入：全空间 case。输出：无。

    预期行为：四道题的答案都与真值规格算出来的一致——choice 是选项名（组装器再换成编号）、
    noul 是布尔、score 是等级下标。这里是「生成器把答案算错了」那条路的第一道闸。
    """
    for case in sp.build_cases(sp.RankConfig(split="train")):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        answers = {
            item.family: item.answer
            for item in questions.build(spec, LOC, 0, random.Random(0))
        }
        assert answers["agg_top"] == case.top(), case.describe()
        assert answers["pair_order"] == case.relation(), case.describe()
        assert answers["cut_size"] == sp.CUTS.index(case.cut()), case.describe()
        assert answers["stable"] is case.stable(), case.describe()


def test_choice_answers_are_option_names_inside_the_criteria() -> None:
    """输入：全空间 case。输出：无。

    预期行为：choice 的答案必须是 `criteria` 里的**键**——组装器靠 `list(criteria).index`
    取编号，答案不在选项里会在组装期直接报错（那是构造 bug，不能进产物）。
    """
    for case in sp.build_cases(sp.RankConfig(split="train")):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        for item in questions.build(spec, LOC, 0, random.Random(0)):
            if item.question["type"] != "choice":
                continue
            assert item.answer in item.question["criteria"], item.family


def test_top_options_are_the_highest_totals() -> None:
    """输入：一条规格。输出：无。

    预期行为：`agg_top` 的选项就是总分最高的四个条目、按展示顺序排——四个都「看起来像」答案，
    正确答案是其中的第一个（展示顺序的头名）。
    """
    spec = _make_spec(3, shape="twin")
    options = questions.top_options(spec, LOC)
    ordered = sp.order_of(spec.totals)
    assert list(options) == list(ordered[: questions.TOP_OPTIONS])
    assert len(options) == questions.TOP_OPTIONS
    assert spec.top == ordered[0]
    assert spec.top in options


def test_pair_options_are_the_three_relations() -> None:
    """输入：一条规格。输出：无。

    预期行为：三个选项键与 `spec.RELATIONS` 一致、描述互不相同，而且每条描述里只点名一个
    条目（判别器靠这个认人）。
    """
    spec = _make_spec(4, shape="twin")
    options = questions.pair_options(spec, LOC)
    assert set(options) == set(sp.RELATIONS)
    assert len(set(options.values())) == len(sp.RELATIONS)
    names = [str(value) for value in LOC["item"].values()]
    for key in ("a_first", "b_first"):
        assert sum(1 for name in names if name in options[key]) == 1, options[key]


def test_score_criteria_is_the_three_level_table() -> None:
    """输入：一条规格。输出：无。

    预期行为：`criteria` 是三档描述（原序，不打乱），答案就是这个数组里的下标——下标与
    `spec.CUTS` 的顺序一一对应。
    """
    spec = _make_spec(5, shape="stair")
    item = next(
        item for item in questions.build(spec, LOC, 0, random.Random(0)) if item.family == "cut_size"
    )
    assert item.question["criteria"] == list(LOC["fam"]["cut_size"]["levels"])
    assert len(set(item.question["criteria"])) == len(sp.CUTS)
    assert item.answer == sp.CUTS.index(spec.cut)
    assert 0 <= item.answer < len(item.question["criteria"])


def test_noul_criteria_has_true_and_false() -> None:
    """输入：一条规格。输出：无。

    预期行为：noul 带 `criteria`（官方支持描述，那我们就带上）——真 / 假两行都在，
    答案必须是布尔而不是字符串 `"true"`。
    """
    item = next(
        item
        for item in questions.build(_make_spec(6), LOC, 0, random.Random(0))
        if item.family == "stable"
    )
    assert set(item.question["criteria"]) == {"true", "false"}
    assert isinstance(item.answer, bool)


def test_phrasing_index_cycles_through_the_pool() -> None:
    """输入：三种问法编号。输出：无。

    预期行为：问法按编号取，取遍池子里的每一条——同一 case 的多条样本靠它换说法，
    三种编号给同一句话等于这一维白设。
    """
    pool = LOC["fam"]["agg_top"]["q"]
    seen = {
        questions.build(_make_spec(10), LOC, index, random.Random(0))[0].question["instructions"]
        for index in range(len(pool))
    }
    assert len(seen) == len(pool)
