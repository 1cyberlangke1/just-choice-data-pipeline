"""题族的单元测试：题型、金答案、选项唯一性、打乱口径。"""

from __future__ import annotations

import copy
import random
from collections import Counter
from typing import Any

import pytest

from jc import assemble
from workflows.search import locales, questions
from workflows.search import spec as sp


def build(spec: sp.SearchSpec, locale: str = "zh", phrasing: int = 0) -> list[Any]:
    """输入：规格、语言、问法。输出：题目 + 金答案。"""
    return questions.build(spec, locales.load(locale), phrasing, random.Random(0))


def test_all_families_are_produced_with_the_right_types(make_spec: Any) -> None:
    """输入：一页普通结果。输出：无。预期行为：8 个题族齐全，题型与声明一致。"""
    spec = make_spec()
    labeled = build(spec)
    assert [item.family for item in labeled] == list(questions.FAMILIES)
    types = {item.family: item.question["type"] for item in labeled}
    assert types["best_result"] == "choice" and types["next_action"] == "choice"
    assert types["relevance"] == "score" and types["covered_points"] == "score"
    assert all(types[name] == "noul" for name in questions.NOUL_FAMILIES)


def test_answers_match_the_page_facts(make_spec: Any) -> None:
    """输入：几条代表性页面。输出：无。

    预期行为：金答案就是按题面那套规则从页面推出来的——答案与判据同源。
    """
    spec = make_spec(pair_kind="conflict", best_flavor="full")
    answers = {item.family: item.answer for item in build(spec)}
    loc = locales.load("zh")
    points = tuple(loc["topic"][spec.topic.name]["points"])
    rows = questions._truth_rows(spec, loc)
    top = sp.best_of(rows)
    assert answers["best_result"] == f"result_{top.index}"
    assert answers["next_action"] == "answer"  # 完整命中 → 直接回答
    assert answers["answerable"] is True
    assert answers["conflict"] is True  # 被点名那两条互相矛盾
    assert answers["same_source"] is False
    assert answers["covered_points"] == sp.covers_of(
        next(row for row in rows if row.index == spec.case.probe), points
    )


def test_stale_best_switches_source(make_spec: Any) -> None:
    """输入：最相关那条过期的页面。输出：无。预期行为：下一步是换数据源。"""
    spec = make_spec(best_flavor="stale")
    answers = {item.family: item.answer for item in build(spec)}
    assert answers["next_action"] == "switch_source"


def test_partial_best_searches_again(make_spec: Any) -> None:
    """输入：最相关那条要点不全的页面。输出：无。预期行为：下一步是再检一轮。"""
    spec = make_spec(best_flavor="partial")
    answers = {item.family: item.answer for item in build(spec)}
    assert answers["next_action"] == "search_more"
    assert answers["answerable"] is False


def test_next_action_reaches_all_three_branches() -> None:
    """输入：三档最佳结果角色。输出：无。预期行为：三个动作都出现过（没有死选项）。"""
    seen = {
        next(
            item.answer
            for item in build(make_spec_fixture(flavor))
            if item.family == "next_action"
        )
        for flavor in sp.BEST_FLAVORS
    }
    assert seen == {"answer", "search_more", "switch_source"}


def make_spec_fixture(flavor: str) -> sp.SearchSpec:
    """输入：最佳结果角色。输出：一条规格。预期行为：配方挑一条与该角色相容的。"""
    mix = next(mix for mix in sp.MIXES if sp.mix_fits(mix, flavor))
    case = sp.SearchCase("refund", flavor, mix, (1, 2), 3, "conflict", 1, 0)
    return sp.sample_spec(random.Random(0), case, "train", 0, 0)


def test_option_descriptions_are_unique_even_when_two_rows_share_a_document() -> None:
    """输入：`both` 那一档（同源又矛盾）。输出：无。

    预期行为：选项描述仍互不相同——被点名那两条是同一份文档、同一个标题，
    只有更新日期把它们分开；描述里带上日期才有唯一性（`_choice` 会当场报重复）。
    """
    spec = make_spec_fixture_with("both")
    options = questions._result_options(spec, locales.load("zh"))
    assert len(set(options.values())) == len(options)


def make_spec_fixture_with(pair_kind: str) -> sp.SearchSpec:
    """输入：被点名两条的关系。输出：一条规格（最佳结果用 `full`）。"""
    case = sp.SearchCase("refund", "full", sp.MIXES[0], (3, 4), 5, pair_kind, 1, 0)
    return sp.sample_spec(random.Random(0), case, "train", 0, 0)


def test_duplicate_option_descriptions_are_rejected() -> None:
    """输入：两个描述一模一样的选项。输出：异常。

    预期行为：直接报错——描述重复意味着考生分不清两个选项，是构造 bug。
    """
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("best_result", "题面", {"a": "同样的话", "b": "同样的话"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("best_result", "题面", {"a": "甲", "b": "乙"}, "c")


def test_shuffle_policy_matches_the_family(make_spec: Any) -> None:
    """输入：全部题目。输出：无。

    预期行为：`best_result` 不打乱（选项名 `result_<编号>` 有自然顺序），
    `next_action` 打乱（三档由描述承载、不靠位置承载），noul 与 score 都不打乱。
    """
    labeled = build(make_spec())
    for item in labeled:
        if item.family == "best_result":
            assert item.shuffle is False
        elif item.family == "next_action":
            assert item.shuffle is True
        else:
            assert item.shuffle is False


def test_next_action_positions_do_not_leak(make_spec: Any) -> None:
    """输入：同一道 next_action 题组装很多次。输出：答案位置的分布。

    预期行为：三个位置都出现过——不打乱就等于把「答案位置」白送给考生（范式 §5.8）。
    """
    spec = make_spec()
    labeled = [item for item in build(spec) if item.family == "next_action"]
    positions: Counter[int] = Counter()
    for seed in range(60):
        row = assemble.assemble_sample("search-train-0000000", "state", labeled, random.Random(seed))
        positions[row["answers"]["next_action"]["choice"]] += 1
    assert set(positions) == {0, 1, 2}


def test_best_result_answer_position_follows_the_layout(make_spec: Any) -> None:
    """输入：四种版式。输出：无。

    预期行为：`best_result` 不打乱，所以答案编号就是最佳结果的位置减去 1——
    版式本身把四个位置都排到了，答案位置因此不是一个常量。
    """
    seen = set()
    for best_at in (3, 4, 5, 6):
        spec = sp.sample_spec(
            random.Random(0),
            sp.SearchCase("refund", "full", sp.MIXES[0], (1, 2), best_at, "none", 1, 0),
            "train",
            0,
            0,
        )
        labeled = build(spec)
        row = assemble.assemble_sample("search-train-0000000", "state", labeled, random.Random(0))
        seen.add(row["answers"]["best_result"]["choice"])
    assert seen == {2, 3, 4, 5}


def test_score_answers_are_indices_into_the_levels(make_spec: Any) -> None:
    """输入：几条页面。输出：无。预期行为：score 答案在等级表范围内，等级表就是题面。"""
    loc = locales.load("zh")
    for flavor in sp.BEST_FLAVORS:
        spec = make_spec_fixture(flavor)
        for item in build(spec):
            if item.question["type"] != "score":
                continue
            assert item.question["criteria"] == (
                loc["fam"][item.family]["levels"]
            )
            assert 0 <= item.answer < len(item.question["criteria"])


def test_question_texts_carry_the_asked_index(make_spec: Any) -> None:
    """输入：一条页面。输出：无。

    预期行为：问「结果 N」的那几族题面里确实出现了 case 点名的编号，
    问「结果 i 与结果 j」的出现了被点名那一对——判别器就是靠这个重算的。
    """
    spec = make_spec(probe=2)
    texts = {item.family: item.question["instructions"] for item in build(spec)}
    for family in questions.PROBE_FAMILIES:
        assert "2" in texts[family], family
    for family in questions.PAIR_FAMILIES:
        assert "1" in texts[family] and "2" in texts[family], family


def test_missing_family_entry_reports_the_name(make_spec: Any) -> None:
    """输入：语言文件里少了一族题的问法。输出：异常。预期行为：报错点名题族。

    预期行为（细节）：先 `deepcopy` 再删——`locales.load` 带缓存，
    直接改会把「少了一族」这个状态漏给后面的测试。
    """
    loc = copy.deepcopy(locales.load("zh"))
    del loc["fam"]["conflict"]
    with pytest.raises(KeyError, match="conflict"):
        questions.build(make_spec(), loc, 0, random.Random(0))
