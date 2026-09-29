"""特征层测试：覆盖率、名次计划、金标签、case 枚举。"""

from __future__ import annotations

import pytest

from workflows.rerank import features


def test_coverage_is_case_insensitive_substring() -> None:
    """输入：一段原文与词表。输出：无。预期行为：按大小写不敏感的子串计数，不看词序。"""
    assert features.coverage("Alpha and BRAVO", ["alpha", "bravo"]) == 2
    assert features.coverage("only alpha here", ["alpha", "bravo"]) == 1
    assert features.coverage("nothing at all", ["alpha", "bravo"]) == 0


def test_pairs_cover_both_directions() -> None:
    """输入：`PAIRS`。输出：无。预期行为：每个无序对恰好出现两个方向，且总共 20 条。"""
    assert len(features.PAIRS) == features.CANDIDATES * (features.CANDIDATES - 1)
    unordered = {frozenset(pair) for pair in features.PAIRS}
    assert len(unordered) == features.CANDIDATES * (features.CANDIDATES - 1) // 2
    assert all(left != right for left, right in features.PAIRS)
    assert (0, 1) in features.PAIRS and (1, 0) in features.PAIRS


@pytest.mark.parametrize("relevant", features.RELEVANT)
@pytest.mark.parametrize("best", features.BESTS)
def test_plan_realizes_the_injected_features(relevant: int, best: str) -> None:
    """输入：每个 (相关条数, 位置段)。输出：无。

    预期行为：计划里的相关条数就是注入的那个数；有相关候选时，第一名落在注入的那一段里、
    且恰好只有它拿到满覆盖。
    """
    case = features.RerankCase(terms=3, relevant=relevant, best=best)
    plan = features.plan_for(case, 0)
    assert plan.relevant == relevant
    assert len(plan.overlaps) == features.CANDIDATES
    if relevant == 0:
        assert plan.best_pos is None
        assert set(plan.overlaps) == {0}
        return
    assert plan.best_pos in features.BEST_THIRDS[best]
    assert plan.overlaps[plan.best_pos] == 3
    assert sorted(plan.overlaps)[-2] < 3  # 满分只有一个，第一名才唯一


def test_plan_is_reproducible_and_phrasing_matters() -> None:
    """输入：同一 case 的两种问法。输出：无。

    预期行为：同 (case, 问法) 必得同一份计划（构造侧与审计侧各算一遍也不会打架）；
    不同问法的**位置排布**至少有一个不一样——否则两种问法会渲染出结构相同的样本。
    """
    case = features.RerankCase(terms=2, relevant=3, best="last")
    assert features.plan_for(case, 0) == features.plan_for(case, 0)
    assert features.plan_for(case, 0).overlaps != features.plan_for(case, 1).overlaps


def test_label_order_breaks_ties_by_position() -> None:
    """输入：一份含并列的计划。输出：无。预期行为：先比得分、同分时原本靠前的排前面。"""
    plan = features.RerankPlan(terms=3, best_pos=3, overlaps=(1, 0, 1, 3, 0))
    labels = features.label_for(plan)
    assert labels.order == (3, 0, 2, 1, 4)
    assert labels.top == 3
    assert labels.relevant == 3
    assert labels.rank_of(0) == 2
    assert labels.above(0, 2) is True
    assert labels.above(2, 0) is False
    assert labels.above(1, 4) is True  # 同分（都是 0）时靠前的赢


def test_label_top_is_none_without_relevant_candidates() -> None:
    """输入：一条都不相关的计划。输出：无。预期行为：`top` 是 None（没有唯一解）。"""
    plan = features.RerankPlan(terms=2, best_pos=None, overlaps=(0,) * features.CANDIDATES)
    labels = features.label_for(plan)
    assert labels.top is None
    assert labels.relevant == 0
    assert labels.order == tuple(range(features.CANDIDATES))


def test_build_cases_enumerates_the_whole_space() -> None:
    """输入：默认维度表。输出：无。预期行为：全枚举 3×5×3 = 45 组，顺序稳定、无重复。"""
    cases = features.build_cases(512)
    assert len(cases) == len(features.TERM_COUNTS) * len(features.RELEVANT) * len(features.BESTS)
    assert len({case.fingerprint() for case in cases}) == len(cases)
    assert cases == features.build_cases(512)
    assert cases[0].describe() == "2词-0相关-first"


def test_case_as_dict_has_stable_key_order() -> None:
    """输入：一个 case。输出：无。预期行为：键序固定（哈希分桶按它算，键序变了切分会漂）。"""
    case = features.RerankCase(terms=2, relevant=1, best="middle")
    assert list(case.as_dict()) == ["terms", "relevant", "best"]
    assert case.fingerprint() == "terms=2|relevant=1|best=middle"
