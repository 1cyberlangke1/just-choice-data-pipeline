"""真值规格的测试：模板形状、三条答案轴的分布、两条独立路径、素材指纹。"""

from __future__ import annotations

from collections import Counter
from typing import Final

import pytest

from workflows.siftrank import spec as sp

#: 全空间的 case（默认配置 = 三个维全开）。
FULL: Final[tuple[sp.RankCase, ...]] = tuple(sp.build_cases(sp.RankConfig(split="train")))

#: 每个模板**按构造**应当落到的答案（形状 → (相关条目数, 谁更相关, 批次是否一致)）。
#: 改模板时先改这张表：改不动就说明这次改动的分布后果没算清楚（范式 §9.70 一）。
DESIGN: Final[dict[str, tuple[int, str, bool]]] = {
    "peak": (1, sp.A_FIRST, True),
    "surge": (1, sp.A_FIRST, True),
    "twin": (2, sp.TIE, False),
    "swirl": (2, sp.TIE, False),
    "chain": (3, sp.B_FIRST, False),
    "stair": (3, sp.B_FIRST, True),
}


def test_case_space_is_not_empty() -> None:
    """输入：无。输出：无。预期行为：全空间够大，否则下面的分布尺子量不到东西。"""
    assert len(FULL) >= 60, f"全空间只有 {len(FULL)} 个 case"


def test_templates_are_well_formed() -> None:
    """输入：无。输出：无。

    预期行为：每个模板都正好 `TRIALS` 个批次、每批 `BATCH_SIZE` 个**互不相同**的位置，
    且位置都落在条目池里——批次里出现重复条目会让「总分」与「名次」自相矛盾。
    """
    assert len(sp.TEMPLATES) == len(DESIGN), "模板清单与设计表对不上"
    for shape, template in sp.TEMPLATES.items():
        assert len(template.batches) == sp.TRIALS, shape
        for batch in template.batches:
            assert len(batch) == sp.BATCH_SIZE, shape
            assert len(set(batch)) == sp.BATCH_SIZE, shape
            assert all(0 <= position < len(sp.ITEMS) for position in batch), shape
        assert len(set(template.pair)) == 2, shape
        assert all(0 <= position < len(sp.ITEMS) for position in template.pair), shape


@pytest.mark.parametrize("shape", sp.SHAPES)
def test_every_template_matches_the_design_table(shape: str) -> None:
    """输入：一个模板。输出：无。

    预期行为：三种映射下，这一模板的 `cut` / `relation` / `stable` 都必须与设计表一致——
    换映射只换「哪个条目占哪个位」，不该改任何答案的形状。
    """
    want = DESIGN[shape]
    for perm in sp.PERM_NAMES:
        case = sp.RankCase(shape, perm, sp.FLAVORS[0])
        assert (case.cut(), case.relation(), case.stable()) == want, case.describe()


def test_every_answer_axis_is_balanced_by_construction() -> None:
    """输入：全空间 case。输出：无。

    预期行为：三条答案轴**按构造**均匀——`cut` 取 1/2/3 各三分之一、`relation` 取三种各
    三分之一、`stable` 真伪各半。这是选模板时就手算好的分布（范式 §9.70 一），
    这一条把它钉在数据上：以后加模板时先算它会把哪一轴拽偏。
    """
    cuts = Counter(case.cut() for case in FULL)
    relations = Counter(case.relation() for case in FULL)
    stables = Counter(case.stable() for case in FULL)
    assert set(cuts) == set(sp.CUTS), cuts
    assert set(relations) == set(sp.RELATIONS), relations
    assert set(stables) == {True, False}, stables
    assert len(set(cuts.values())) == 1, cuts
    assert len(set(relations.values())) == 1, relations
    assert len(set(stables.values())) == 1, stables


def test_totals_always_add_up_to_the_batches() -> None:
    """输入：全空间 case。输出：无。

    预期行为：总分之和恒等于「批次数 × 每批得分之和」——每个批次贡献 0+1+…+(k-1) 分。
    这条是计分口径的算术自检：写错系数（例如从 1 起算）会让等式两边差一个常数。
    """
    want = sp.TRIALS * (sp.BATCH_SIZE * (sp.BATCH_SIZE - 1) // 2)
    for case in FULL:
        assert sum(case.totals().values()) == want, case.describe()


def test_unknown_shape_and_perm_are_rejected() -> None:
    """输入：表外的形状 / 映射。输出：无。预期行为：报 KeyError，不悄悄退回第一个模板。"""
    with pytest.raises(KeyError):
        sp.RankCase("nowhere", "identity", sp.FLAVORS[0]).batches()
    with pytest.raises(KeyError):
        sp.RankCase("peak", "sideways", sp.FLAVORS[0]).batches()


def test_cut_of_all_equal_totals_counts_every_item() -> None:
    """输入：所有条目总分相同的总分表。输出：无。

    预期行为：没有落差时**全部算相关**——返回条目总数，而不是 0 或者报错。
    """
    totals = {key: 3 for key in sp.ITEMS}
    assert sp.cut_of(totals) == len(sp.ITEMS)


def test_cut_takes_the_leftmost_of_several_equal_gaps() -> None:
    """输入：多处落差同样大的总分表。输出：无。

    预期行为：取**最靠前**的那一处——判据里写死了这一条，取最后一处会让答案在两种口径下
    不同，而两种都「说得通」，属于必须钉死的那类歧义。
    """
    totals = {key: 0 for key in sp.ITEMS}
    totals["overview"], totals["runbook"] = 6, 6
    totals["changelog"], totals["metrics"] = 4, 4
    totals["alerts"], totals["postmortem"] = 2, 2
    assert sp.cut_of(totals) == 2


def test_relation_of_reports_a_tie_on_equal_totals() -> None:
    """输入：总分相同的两个条目。输出：无。预期行为：判并列，而不是按清单顺序分先后。"""
    totals = {key: 0 for key in sp.ITEMS}
    assert sp.relation_of(totals, "alerts", "postmortem") == sp.TIE
    totals["alerts"] = 1
    assert sp.relation_of(totals, "alerts", "postmortem") == sp.A_FIRST
    assert sp.relation_of(totals, "postmortem", "alerts") == sp.B_FIRST


def test_is_stable_ignores_pairs_that_never_co_occur() -> None:
    """输入：两条完全不重叠的批次。输出：无。

    预期行为：没有共同出现过的对不参与判定——否则「四个批次各排各的」会被误判成不一致。
    """
    batches = (("overview", "runbook"), ("changelog", "metrics"))
    assert sp.is_stable(batches)
    crossed = (("overview", "runbook"), ("runbook", "overview"))
    assert not sp.is_stable(crossed)


def test_both_paths_agree_on_every_case() -> None:
    """输入：全空间 case。输出：无。

    预期行为：构造侧（`RankCase` 的属性）与审计侧（`observed_labels` 从批次重算）对四个答案
    **逐条一致**——两条独立路径是这一族的判据骨架，一条歪了整族都不可信。
    """
    for case in FULL:
        seen = sp.observed_labels(case.batches())
        left, right = case.pair()
        assert seen["cut"] == case.cut(), case.describe()
        assert seen["stable"] == case.stable(), case.describe()
        assert seen["top"] == case.top(), case.describe()
        assert sp.relation_of(seen["totals"], left, right) == case.relation(), case.describe()
        assert seen["order"] == list(sp.order_of(case.totals())), case.describe()


def test_material_ignores_the_text_only_dim() -> None:
    """输入：同一形状 + 同一映射、四种提示。输出：无。

    预期行为：同一份请求的几种提示**共用同一个素材指纹**——切分按素材分桶，指纹里带上
    提示就会把同一份请求拆到两个切分里去。
    """
    base = sp.RankCase("peak", "identity", sp.FLAVORS[0])
    for flavor in sp.FLAVORS:
        assert sp.RankCase("peak", "identity", flavor).material() == base.material()


def test_request_ids_are_stable_and_share_across_flavors() -> None:
    """输入：同一素材的四种提示。输出：无。

    预期行为：请求号由**素材**指纹派生，不含提示——「同输入同标签」这条一致性校验靠它认人；
    同一素材在任何进程里都得到同一个号。
    """
    ids = {sp.request_id_of(sp.RankCase("twin", "reverse", flavor)) for flavor in sp.FLAVORS}
    assert len(ids) == 1, ids
    assert ids == {sp.request_id_of(sp.RankCase("twin", "reverse", sp.FLAVORS[0]))}


def test_ood_side_only_holds_the_held_out_flavor() -> None:
    """输入：OOD 配置。输出：无。

    预期行为：OOD 侧的每一条都带一个训练侧没见过的提示，而且**只**带那一个（`unseen_dims`
    非空）；同分布侧一条都不带。
    """
    ood = sp.build_cases(sp.RankConfig(split="test_ood", ood_only=True))
    assert ood, "OOD 侧一个 case 都没有"
    for case in ood:
        assert case.unseen_dims(), case.describe()
        assert case.flavor not in sp.SEEN_FLAVORS, case.describe()
    for case in sp.build_cases(sp.RankConfig(split="train", flavors=sp.ID_FLAVORS)):
        assert not case.unseen_dims(), case.describe()
