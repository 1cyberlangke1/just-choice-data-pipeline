"""真值规格的不变量：画像表均衡、破并列规则、排序全序、值留出。"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

import pytest

from workflows.rerank import spec as sp


def order_key(cands: tuple[sp.Cand, ...], index: int) -> tuple[Any, ...]:
    """输入：四条候选、下标。输出：排序键（与 `spec.order_of` 同构，独立写一份当尺子）。"""
    cand = cands[index]
    return (sp.relevance(cand), sp.SOURCE_RANK[cand.source], cand.day, -index)


def deciding_rule(cands: tuple[sp.Cand, ...], left: int, right: int) -> int:
    """输入：四条候选、名次相邻的两条。输出：分出先后的那条规则的序号（0~3）。

    预期行为：返回**第一个**不相同的键位——0 相关度 / 1 权威度 / 2 日期 / 3 编号。
    """
    for position, (a, b) in enumerate(zip(order_key(cands, left), order_key(cands, right))):
        if a != b:
            return position
    raise AssertionError(f"两条候选的排序键完全相同：{left} / {right}")


def test_shape_table_has_ten_four_cand_shapes() -> None:
    """输入：画像表。输出：无。预期行为：十个画像、每个四条候选。"""
    assert len(sp.SHAPES) == 10
    assert all(len(shape.cands) == 4 for shape in sp.SHAPES)


def test_cand_bits_are_real_booleans() -> None:
    """输入：画像表。输出：无。

    预期行为：三个布尔位是 `bool` 而不是 `0`/`1`（int）。画像表里写的是 0/1，
    解包时漏了转换就会让 `noul` 的答案变成 int，装配层当场拒掉整条 case
    （实测报错：`noul 的答案必须是布尔：on_topic -> 1`）。
    """
    for shape in sp.SHAPES:
        for cand in shape.cands:
            assert isinstance(cand.topic_ok, bool), cand
            assert isinstance(cand.aspect_ok, bool), cand
            assert isinstance(cand.example, bool), cand


def test_on_topic_of_returns_a_real_boolean() -> None:
    """输入：每条候选。输出：无。预期行为：`on_topic_of` 恒返回 `bool`（不是 int）。"""
    for shape in sp.SHAPES:
        for cand in shape.cands:
            assert isinstance(sp.on_topic_of(cand), bool)


def test_probe_relevance_levels_are_flat() -> None:
    """输入：画像表。输出：无。预期行为：被点名那条的相关度 0~4 各出现两次。"""
    counts = Counter(sp.relevance(shape.cands[shape.probe]) for shape in sp.SHAPES)
    assert dict(sorted(counts.items())) == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}


def test_probe_on_topic_is_flat() -> None:
    """输入：画像表。输出：无。预期行为：被点名那条「主题 + 方面都对」5 真 5 假。"""
    counts = Counter(sp.on_topic_of(shape.cands[shape.probe]) for shape in sp.SHAPES)
    assert counts[True] == 5 and counts[False] == 5


def test_name_answers_are_flat() -> None:
    """输入：画像表。输出：无。预期行为：第一 / 第 j / 最后 的答案都是 3/3/2/2。"""
    first: Counter = Counter()
    nth: Counter = Counter()
    last: Counter = Counter()
    for shape in sp.SHAPES:
        order = sp.order_of(shape.cands)
        first[order[0]] += 1
        nth[order[shape.nth - 1]] += 1
        last[order[-1]] += 1
    for counts in (first, nth, last):
        assert sorted(counts.values()) == [2, 2, 3, 3], counts


def test_pair_answers_are_flat() -> None:
    """输入：画像表。输出：无。预期行为：逐对先后、相关度是否相同都是 5 真 5 假。"""
    ahead: Counter = Counter()
    tie: Counter = Counter()
    for shape in sp.SHAPES:
        order = sp.order_of(shape.cands)
        left, right = shape.ahead
        ahead[order.index(left) < order.index(right)] += 1
        left, right = shape.tie
        tie[sp.relevance(shape.cands[left]) == sp.relevance(shape.cands[right])] += 1
    for counts in (ahead, tie):
        assert counts[True] == 5 and counts[False] == 5, counts


def test_order_is_a_total_permutation() -> None:
    """输入：画像表。输出：无。

    预期行为：名次是 0~3 的一个排列——排序键里带 `-编号`，四条规则合起来保证不会
    出现「两条完全并列」这种排序结果依赖实现细节的情况。
    """
    for shape in sp.SHAPES:
        assert sorted(sp.order_of(shape.cands)) == [0, 1, 2, 3]


def test_every_tie_break_rule_is_used() -> None:
    """输入：画像表。输出：无。

    预期行为：②权威度、③日期、④编号三条破并列规则每条都**真的**分过至少一次先后。
    只写注释不写断言的话，「相关度相同时怎么排」这一课会整族缺位——实测原始画像表里
    ④编号一次都没被用到，而规则原文却写在每道题的题面里。
    """
    used: Counter = Counter()
    for shape in sp.SHAPES:
        order = sp.order_of(shape.cands)
        for left, right in zip(order, order[1:]):
            used[deciding_rule(shape.cands, left, right)] += 1
    assert used[1] >= 1, "②权威度一次都没用上"
    assert used[2] >= 1, "③日期一次都没用上"
    assert used[3] >= 1, "④编号一次都没用上"


def test_fresh_compares_iso_strings() -> None:
    """输入：时效线前后的两个日期。输出：无。预期行为：不早于时效线算满足。"""
    assert sp.fresh(sp.DEADLINE) is True
    assert sp.fresh("2024-05-31") is False
    assert sp.fresh("2025-01-01") is True


def test_relevance_counts_conditions_equally() -> None:
    """输入：几条候选。输出：无。预期行为：相关度 = 四条必要条件满足的条数（平权相加）。"""
    assert sp.relevance(sp.Cand(True, True, True, "blog", "2024-06-01")) == 4
    assert sp.relevance(sp.Cand(True, True, False, "blog", "2024-06-01")) == 3
    assert sp.relevance(sp.Cand(False, False, True, "official", "2024-06-01")) == 2
    assert sp.relevance(sp.Cand(False, False, False, "official", "2025-01-01")) == 1
    assert sp.relevance(sp.Cand(False, False, False, "official", "2023-11-18")) == 0
    assert sp.relevance(sp.Cand(True, False, False, "official", "2023-11-18")) == 1


def test_on_topic_ignores_freshness_and_example() -> None:
    """输入：主题 / 方面都对但既不新也没示例的候选。输出：无。

    预期行为：`on_topic_of` 为真而 `relevance` 只有 2——两族问的不是一回事，
    这正是「相关度五档」与「在不在讲这件事」能同时摊平的原因。
    """
    cand = sp.Cand(True, True, False, "blog", "2023-11-18")
    assert sp.on_topic_of(cand) is True
    assert sp.relevance(cand) == 2


def test_full_matrix_is_enumerated() -> None:
    """输入：默认配置。输出：无。预期行为：8 × 3 × 10 × 4 × 3 = 2880 条，顺序稳定。"""
    cases = sp.build_cases(sp.RerankConfig())
    assert len(cases) == 8 * 3 * 10 * 4 * 3 == 2880
    assert cases == sp.build_cases(sp.RerankConfig())


def test_ood_only_keeps_only_unseen_values() -> None:
    """输入：`ood_only` 配置。输出：无。预期行为：每条都至少含一个训练侧没见过的取值。"""
    cases = sp.build_cases(sp.RerankConfig(split="test_ood", ood_only=True))
    assert cases and all(case.unseen_dims() for case in cases)


def test_holdout_values_really_appear_on_the_ood_side() -> None:
    """输入：`ood_only` 的 case。输出：无。

    预期行为：三个被点名的维度在 OOD 侧都真有「训练侧没见过」的取值——
    留出清单是设计意图，这条把它钉在数据上。
    """
    ood = sp.build_cases(sp.RerankConfig(split="test_ood", ood_only=True))
    assert {case.topic for case in ood} - sp.SEEN_TOPICS
    assert {case.traps for case in ood} - sp.SEEN_TRAPS
    assert {case.wording for case in ood} - sp.SEEN_WORDINGS


def test_label_of_normalizes_the_subset_order() -> None:
    """输入：同一组合的两种书写顺序。输出：无。

    预期行为：标签按 `TRAPS_KEYS` 顺序归一——直接 `str(tuple)` 会把同一个集合写成
    两种样子，统计会裂开。
    """
    assert sp.label_of(("edited", "forwarded")) == "forwarded+edited"
    assert sp.label_of(("forwarded", "edited")) == sp.label_of(("edited", "forwarded"))
    assert sp.label_of(()) == "none"


def test_material_of_changes_with_every_dimension() -> None:
    """输入：默认 case 与五个维度各改一次的 case。输出：无。

    预期行为：五个维度都进素材指纹——否则切分会把两条不同的请求当成同一素材。
    """
    base = sp.RerankCase("backup", "steps", sp.SHAPES[0], (), "plain")
    variants = [
        sp.RerankCase("rotation", "steps", sp.SHAPES[0], (), "plain"),
        sp.RerankCase("backup", "config", sp.SHAPES[0], (), "plain"),
        sp.RerankCase("backup", "steps", sp.SHAPES[1], (), "plain"),
        sp.RerankCase("backup", "steps", sp.SHAPES[0], ("forwarded",), "plain"),
        sp.RerankCase("backup", "steps", sp.SHAPES[0], (), "terse"),
    ]
    materials = {sp.material_of(case) for case in variants}
    assert len(materials) == len(variants)
    assert sp.material_of(base) not in materials


def test_spec_answers_come_only_from_the_case() -> None:
    """输入：同 case、不同种子 / 切分 / 序号 / 问法。输出：无。

    预期行为：名次、相关度、查询号完全一致——构造式的规格里没有任何随机成分，
    这样「同素材同题必须同答」这条一致性校验才成立。
    """
    case = sp.RerankCase("cache", "config", sp.SHAPES[3], ("saved",), "titled")
    first = sp.sample_spec(random.Random(1), case, "train", 0, 0)
    second = sp.sample_spec(random.Random(99), case, "test_ood", 7, 2)
    assert first.order == second.order
    assert first.relevance == second.relevance
    assert first.query_id == second.query_id


def test_query_id_is_prefixed_and_split_free() -> None:
    """输入：一条 case。输出：无。

    预期行为：查询号形如 `Q-#####`，且不含切分名——把切分名拼进查询号就等于把切分
    信息写进 state，同一条请求在不同切分里会拿到不同的号。
    """
    case = sp.RerankCase("backup", "steps", sp.SHAPES[0], (), "plain")
    query_id = sp.query_id_of(case)
    assert query_id.startswith(sp.QUERY_PREFIX)
    assert query_id[len(sp.QUERY_PREFIX) :].isdigit()
    for split in ("train", "val", "test_iid", "test_ood"):
        assert split not in query_id


def test_spec_to_dict_is_json_friendly() -> None:
    """输入：一条规格。输出：无。预期行为：摘要可 JSON 序列化，含名次与相关度。"""
    case = sp.RerankCase("backup", "steps", sp.SHAPES[0], (), "plain")
    spec = sp.sample_spec(random.Random(0), case, "train", 3, 0)
    payload = json.loads(json.dumps(sp.spec_to_dict(spec), ensure_ascii=False))
    assert payload["query_id"] == spec.query_id
    assert payload["order"] == list(spec.order)
    assert payload["relevance"] == list(spec.relevance)


def test_shape_builder_rejects_a_short_list() -> None:
    """输入：只有三条候选。输出：无。预期行为：导入期就报错（画像表是手推的）。"""
    with pytest.raises(ValueError, match="四条候选"):
        sp._shape(((1, 1, 1, "official", "2024-07-05"),) * 3, 2, (0, 1), (1, 2), 0)


def test_shape_keys_are_distinct() -> None:
    """输入：画像表。输出：无。预期行为：每个画像的指纹互不相同（统计分组靠它）。"""
    keys = [shape.key() for shape in sp.SHAPES]
    assert len(set(keys)) == len(keys)
    assert all(shape.key() == shape.key() for shape in sp.SHAPES)
