"""真值规格：目录 / 关系表 / 八个题族的分布 / 规则 / 配置校验。"""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from workflows.match import spec as sp


def test_catalog_matches_the_derived_tables() -> None:
    """输入：商品目录。输出：无。

    预期行为：规格 / 度数 / 包装的取值表由目录推出来（不手写第二份），商品键顺序稳定。
    """
    assert set(sp.CATALOG) == set(sp.PRODUCTS)
    assert sp.SIZES == tuple(sorted({value[0] for value in sp.CATALOG.values()}))
    assert sp.ABVS == tuple(sorted({value[1] for value in sp.CATALOG.values()}))
    assert sp.PACKS == tuple(sorted({value[2] for value in sp.CATALOG.values()}))


def test_fields_and_kinds_are_five_and_three() -> None:
    """输入：栏表与关系表。输出：无。预期行为：五栏、三种关系，且关系名不重复。"""
    assert len(sp.FIELDS) == 5
    assert sp.FIELDS[0] == "name" and sp.FIELDS[1] == "brand"
    assert sp.KINDS == (sp.SAME, sp.VARIANT, sp.DIFF)
    assert len(set(sp.KINDS)) == 3


def test_patterns_are_twelve_unique_shapes() -> None:
    """输入：十二种字段关系。输出：无。

    预期行为：两两不同、每条恰好五栏、取值都在三种关系里——关系表是八个题族分布的
    唯一来源，重复一条就是重复样本。
    """
    assert len(sp.PATTERNS) == 12
    assert len(set(sp.PATTERNS)) == 12
    for pattern in sp.PATTERNS:
        assert len(pattern) == len(sp.FIELDS)
        assert set(pattern) <= set(sp.KINDS)


def test_agreement_histogram_is_the_designed_one() -> None:
    """输入：十二种字段关系。输出：无。

    预期行为：一致字段数的直方图是 {0:1, 1:3, 2:3, 3:1, 4:2, 5:2}（总数 30 = 5 栏 × 6）
    ——它同时钉死三件事：三个档位各 4 条、每一栏 `diff` 恰好 6 条、variant 栏数 0~3 各 3 条。
    """
    counts = Counter(sp.agree_of(pattern) for pattern in sp.PATTERNS)
    assert dict(sorted(counts.items())) == {0: 1, 1: 3, 2: 3, 3: 1, 4: 2, 5: 2}
    assert sum(counts.values()) == len(sp.PATTERNS)
    assert sum(count * weight for count, weight in counts.items()) == len(sp.FIELDS) * 6


def test_every_question_family_is_balanced_by_construction() -> None:
    """输入：十二种字段关系。输出：无。

    预期行为：八个题族的答案在**关系表这一层**就摊平了——`band` 三档各 4 条、
    `verdict` 三档各 4 条、`variant_count` 四档各 3 条、五个 `same_<栏>` 各 6 真 6 假。
    """
    assert dict(Counter(sp.band_of(sp.agree_of(p)) for p in sp.PATTERNS)) == {0: 4, 1: 4, 2: 4}
    assert dict(Counter(sp.verdict_of(p) for p in sp.PATTERNS)) == {
        "different": 4,
        "same": 4,
        "review": 4,
    }
    assert dict(Counter(sp.variant_count_of(p) for p in sp.PATTERNS)) == {0: 3, 1: 3, 2: 3, 3: 3}
    for index, field in enumerate(sp.FIELDS):
        assert sum(1 for p in sp.PATTERNS if p[index] == sp.DIFF) == 6, field


def test_band_thresholds() -> None:
    """输入：一致字段数 0~5。输出：无。预期行为：≤1 → 0 档，2~3 → 1 档，≥4 → 2 档。"""
    assert [sp.band_of(count) for count in range(6)] == [0, 0, 1, 1, 2, 2]


def test_verdict_rules_are_mutually_exclusive() -> None:
    """输入：十二种字段关系。输出：无。

    预期行为：①（名称与品牌都不一致）与 ②（一致字段数 ≥ 4）不可能同时成立——五栏里
    前两栏不一致时，剩下三栏最多凑 3 栏一致。所以处置是一个干净的三分区，「按顺序判」
    只是把规则读起来像「先看标识、再数个数」，本身不承担判据。
    """
    for pattern in sp.PATTERNS:
        if pattern[0] == sp.DIFF and pattern[1] == sp.DIFF:
            assert sp.agree_of(pattern) < 4


def test_verdict_needs_both_identifiers_to_disagree() -> None:
    """输入：只坏一栏标识的关系。输出：无。

    预期行为：名称与品牌**都**不一致才是 different；只坏一个（哪怕五栏里只对得上三栏）
    仍然交人工复核——「一个标识栏还对得上就仍可能是同一个商品」的口径。
    """
    assert sp.verdict_of((sp.DIFF, sp.SAME, sp.DIFF, sp.DIFF, sp.DIFF)) == "review"
    assert sp.verdict_of((sp.DIFF, sp.DIFF, sp.SAME, sp.SAME, sp.SAME)) == "different"
    assert sp.verdict_of((sp.VARIANT, sp.SAME, sp.SAME, sp.SAME, sp.VARIANT)) == "same"
    assert sp.verdict_of((sp.SAME, sp.SAME, sp.SAME, sp.SAME, sp.SAME)) == "same"


def test_same_and_variant_counters() -> None:
    """输入：一条关系。输出：无。预期行为：`same_of` 只把 `diff` 记成假，`variant_count` 只数 variant。"""
    pattern = (sp.SAME, sp.VARIANT, sp.DIFF, sp.VARIANT, sp.SAME)
    assert sp.same_of(pattern) == (True, True, False, True, True)
    assert sp.agree_of(pattern) == 4
    assert sp.variant_count_of(pattern) == 2


def test_other_product_really_differs_in_that_field() -> None:
    """输入：每个商品 × 每一栏。输出：无。

    预期行为：`other_product` 换过去的商品在这一栏上的值**确实不同**——规格 330 出现两次、
    度数 0.0 出现四次，直接取「下一个」会让 `diff` 那一栏其实相等，那是把答案写错。
    """
    for product in sp.PRODUCTS:
        for field in sp.FIELDS:
            other = sp.other_product(field, product, sp.PRODUCTS)
            assert other != product
            assert sp.field_value(field, other) != sp.field_value(field, product), (field, product)


def test_diff_pool_keeps_the_held_out_product_out_of_the_id_side() -> None:
    """输入：三个同分布切分名 + `test_ood`。输出：无。

    预期行为：同分布侧的 `diff` 取值池**不含**留出商品，`test_ood` 才放开全目录；
    池子里每个商品、每一栏都取得到不同的值（取不到会直接报错，不静默少给）。
    """
    held = sp.PRODUCTS[-1]
    assert sp.diff_pool("test_ood") == sp.PRODUCTS
    assert held not in sp.ID_PRODUCTS
    for split in ("train", "val", "test_iid"):
        assert sp.diff_pool(split) == sp.ID_PRODUCTS, split
    for product in sp.ID_PRODUCTS:
        for field in sp.FIELDS:
            other = sp.other_product(field, product, sp.ID_PRODUCTS)
            assert other in sp.ID_PRODUCTS and other != held, (field, product, other)


def test_other_product_reports_an_empty_pool() -> None:
    """输入：空取值池。输出：无。预期行为：报人话（池子里挑不出不同的值，不静默少给）。"""
    with pytest.raises(ValueError, match="池子"):
        sp.other_product("name", "lager", ())


def test_field_value_uses_product_keys_for_name_and_brand() -> None:
    """输入：名称 / 品牌栏。输出：无。

    预期行为：名称与品牌拿商品键当代理值（每个商品的名称与品牌都唯一）；其余三栏取目录里的数值。
    """
    for product in sp.PRODUCTS:
        assert sp.field_value("name", product) == product
        assert sp.field_value("brand", product) == product
    size, abv, pack = sp.CATALOG["lager"]
    assert sp.field_value("size", "lager") == size
    assert sp.field_value("abv", "lager") == abv
    assert sp.field_value("pack", "lager") == pack


def test_build_cases_enumerates_the_matrix() -> None:
    """输入：全默认配置。输出：无。预期行为：1728 条、顺序稳定、可复算。"""
    cases = sp.build_cases(sp.MatchConfig())
    assert len(cases) == 8 * 12 * 3 * 3 * 2 == 1728
    assert cases == sp.build_cases(sp.MatchConfig())
    assert cases[0] == sp.MatchCase(sp.PRODUCTS[0], sp.PATTERNS[0], sp.SPELLINGS[0], "plain", "none")
    assert cases[-1] == sp.MatchCase(sp.PRODUCTS[-1], sp.PATTERNS[-1], sp.SPELLINGS[-1], "interleaved", "origin")


def test_ood_only_keeps_cases_with_unseen_values() -> None:
    """输入：`ood_only` 配置。输出：无。

    预期行为：只留「至少含一个训练侧没见过的取值」的 case；同分布侧则一个留出取值都没有。
    """
    ood = sp.build_cases(sp.MatchConfig(split="test_ood", ood_only=True))
    assert ood and all(case.unseen_dims() for case in ood)
    id_config = sp.MatchConfig(
        products=tuple(key for key in sp.PRODUCTS if key in sp.SEEN_PRODUCTS),
        spellings=tuple(pair for pair in sp.SPELLINGS if pair in sp.SEEN_SPELLINGS),
    )
    id_cases = sp.build_cases(id_config)
    assert id_cases and not any(case.unseen_dims() for case in id_cases)
    assert len(ood) < len(id_cases)


def test_unseen_dims_names_the_held_out_values() -> None:
    """输入：含留出取值的 case。输出：无。预期行为：点名是哪一维留出了（商品 / 写法对）。"""
    case = sp.MatchCase(sp.PRODUCTS[-1], sp.PATTERNS[0], sp.SPELLINGS[-1], "plain", "none")
    assert case.unseen_dims() == ("product", "spelling")
    seen = sp.MatchCase(sp.PRODUCTS[0], sp.PATTERNS[0], sp.SPELLINGS[0], "plain", "none")
    assert seen.unseen_dims() == ()


def test_material_and_query_id_are_deterministic() -> None:
    """输入：一个 case 与它的一个邻居。输出：无。

    预期行为：素材指纹是五维拼串、请求号由它派生（同 case 同号、换一维就换号），
    请求号里不带切分信息（切分名不参与派生）。
    """
    case = sp.MatchCase("lager", sp.PATTERNS[0], sp.SPELLINGS[0], "plain", "none")
    other = sp.MatchCase("lager", sp.PATTERNS[0], sp.SPELLINGS[0], "plain", "origin")
    assert sp.material_of(case) == sp.material_of(case)
    assert sp.material_of(case) != sp.material_of(other)
    assert sp.query_id_of(case) == sp.query_id_of(case)
    assert sp.query_id_of(case) != sp.query_id_of(other)
    assert sp.query_id_of(case).startswith(sp.QUERY_PREFIX)
    assert sp.query_id_of(case)[len(sp.QUERY_PREFIX) :].isdigit()


def test_sample_spec_carries_the_eight_answers() -> None:
    """输入：一条规格。输出：无。预期行为：八个答案都在规格里，且与规则函数逐一一致。"""
    case = sp.MatchCase("ipa", sp.PATTERNS[8], sp.SPELLINGS[0], "plain", "none")
    spec = sp.sample_spec(None, case, "train", 3, 1)
    pattern = spec.case.pattern
    assert spec.same == sp.same_of(pattern)
    assert spec.agree == sp.agree_of(pattern)
    assert spec.band == sp.band_of(spec.agree)
    assert spec.verdict == sp.verdict_of(pattern)
    assert spec.variant_count == sp.variant_count_of(pattern)
    assert spec.first_same() == (pattern[0] != sp.DIFF)
    assert spec.any_diff() == (spec.agree < len(sp.FIELDS))
    assert spec.split == "train" and spec.seq == 3 and spec.phrasing == 1


def test_sample_spec_ignores_the_rng() -> None:
    """输入：两个不同种子的 RNG。输出：无。预期行为：规格完全由 case 决定（同素材同题同答）。"""
    case = sp.MatchCase("stout", sp.PATTERNS[3], sp.SPELLINGS[1], "interleaved", "origin")
    assert sp.sample_spec(random.Random(1), case, "train", 0, 0) == sp.sample_spec(
        random.Random(99), case, "train", 0, 0
    )


def test_spec_to_dict_is_json_ready() -> None:
    """输入：一条规格。输出：无。预期行为：能直接 json 序列化，且摊平了五个维度。"""
    case = sp.MatchCase("lager", sp.PATTERNS[1], sp.SPELLINGS[1], "reversed", "origin")
    payload = sp.spec_to_dict(sp.sample_spec(None, case, "val", 1, 2))
    assert json.loads(json.dumps(payload)) == payload
    assert payload["split"] == "val" and payload["seq"] == 1
    assert len(payload["same"]) == len(sp.FIELDS)
    assert "reversed" in payload["case"]
    assert payload["unseen_dims"] == []


def test_case_describe_and_as_dict() -> None:
    """输入：一个 case。输出：无。预期行为：紧凑串用关系首字母，描述里带商品与五个维度。"""
    case = sp.MatchCase("stout", sp.PATTERNS[0], sp.SPELLINGS[0], "plain", "none")
    assert case.as_dict() == {
        "product": "stout",
        "pattern": "ddddd",
        "spelling": "0-1",
        "layout": "plain",
        "extra": "none",
    }
    described = case.describe()
    assert "stout" in described and "ddddd" in described and "plain" in described


def test_config_rejects_unknown_values() -> None:
    """输入：各类坏旋钮。输出：无。预期行为：构造期 ValueError，报错点名是哪一类。"""
    with pytest.raises(ValueError, match="不存在的商品"):
        sp.MatchConfig(products=("sake",))
    with pytest.raises(ValueError, match="不存在的显示顺序"):
        sp.MatchConfig(layouts=("zigzag",))
    with pytest.raises(ValueError, match="不存在的额外栏"):
        sp.MatchConfig(extras=("barcode",))
    with pytest.raises(ValueError, match="不合法的字段关系"):
        sp.MatchConfig(patterns=((sp.DIFF, sp.SAME),))
    with pytest.raises(ValueError, match="不合法的字段关系"):
        sp.MatchConfig(patterns=((sp.DIFF,) * 5 + ("maybe",),))
    with pytest.raises(ValueError, match="不合法的写法对"):
        sp.MatchConfig(spellings=((0, 0),))
    with pytest.raises(ValueError, match="不合法的写法对"):
        sp.MatchConfig(spellings=((0,),))
    with pytest.raises(ValueError, match="不合法的写法对"):
        sp.MatchConfig(spellings=((0, True),))


def test_config_defaults_are_legal() -> None:
    """输入：全默认参数。输出：无。预期行为：默认值合法，校验不误伤。"""
    config = sp.MatchConfig()
    assert config.products == sp.PRODUCTS
    assert config.patterns == sp.PATTERNS
    assert config.ood_only is False
    assert config.split == "train"


def test_layout_orders_are_permutations_of_the_five_fields() -> None:
    """输入：三种显示顺序。输出：无。

    预期行为：每种顺序都是五栏的一个排列（漏一栏 / 重一栏会让反解缺格或多格）。
    """
    assert set(sp.LAYOUT_ORDERS) == set(sp.LAYOUTS)
    for name, order in sp.LAYOUT_ORDERS.items():
        assert sorted(order) == list(range(len(sp.FIELDS))), name


def test_seen_value_sets_hold_out_the_last_entry() -> None:
    """输入：留出集合。输出：无。预期行为：只留最后一档，其余都算见过。"""
    assert sp.SEEN_PRODUCTS == frozenset(sp.PRODUCTS[:-1])
    assert sp.SEEN_SPELLINGS == frozenset(sp.SPELLINGS[:-1])
    assert sp.PRODUCTS[-1] not in sp.SEEN_PRODUCTS
    assert sp.SPELLINGS[-1] not in sp.SEEN_SPELLINGS
