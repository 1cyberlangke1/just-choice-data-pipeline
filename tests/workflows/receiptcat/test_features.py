"""receiptcat 特征层的单元测试：金额、档位、关键词表、诱饵与两条标签路径。

输入：`workflows.receiptcat.features` 的公开函数与两张语言表。
输出：无（断言）。预期行为：手写的常量表（金额 / 上限 / 关键词 / 诱饵）必须真的自洽——
它们不是算出来的，写歪一条不会有任何东西报错，只会让标签悄悄错掉。
"""

from __future__ import annotations

import pytest

from workflows.receiptcat import features, locales

#: 两种语言，参数化用。
LANGS: tuple[str, ...] = ("zh", "en")


def test_amounts_cover_every_pair() -> None:
    """输入：无。输出：无。预期行为：每个 (类别, 档位) 都有金额，且一个不多一个不少。"""
    assert set(features.AMOUNTS) == {
        (cid, band) for cid in features.CATEGORY_IDS for band in features.BANDS
    }
    assert len(features.AMOUNT_VALUES) == len(set(features.AMOUNT_VALUES))


@pytest.mark.parametrize("cid", features.CATEGORY_IDS)
@pytest.mark.parametrize("band", features.BANDS)
def test_amount_really_lands_in_its_band(cid: str, band: str) -> None:
    """输入：一个 (类别, 档位)。输出：无。

    预期行为：那条手写金额按 `band_of` 算出来必须正好是它标的那一档——手写金额最怕的就是
    「看着像超限、其实没超」，而这种错在产物里完全看不出来。
    """
    cents = features.cents_of(features.AMOUNTS[(cid, band)])
    assert cents is not None
    assert features.band_of(cents, features.CAPS_CENTS[cid]) == band


def test_no_amount_is_a_substring_of_another() -> None:
    """输入：无。输出：无。

    预期行为：12 个金额两两互不为子串。互为子串时「正文里出现了哪个金额」就有歧义，
    `amounts_in` 会同时认出两个，档位推不出来。
    """
    for left in features.AMOUNT_VALUES:
        for right in features.AMOUNT_VALUES:
            if left != right:
                assert left not in right, f"{left} 是 {right} 的子串"


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("168.00", 16800),
        ("0.05", 5),
        ("96.50", 9650),
        ("168.0", None),
        ("168", None),
        ("168.001", None),
        ("a168.00", None),
        ("-168.00", None),
    ],
)
def test_cents_of_only_accepts_two_decimals(text: str, expect: int | None) -> None:
    """输入：几种金额写法。输出：无。预期行为：只认「整数 + 两位小数」，别的给 None。"""
    assert features.cents_of(text) == expect


@pytest.mark.parametrize(
    ("amount", "cap", "expect"),
    [
        (19999, 20000, "under"),
        (20000, 20000, "mild"),
        (39999, 20000, "mild"),
        (40000, 20000, "heavy"),
        (0, 20000, "under"),
    ],
)
def test_band_boundaries(amount: int, cap: int, expect: str) -> None:
    """输入：金额与上限（分）。输出：无。

    预期行为：**恰好等于上限算超限**、**恰好两倍算两倍以上**——档位表那三句话就是这两个
    边界，`mild` 是左闭右开区间 `[上限, 两倍上限)`。
    """
    assert features.band_of(amount, cap) == expect


@pytest.mark.parametrize(
    ("band", "category", "expect"),
    [
        ("under", "C1", "auto"),
        ("under", "C4", "auto"),
        ("mild", "C1", "receipt"),
        ("mild", "C4", "receipt"),
        ("heavy", "C1", "manager"),
        ("heavy", "C2", "manager"),
        ("heavy", "C3", "manager"),
        ("heavy", "C4", "finance"),
    ],
)
def test_tier_rules(band: str, category: str, expect: str) -> None:
    """输入：档位与类别。输出：无。

    预期行为：只有**软件订阅 + 两倍以上**走财务复核，其余三档与类别无关——这一格交互是
    `tier` 那一族不跟 `over` 重复的原因，所以它得被钉死。
    """
    assert features.tier_of(band, category) == expect


@pytest.mark.parametrize("lang", LANGS)
def test_keyword_table_is_disjoint(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四个类别的关键词两两互不包含（也没有跨类别重复）——关键词表是**按字面包含**
    判命中的，两个类共用一个词时「命中哪一类」就没有唯一答案了。
    """
    words = [word for cid in features.CATEGORY_IDS for word in features.keyword_table(locales.load(lang))[cid]]
    for left in words:
        for right in words:
            if left != right:
                assert left not in right, f"{lang}: {left} 是 {right} 的子串"


@pytest.mark.parametrize("lang", LANGS)
def test_decoys_are_not_keywords_and_not_category_names(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：诱饵词不含任何关键词、也不含任何类别名——诱饵的全部意义就是「看着像、按字面
    不算」，只要它蹭到关键词或类别名，那一族题（以及 `near` 的答案）当场失真。
    """
    loc = locales.load(lang)
    keywords = [word for cid in features.CATEGORY_IDS for word in features.keyword_table(loc)[cid]]
    decoys = [word for cid in features.CATEGORY_IDS for word in features.decoy_table(loc)[cid]]
    names = [str(loc["category"][cid]["name"]) for cid in features.CATEGORY_IDS]
    assert len(set(decoys)) == len(decoys)
    for word in decoys:
        assert not any(keyword in word for keyword in keywords), f"{lang}: 诱饵 {word} 蹭到了关键词"
        assert not any(name in word for name in names), f"{lang}: 诱饵 {word} 蹭到了类别名"


@pytest.mark.parametrize("lang", LANGS)
def test_category_names_are_not_keywords(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：类别名本身不能是关键词，否则「不许出现类别名」
    这条契约与「必须出现关键词」会互相打架。"""
    loc = locales.load(lang)
    keywords = [word for cid in features.CATEGORY_IDS for word in features.keyword_table(loc)[cid]]
    for cid in features.CATEGORY_IDS:
        assert str(loc["category"][cid]["name"]) not in keywords


def test_categories_hit_is_literal() -> None:
    """输入：含关键词 / 诱饵 / 别的词的几段文本。输出：无。

    预期行为：只有**原样出现**的关键词才算命中；诱饵词、类别名、半个关键词都不算。
    """
    loc = locales.load("zh")
    table = features.keyword_table(loc)
    assert features.categories_hit("打车 报销明细 168.00", table) == ("C1",)
    assert features.categories_hit("打车 咖啡 168.00", table) == ("C1", "C2")
    assert features.categories_hit("顺风车 报销明细 168.00", table) == ()
    assert features.categories_hit("交通出行 168.00", table) == ()
    assert features.categories_hit("打 168.00", table) == ()


def test_decoy_hit_ignores_keywords() -> None:
    """输入：带诱饵与不带诱饵的文本。输出：无。预期行为：诱饵命中与关键词命中互不干涉。"""
    loc = locales.load("zh")
    assert features.decoy_hit("顺风车 168.00", features.decoy_table(loc))
    assert not features.decoy_hit("打车 168.00", features.decoy_table(loc))


def test_observed_labels_needs_exactly_one_category_and_one_amount() -> None:
    """输入：几种收据行。输出：无。

    预期行为：命中类别必须恰好一个、注入金额必须恰好一个，否则一律认不出来（None）——
    认不出来会被审计记成 content_bad，而不是拿一个猜出来的标签去比。
    """
    loc = locales.load("zh")
    good = features.observed_labels("打车 报销明细 168.00", loc)
    assert good is not None
    assert (good.category, good.band, good.decoy) == ("C1", "under", "none")
    assert features.observed_labels("报销明细 168.00", loc) is None
    assert features.observed_labels("打车 咖啡 168.00", loc) is None
    assert features.observed_labels("打车 报销明细 168.00 254.00", loc) is None
    assert features.observed_labels("打车 报销明细 12.34", loc) is None
    near = features.observed_labels("打车 顺风车 报销明细 168.00", loc)
    assert near is not None and near.decoy == features.DECOY_NEAR


def test_labels_projection() -> None:
    """输入：三种 case。输出：无。预期行为：`over` / `tier` / `near` 三个投影都对。"""
    under = features.label_for(features.ReceiptCase("C1", "under", "none"))
    assert (under.over, under.tier, under.near) == (False, "auto", False)
    heavy = features.label_for(features.ReceiptCase("C4", "heavy", "near"))
    assert (heavy.over, heavy.tier, heavy.near) == (True, "finance", True)
    mild = features.label_for(features.ReceiptCase("C3", "mild", "none"))
    assert (mild.over, mild.tier, mild.near) == (True, "receipt", False)


def test_build_cases_enumerates_the_whole_space() -> None:
    """输入：无。输出：无。预期行为：全枚举 4×3×2 = 24 组，顺序稳定、不重不漏。"""
    cases = features.build_cases(512)
    assert len(cases) == 24
    assert len({case.fingerprint() for case in cases}) == 24
    assert cases == features.build_cases(512)


def test_build_cases_respects_a_restricted_dimension() -> None:
    """输入：只留一个诱饵取值的维度表。输出：无。预期行为：枚举只出那一档（OOD 侧的形状）。"""
    cases = features.build_cases(512, dims={**features.DIMS, "decoy": (features.DECOY_NEAR,)})
    assert {case.decoy for case in cases} == {features.DECOY_NEAR}
    assert len(cases) == 12


@pytest.mark.parametrize("cid", features.CATEGORY_IDS)
def test_distractors_are_the_other_categories(cid: str) -> None:
    """输入：一个类别。输出：无。

    预期行为：干扰类别正好是另外三类、按 `CATEGORY_IDS` 顺序、不含真值——四选三的题里干扰集
    恒定，所以它是**算**出来的，不是抽出来的（抽样池只有三个元素，抽三次必然全抽走）。
    """
    case = features.ReceiptCase(cid, "mild", "none")
    others = tuple(item for item in features.CATEGORY_IDS if item != cid)
    assert case.distractors == others
    assert len(set(case.distractors)) == 3
    assert case.category not in case.distractors
