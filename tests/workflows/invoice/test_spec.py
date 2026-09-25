"""真值规格的测试：维度组合、构造不变量、取值口径。"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from workflows.invoice import spec as sp


def test_percent_of_rounds_half_up() -> None:
    """输入：金额与税率。输出：税额。

    预期行为：四舍五入（half-up）——**不用 Python 的 `round`**，后者是银行家舍入
    （`round(4.5) == 4`），构造与判别器算同一个式子时必须口径一致。
    """
    assert sp.percent_of(150, 3) == 5
    assert round(4.5) == 4  # 对照：银行家舍入会给出 4
    assert sp.percent_of(100, 3) == 3
    assert sp.percent_of(0, 13) == 0


def test_compatible_only_drops_impossible_defect_counts() -> None:
    """输入：两个 case。输出：bool。预期行为：错处数超过行数 = 无解组合（2 行不可能错 3 处）。"""
    assert sp.compatible(sp.InvoiceCase("labeled_lines", "canonical", "cny", "dashed", 2, 2, True, "none"))
    assert not sp.compatible(
        sp.InvoiceCase("labeled_lines", "canonical", "cny", "dashed", 2, 3, True, "none")
    )


def test_build_cases_enumerates_every_combination() -> None:
    """输入：默认配置。输出：case 列表。

    预期行为：4×2×3×3×3×4×2×3 = 5184 组全枚举，剔掉无解的 432 条（「2 行却错 3 处」
    那类）后是 4752 条；同一配置两次枚举逐项相同（顺序稳定）。
    """
    cases = sp.build_cases(sp.InvoiceConfig())
    assert len(cases) == 4752
    assert cases == sp.build_cases(sp.InvoiceConfig())
    assert all(sp.compatible(case) for case in cases)


def test_seen_sets_are_proper_subsets_of_the_value_tables() -> None:
    """输入：无。输出：无。

    预期行为：每个留出维度都至少有一个「训练侧没见过」的取值——否则值留出是假的。
    """
    assert sp.SEEN_LAYOUTS < set(sp.LAYOUTS)
    assert sp.SEEN_LABEL_STYLES < set(sp.LABEL_STYLES)
    assert sp.SEEN_CURRENCIES < set(sp.CURRENCIES)
    assert sp.SEEN_DATE_FORMATS < set(sp.DATE_FORMATS)


def test_unseen_dims_names_the_holdout_dimensions() -> None:
    """输入：两个 case。输出：没见过的维度名。

    预期行为：全部取「见过」的取值时为空；含 `paragraph` / 别名 / `eur` / `dotted`
    时点名对应维度（4 个留出维度各一个没见过的取值）。
    """
    seen = sp.InvoiceCase("labeled_lines", "canonical", "cny", "dashed", 2, 0, True, "none")
    assert seen.unseen_dims() == ()
    fresh = sp.InvoiceCase("paragraph", "aliased", "eur", "dotted", 2, 0, True, "none")
    assert set(fresh.unseen_dims()) == {"layout", "labels", "currency", "date_format"}


def test_build_cases_ood_only_keeps_exactly_the_novel_combinations() -> None:
    """输入：`ood_only=True` 的配置。输出：case 列表。

    预期行为：每条都至少含一个没见过的取值，且总数 = 全部 - 只用训练侧取值枚举出来的那些。
    """
    all_cases = sp.build_cases(sp.InvoiceConfig())
    ood = sp.build_cases(sp.InvoiceConfig(split="test_ood", ood_only=True))
    assert all(case.unseen_dims() for case in ood)
    seen_only = [case for case in all_cases if not case.unseen_dims()]
    assert len(ood) == len(all_cases) - len(seen_only)
    assert len(ood) == 3960


def test_sample_spec_is_deterministic(make_spec: Any) -> None:
    """输入：两个种子。输出：规格。预期行为：同种子完全相同，不同种子不同。"""
    assert make_spec(7) == make_spec(7)
    assert make_spec(7) != make_spec(8)


@pytest.mark.parametrize("tax_state", sp.TAX_STATES)
def test_tax_state_decides_the_tax_amount(make_spec: Any, tax_state: str) -> None:
    """输入：税额行状态。输出：规格。

    预期行为：`none` 没有税额行（金额记 0）；`ok` 恰好等于「小计 × 税率」；
    `off` 一定对不上——否则那道「税额是否一致」的 noul 题答案就是错的。
    """
    spec = make_spec(3, tax_state=tax_state)
    assert spec.has_tax is (tax_state != "none")
    expected = sp.percent_of(spec.subtotal, spec.tax_rate)
    if tax_state == "none":
        assert spec.tax_amount == 0
    elif tax_state == "ok":
        assert spec.tax_amount == expected
        assert spec.tax_consistent is True
    else:
        assert spec.tax_amount != expected
        assert spec.tax_consistent is False


@pytest.mark.parametrize("subtotal_ok", (True, False))
def test_subtotal_flag_decides_the_subtotal(make_spec: Any, subtotal_ok: bool) -> None:
    """输入：小计正确与否。输出：规格。

    预期行为：`True` 时小计恰好等于明细金额之和，`False` 时一定不等（差 ≥ 3）。
    """
    spec = make_spec(4, subtotal_ok=subtotal_ok)
    written = sum(item.amount for item in spec.items)
    assert spec.subtotal_consistent is subtotal_ok
    assert (spec.subtotal == written) is subtotal_ok


@pytest.mark.parametrize("defects", sp.DEFECT_COUNTS)
def test_defect_count_matches_the_broken_lines(make_spec: Any, defects: int) -> None:
    """输入：注入的错处数。输出：规格。

    预期行为：恰好这么多行的「数量 × 单价 ≠ 金额」，且 `bad_lines` 与之一致。
    """
    spec = make_spec(6, lines=4, defects=defects)
    assert sum(1 for item in spec.items if not item.consistent) == defects
    assert spec.bad_lines == defects


def test_same_kind_values_are_pairwise_distinct(make_spec: Any) -> None:
    """输入：一批规格（覆盖四种版式 × 税额状态 × 小计状态）。输出：无。

    预期行为：同一份单据里单价两两不同、金额槽两两不同、日期两两不同、当事方互不相同——
    否则那道选择题要么有两个正确答案，要么两个选项的描述一模一样。
    """
    for layout in sp.LAYOUTS:
        for tax_state in sp.TAX_STATES:
            for subtotal_ok in (True, False):
                spec = make_spec(
                    9, layout=layout, tax_state=tax_state, subtotal_ok=subtotal_ok, lines=4, defects=2
                )
                units = [item.unit for item in spec.items]
                money = [spec.money_of(slot) for slot in spec.money_slots]
                dates = [spec.date_of(slot) for slot in spec.date_slots]
                assert len(set(units)) == len(units)
                assert len(set(money)) == len(money), (layout, tax_state, money)
                assert len(set(dates)) == len(dates)
                assert len(set(spec.parties)) == 3


def test_dates_are_ordered_and_ids_are_prefixed(make_spec: Any) -> None:
    """输入：一份规格。输出：无。

    预期行为：收件 < 开票 < 到期（构造时按 `timedelta` 偏移）；三种编号各有前缀，
    方便从文本一眼分辨（也方便审计）。
    """
    spec = make_spec(12)
    assert spec.received_date < spec.issue_date < spec.due_date
    assert spec.doc_no.startswith("INV-")
    assert spec.po_no.startswith("PO-")
    assert spec.contract_no.startswith("CT-")


def test_slot_lookups_reject_unknown_slots(make_spec: Any) -> None:
    """输入：未知槽名 / 没有税额行时问 `tax`。输出：KeyError。

    预期行为：一律报错而不是给个默认值——静默兜底会让「答案」悄悄变成另一个字段。
    """
    spec = make_spec(13, tax_state="none")
    for call, slot in (
        (spec.money_of, "tax"),
        (spec.money_of, "nope"),
        (spec.date_of, "nope"),
        (spec.party_of, "nope"),
        (spec.id_of, "nope"),
    ):
        with pytest.raises(KeyError):
            call(slot)


def test_money_slots_drop_tax_without_a_tax_row(make_spec: Any) -> None:
    """输入：有/没有税额行的规格。输出：可问的金额槽。

    预期行为：没有税额行时 `tax` 不在选项集合里（否则选项描述会缺失或重复）。
    """
    assert make_spec(14, tax_state="none").money_slots == ("subtotal", "shipping", "total")
    assert make_spec(14, tax_state="ok").money_slots == sp.MONEY_SLOTS


def test_spec_to_dict_is_serializable_and_carries_the_case(make_spec: Any) -> None:
    """输入：一份规格。输出：字典。预期行为：能直接 `json.dumps`（manifest 要用），
    并且带上 case 描述、每行的一致性结论。
    """
    spec = make_spec(15, tax_state="off", lines=4, defects=2)
    data = sp.spec_to_dict(spec)
    json.dumps(data)
    assert data["case"] == spec.case.describe()
    assert len(data["items"]) == spec.case.lines
    assert sum(1 for item in data["items"] if not item["ok"]) == 2
    assert data["tax_rate"] == spec.tax_rate


def test_case_describe_and_as_dict_cover_every_dimension() -> None:
    """输入：一个 case。输出：可读描述与维度字典。

    预期行为：描述里 8 个维度都能看出来；`as_dict` 的键顺序固定
    （切分的哈希分桶按它算，顺序一变切分就漂）。
    """
    case = sp.InvoiceCase("pipe_table", "aliased", "eur", "dotted", 3, 2, False, "off")
    text = case.describe()
    for token in ("pipe_table", "aliased", "eur", "dotted", "3行", "2处错", "小计错", "税额off"):
        assert token in text
    assert list(case.as_dict()) == [
        "layout",
        "labels",
        "currency",
        "date_format",
        "lines",
        "defects",
        "subtotal_ok",
        "tax_state",
    ]


class _StuckRandom(random.Random):
    """把指定区间的 `randint` 钉成同一个数（其余照常），用来逼构造器走「撞了要顺延」那条路。

    输入：`{区间: 固定值}` 映射与种子。输出：随机源。
    预期行为：命中映射就返回固定值，没命中就交给真随机——只影响被测的那个抽签口。
    """

    def __init__(self, fixed: dict[tuple[int, int], int], seed: int) -> None:
        """输入：固定值映射、种子。输出：实例。"""
        super().__init__(seed)
        self.fixed = fixed

    def randint(self, a: int, b: int) -> int:
        """输入：区间。输出：固定值（命中时）或真随机数。"""
        return self.fixed.get((a, b), super().randint(a, b))


def test_repeated_unit_prices_are_shifted_apart(make_case: Any) -> None:
    """输入：一个「单价永远抽到同一个数」的随机源。输出：规格。

    预期行为：撞车时顺延（500 → 501 → 502）——单价是 `line_field` 的选项描述，
    两个一样的单价就是两个正确答案。
    """
    rng = _StuckRandom({(120, 980): 500}, 1)
    spec = sp.sample_spec(rng, make_case(lines=3, defects=0), "train", 0, 0)
    assert [item.unit for item in spec.items] == [500, 501, 502]


def test_shipping_is_shifted_off_a_collision(monkeypatch: Any, make_case: Any) -> None:
    """输入：把税率函数钉成常数、运费第一次抽到同一个数。输出：规格。

    预期行为：运费撞上别的金额时顺延——否则那道金额题的选项描述会重复，
    没有税额行时「总计」与「小计」也会变成同一个数。
    """
    monkeypatch.setattr(sp, "percent_of", lambda value, rate: 50)
    rng = _StuckRandom({(12, 98): 50}, 3)
    spec = sp.sample_spec(rng, make_case(lines=2, defects=0, tax_state="ok"), "train", 0, 0)
    assert spec.tax_amount == 50
    assert spec.shipping == 51
    assert len({spec.subtotal, spec.shipping, spec.tax_amount, spec.total}) == 4
