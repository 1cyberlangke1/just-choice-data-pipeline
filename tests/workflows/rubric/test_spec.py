"""真值规格的单元测试：规则函数、三张判据表、十八张判据表的构造、case 校验与单号。"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from tests.workflows.rubric.conftest import _make_case, _make_spec
from workflows.rubric import spec as sp


def _row(criterion: str, need: str, count: str) -> sp.RubricRow:
    """输入：判据键、门槛键、观测键。输出：一行（权重从表里取）。"""
    return sp.RubricRow(criterion, sp.weight_of(criterion), need, count)


def test_met_of_is_a_threshold() -> None:
    """输入：三档门槛 × 五档观测。输出：无。

    预期行为：**观测不低于要求**才算达标；比的是数值不是字符串——「观测 10」这种写法在
    这里根本不合法，所以字符串比较的坑不存在。认不出的键报人话。
    """
    for need in sp.NEEDS:
        for count in sp.COUNTS:
            want = sp.COUNT_VALUE[count] >= sp.NEED_VALUE[need]
            assert sp.met_of(need, count) is want, (need, count)
    assert sp.met_of("need2", "n2") is True  # 压线算达标
    assert sp.met_of("need2", "n1") is False
    with pytest.raises(KeyError, match="need"):
        sp.met_of("nope", "n2")
    with pytest.raises(KeyError, match="count"):
        sp.met_of("need2", "nope")


def test_count_of_tight_and_roomy() -> None:
    """输入：三档门槛 × 两档松紧 × 达标与否。输出：无。

    预期行为：达标时 `tight` 刚好压线（观测 = 要求）、`roomy` 高出一档（要求 + 1）；
    没达标时两档都差一档（要求 − 1）。于是「观测 = 要求」**不等价于**达标——
    看等号的捷径被破掉，考生必须真的比大小。认不出的键报人话。
    """
    for need in sp.NEEDS:
        base = sp.NEED_VALUE[need]
        assert sp.count_of(need, True, "tight") == f"n{base}"
        assert sp.count_of(need, True, "roomy") == f"n{base + 1}"
        assert sp.count_of(need, False, "tight") == f"n{base - 1}"
        assert sp.count_of(need, False, "roomy") == f"n{base - 1}"
        # 反查一遍：推出来的观测格必须真的落在「达标 / 没达标」那一侧。
        assert sp.met_of(need, sp.count_of(need, True, "roomy")) is True
        assert sp.met_of(need, sp.count_of(need, False, "roomy")) is False
    with pytest.raises(KeyError, match="need"):
        sp.count_of("nope", True, "tight")
    with pytest.raises(KeyError, match="slack"):
        sp.count_of("need1", True, "nope")


def test_count_of_never_leaves_the_range() -> None:
    """输入：全部门槛 × 松紧 × 达标与否。输出：无。

    预期行为：观测永远落在 0..4 之内——`need1` 没达标算 0、`need3` 达标且宽裕算 4，
    两个边界都在表里；出界说明松紧的定义被改坏了。
    """
    for need in sp.NEEDS:
        for met in (True, False):
            for slack in sp.SLACKS:
                assert sp.count_of(need, met, slack) in sp.COUNTS


def test_count_of_guards_a_broken_threshold_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：把门槛表里的一档改到观测范围之外。输出：无。

    预期行为：当场报人话，而不是产出一个「观测 10」这种评分单里根本不存在的格子——
    这条守的是「改表忘了改松紧定义」。
    """
    monkeypatch.setitem(sp.NEED_VALUE, "need3", 9)
    with pytest.raises(ValueError, match="观测算出界了"):
        sp.count_of("need3", True, "roomy")


def test_is_redline_only_when_the_redline_criterion_is_unmet() -> None:
    """输入：六条判据 × 达标与否。输出：无。

    预期行为：只有 `original` **没达标**才算踩红线；别的判据怎么都没事，`original`
    达标了也没事。认不出的判据报人话。
    """
    for criterion in sp.CRITERIA:
        for met in (True, False):
            want = criterion == sp.REDLINE and not met
            assert sp.is_redline(criterion, met) is want, (criterion, met)
    assert sp.is_redline(sp.REDLINE, False) is True
    with pytest.raises(KeyError, match="criterion"):
        sp.is_redline("nope", False)


def test_total_of_sums_the_weights_of_the_met_rows() -> None:
    """输入：一张判据表。输出：无。

    预期行为：总分只由**达标**那几条的权重相加；同一张表换个排布总分不变（排布只换行序）。
    """
    for sheet in sp.SHEETS:
        met = frozenset(sp.CRITERIA[i] for i in sp.MET_INDEX[sheet])
        want = sum(sp.WEIGHT_VALUE[sp.WEIGHT_OF[c]] for c in met)
        assert sp.total_of(met) == want
        for layout in sp.LAYOUTS:
            rows = sp.rows_of(sheet, layout, "tight")
            assert sp.total_of(sp.met_criteria(rows)) == want
    with pytest.raises(KeyError, match="criterion"):
        sp.total_of(["nope"])


def test_band_of_splits_on_the_two_thresholds() -> None:
    """输入：总分 0..12。输出：无。

    预期行为：≥ 7 高档、= 6 中档、≤ 5 低档；低档是地板（再低也只有低档）。
    """
    for total in range(0, 13):
        want = "b3" if total >= sp.BAND_MIN["b3"] else ("b2" if total >= sp.BAND_MIN["b2"] else "b1")
        assert sp.band_of(total) == want, total
    assert sp.band_of(-5) == "b1"


def test_gate_table_covers_every_cell() -> None:
    """输入：三档 × 红线与否六个格子。输出：无。

    预期行为：六个格子都有处置、每个处置各占两格（于是 `gate` 的分布是构造均匀的）；
    认不出的档位报人话，红线不是布尔报类型错。
    """
    assert set(sp.GATE_TABLE) == {(band, red) for band in sp.BANDS for red in (True, False)}
    counts = Counter(sp.GATE_TABLE.values())
    assert set(counts) == set(sp.GATES)
    assert set(counts.values()) == {2}
    for (band, red), gate in sp.GATE_TABLE.items():
        assert sp.gate_of(band, red) == gate
    with pytest.raises(KeyError, match="band"):
        sp.gate_of("nope", False)
    with pytest.raises(TypeError, match="红线"):
        sp.gate_of("b1", 1)


def test_sheets_are_three_subsets_minus_two_disjoint_ones() -> None:
    """输入：十八张判据表。输出：无。

    预期行为：每张恰好三条达标；被去掉的两张**互不相交**（于是每条判据恰好被去掉一次）；
    每条判据达标九次——`met` 的 50/50 就是这条的推论。
    """
    assert len(sp.SHEETS) == 18
    assert len(set(sp.SHEETS)) == 18
    assert all(len(sp.MET_INDEX[sheet]) == 3 for sheet in sp.SHEETS)
    dropped = [frozenset(t) for t in sp.DROPPED]
    assert len(dropped) == 2
    assert dropped[0] & dropped[1] == frozenset()
    assert dropped[0] | dropped[1] == frozenset(range(len(sp.CRITERIA)))
    counts = Counter()
    for sheet in sp.SHEETS:
        for index in sp.MET_INDEX[sheet]:
            counts[sp.CRITERIA[index]] += 1
    assert set(counts) == set(sp.CRITERIA)
    assert set(counts.values()) == {9}


def test_bands_and_gates_are_balanced_by_construction() -> None:
    """输入：十八张判据表。输出：无。

    预期行为：三档各六张、红线在三档里各三张（六个格子一样大）、三个处置各六张——
    `band` 与 `gate` 的均衡是**构造**出来的，不靠事后过滤。
    """
    bands = Counter()
    cells = Counter()
    for sheet in sp.SHEETS:
        met = frozenset(sp.CRITERIA[i] for i in sp.MET_INDEX[sheet])
        band = sp.band_of(sp.total_of(met))
        redline = sp.is_redline(sp.REDLINE, sp.REDLINE in met)
        bands[band] += 1
        cells[(band, redline)] += 1
    assert set(bands.values()) == {6}
    assert set(cells.values()) == {3}
    assert set(Counter(sp.gate_of(b, r) for (b, r) in cells.elements()).values()) == {6}


def test_the_redline_criterion_has_weight_two() -> None:
    """输入：六条判据各自的权重。输出：无。

    预期行为：红线必须是**权重 2** 的那条：红线在三档里的达标次数要各占一半，只有权重 2
    的判据做得到（权重 3 是 5 / 3 / 1、权重 1 是 1 / 3 / 5）——换掉它这一条会当场红。
    """
    assert sp.WEIGHT_OF[sp.REDLINE] == "w2"
    for criterion in sp.CRITERIA:
        per_band = Counter()
        for sheet in sp.SHEETS:
            met = frozenset(sp.CRITERIA[i] for i in sp.MET_INDEX[sheet])
            per_band[sp.band_of(sp.total_of(met))] += 1 if criterion in met else 0
        even = set(per_band.values()) == {3}
        assert even is (criterion == sp.REDLINE or sp.WEIGHT_OF[criterion] == "w2"), criterion


def test_layouts_are_a_latin_square() -> None:
    """输入：六种排布。输出：无。

    预期行为：每条判据在六个位置上各出现一次——这正是 `top` 答案位置均匀的原因，
    破了它答案位置就会偏（均衡闸当场标红）。
    """
    seen: dict[str, dict[int, int]] = {criterion: {} for criterion in sp.CRITERIA}
    for layout in sp.LAYOUTS:
        placed = [sp.criterion_at(layout, slot) for slot in range(1, len(sp.CRITERIA) + 1)]
        assert sorted(placed) == sorted(sp.CRITERIA)
        for position, criterion in enumerate(placed, start=1):
            seen[criterion][position] = seen[criterion].get(position, 0) + 1
    for criterion, counts in seen.items():
        assert counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, (criterion, counts)


def test_slot_of_is_the_inverse_of_criterion_at() -> None:
    """输入：六种排布 × 六条判据。输出：无。预期行为：互为反函数；位置越界报人话。"""
    for layout in sp.LAYOUTS:
        for slot in range(1, len(sp.CRITERIA) + 1):
            assert sp.slot_of(layout, sp.criterion_at(layout, slot)) == slot
    with pytest.raises(ValueError, match="位置"):
        sp.criterion_at("l0", 7)
    with pytest.raises(KeyError, match="criterion"):
        sp.slot_of("l0", "nope")


def test_rows_are_the_sheet_in_layout_order() -> None:
    """输入：十八张判据表 × 六种排布 × 两档松紧。输出：无。

    预期行为：六行、位置顺序稳定；达标的那几条与判据表一致（松紧只换观测那一格，
    不换达标结果）。
    """
    for sheet in sp.SHEETS:
        met = frozenset(sp.CRITERIA[i] for i in sp.MET_INDEX[sheet])
        for layout in sp.LAYOUTS:
            for slack in sp.SLACKS:
                rows = sp.rows_of(sheet, layout, slack)
                assert len(rows) == len(sp.CRITERIA)
                assert [row.criterion for row in rows] == [
                    sp.criterion_at(layout, slot) for slot in range(1, len(sp.CRITERIA) + 1)
                ]
                assert {row.criterion for row in rows if row.met()} == met
                for row in rows:
                    assert row.weight == sp.WEIGHT_OF[row.criterion]
                    assert row.need == sp.NEED_OF[row.criterion]
                    assert row.count == sp.count_of(row.need, row.criterion in met, slack)
    with pytest.raises(KeyError, match="sheet"):
        sp.rows_of("nope", "l0", "tight")


def test_top_slot_picks_the_heaviest_unmet_row() -> None:
    """输入：一行权重 3 没达标、一行权重 2 没达标、一行权重 1 没达标。输出：无。

    预期行为：最严重 = 没达标的判据里权重最大的那条（这里权重 3 只有一条 → 唯一）。
    """
    rows = (
        _row("example", "need2", "n2"),  # 达标（w3）
        _row("source", "need3", "n2"),  # 没达标（w3）
        _row("original", "need2", "n1"),  # 没达标（w2）
        _row("sections", "need2", "n2"),  # 达标（w2）
        _row("typos", "need1", "n0"),  # 没达标（w1）
        _row("length", "need2", "n2"),  # 达标（w1）
    )
    assert sp.top_slot(rows) == 2
    assert sp.met_criteria(rows) == ("example", "sections", "length")


def test_top_slot_is_none_when_the_worst_is_tied() -> None:
    """输入：两条权重 3 都没达标。输出：无。

    预期行为：并列时「最严重的是哪一条」没有唯一解，给 None（`top` 一族因此不出题）；
    一条都没达标时也一样（没有唯一解，而不是随便挑一条）。
    """
    tied = (
        _row("example", "need2", "n1"),  # 没达标（w3）
        _row("source", "need3", "n1"),  # 没达标（w3）
        _row("original", "need2", "n2"),  # 达标（w2）
        _row("sections", "need2", "n2"),
        _row("typos", "need1", "n1"),
        _row("length", "need2", "n2"),
    )
    assert sp.top_slot(tied) is None
    assert sp.top_slot((_row("typos", "need1", "n0"),)) == 1  # 只有一条没达标时唯一
    assert sp.top_slot((_row("typos", "need1", "n2"),)) is None  # 全达标时没有「最严重」


def test_unknown_keys_report_the_field() -> None:
    """输入：各规则函数喂坏键。输出：无。预期行为：都报人话并点名字段。"""
    with pytest.raises(KeyError, match="criterion"):
        sp.weight_of("nope")
    with pytest.raises(KeyError, match="criterion"):
        sp.need_of("nope")
    with pytest.raises(KeyError, match="layout"):
        sp.layout_of("nope")


def test_case_dict_and_describe() -> None:
    """输入：一个 case。输出：无。预期行为：四维摊平；描述与素材指纹同形。"""
    case = _make_case()
    assert case.as_dict() == {
        "sheet": "s0",
        "layout": "l0",
        "slack": "tight",
        "phrasing": 0,
    }
    assert case.describe() == sp.material_of(case) == "s0|l0|tight|0"


@pytest.mark.parametrize(
    ("layout", "phrasing", "want"),
    [
        ("l0", 0, ()),
        ("l4", 0, ()),
        ("l5", 0, ("layout",)),
        ("l0", 1, ("phrasing",)),
        ("l5", 1, ("layout", "phrasing")),
    ],
)
def test_unseen_dims(layout: str, phrasing: int, want: tuple[str, ...]) -> None:
    """输入：各排布 × 两套句面。输出：无。预期行为：只认排布 `l5` 与第二套句面这两个留出维。"""
    assert _make_case(layout=layout, phrasing=phrasing).unseen_dims() == want


def test_bad_case_values_report_the_field() -> None:
    """输入：四个维度各喂一个坏值。输出：无。预期行为：都报人话并点名字段。"""
    with pytest.raises(KeyError, match="sheet"):
        _make_case(sheet="nope")
    with pytest.raises(KeyError, match="layout"):
        _make_case(layout="nope")
    with pytest.raises(KeyError, match="slack"):
        _make_case(slack="nope")
    with pytest.raises(KeyError, match="句面"):
        _make_case(phrasing=2)


def test_request_id_is_deterministic_and_leaks_no_split() -> None:
    """输入：同一个 case 的两次取号 / 换个切分的取号。输出：无。

    预期行为：同一素材必得同一单号；单号只看素材指纹，**与切分无关**（否则同一条样本
    在不同切分里拿到不同单号，等于把切分信息写进 state）。格式是 `G-` + 五位数字。
    """
    case = _make_case()
    first = sp.request_id_of(case)
    assert first == sp.request_id_of(case)
    assert first.startswith(sp.REQUEST_PREFIX)
    assert first[len(sp.REQUEST_PREFIX) :].isdigit()
    assert 10000 <= int(first[len(sp.REQUEST_PREFIX) :]) < 100000
    spec_a = sp.sample_spec(case, "train", 0, 0, None)
    spec_b = sp.sample_spec(case, "test_ood", 7, 1, None)
    assert spec_a.request_id == spec_b.request_id == first


def test_sample_spec_carries_the_derived_truth() -> None:
    """输入：全开网格。输出：无。

    预期行为：达标集合、总分、档位、红线、处置、最严重的位置都由规则函数算出来，
    与 `rows_of` 里的行对得上——规格里没有一处是掷骰子来的。
    """
    for case in sp.build_cases(sp.RubricConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        met = sp.met_criteria(spec.rows)
        assert spec.met == met
        assert spec.total == sp.total_of(met)
        assert spec.band == sp.band_of(spec.total)
        assert spec.redline is sp.is_redline(sp.REDLINE, sp.REDLINE in met)
        assert spec.gate == sp.gate_of(spec.band, spec.redline)
        assert spec.top_slot == sp.top_slot(spec.rows)


def test_spec_to_dict_is_json_shaped() -> None:
    """输入：一份规格。输出：无。预期行为：摘要能被 json 序列化，键齐全。"""
    spec = _make_spec()
    payload = sp.spec_to_dict(spec)
    assert set(payload) == {
        "case",
        "unseen_dims",
        "request_id",
        "met",
        "total",
        "band",
        "redline",
        "gate",
        "top_slot",
        "rows",
    }
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert len(payload["rows"]) == len(sp.CRITERIA)


def test_row_helpers() -> None:
    """输入：一行判据。输出：无。预期行为：四格摊平、描述可读、达标只吃要求与观测。"""
    row = _row("example", "need2", "n3")
    assert row.as_dict() == {
        "criterion": "example",
        "weight": "w3",
        "need": "need2",
        "count": "n3",
    }
    assert row.describe() == "example/w3/need2/n3"
    assert row.met() is True
    assert _row("example", "need2", "n1").met() is False


def test_build_cases_enumerates_the_whole_grid() -> None:
    """输入：全默认配置。输出：无。

    预期行为：18 张判据表 × 6 排布 × 2 松紧 × 2 句面 = 432 格，顺序稳定、无重复。
    """
    cases = sp.build_cases(sp.RubricConfig())
    assert len(cases) == 18 * 6 * 2 * 2 == 432
    assert len({sp.material_of(case) for case in cases}) == len(cases)
    assert cases == sp.build_cases(sp.RubricConfig())


def test_ood_only_keeps_exactly_the_cases_with_an_unseen_value() -> None:
    """输入：`ood_only` 配置。输出：无。

    预期行为：留下的每一条都至少含一个留出取值，滤掉的正是不含留出取值的那些；
    432 − 18 × 5 × 2 × 1 = 252。
    """
    kept = sp.build_cases(sp.RubricConfig(ood_only=True))
    assert kept and all(case.unseen_dims() for case in kept)
    assert len(kept) == 432 - 18 * 5 * 2 * 1 == 252
    id_materials = {
        sp.material_of(case)
        for case in sp.build_cases(sp.RubricConfig())
        if not case.unseen_dims()
    }
    assert not id_materials & {sp.material_of(case) for case in kept}


def test_config_reports_empty_pools() -> None:
    """输入：四张取值表各清空一次。输出：无。预期行为：构造期报人话并点名字段。"""
    for field in ("sheets", "layouts", "slacks", "phrasings"):
        with pytest.raises(ValueError, match=field):
            sp.RubricConfig(**{field: ()})
