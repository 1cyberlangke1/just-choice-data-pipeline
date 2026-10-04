"""真值规格的单元测试：规则函数、二十四张局面的构造、只吃键的重算、case 校验与单号。"""

from __future__ import annotations

from collections import Counter
from itertools import product

import pytest

from tests.workflows.sim.conftest import _make_case
from workflows.sim import spec as sp

#: 冷却组里**哪些动作处在冷却中**——与 `spec.CD_SETS` 是同一份数据的另一种写法，
#: 拿它当独立参照比「照抄一遍公式」有意义。
COOLED: dict[str, frozenset[int]] = {
    "c0": frozenset(),
    "c1": frozenset({1, 2, 4, 5}),
    "c2": frozenset({1, 2, 3, 5}),
}


def test_is_legal_needs_both_resources_and_no_cooldown() -> None:
    """输入：全部局面 × 六个动作。输出：无。

    预期行为：**资源不低于耗、且冷却为 0** 才合法；两个条件各单独喂一次坏值都要变成不合法。
    """
    for board in (sp.board_of(name) for name in sp.BOARDS):
        resources, cds, _turn, _haul = board
        for index, action in enumerate(sp.ACTIONS):
            want = resources >= sp.COST_VALUE[action] and index not in COOLED[cds]
            assert sp.is_legal(board, index) is want, (board, action)
    # 两个条件各单独反例：资源够但冷却中 / 冷却好但资源不够。
    assert sp.is_legal((4, "c1", 3, 4), 1) is False  # 探测：资源够，冷却 1
    assert sp.is_legal((0, "c0", 3, 4), 0) is False  # 前进：冷却 0，资源不够
    assert sp.is_legal((4, "c1", 3, 4), 0) is True


def test_is_done_is_a_threshold_on_haul_or_turn() -> None:
    """输入：回合 3..5 × 战利品 4..6。输出：无。

    预期行为：**战利品到 6、或者回合到 5** 就结束；两个信号各单独成立一次都算结束。
    """
    for turn, haul in product(sp.TURNS, sp.HAULS):
        board = (2, "c0", turn, haul)
        assert sp.is_done(board) is (haul >= sp.TARGET or turn >= sp.ROUND_CAP)
    assert sp.is_done((0, "c0", 5, 4)) is True  # 回合到顶
    assert sp.is_done((0, "c0", 3, 6)) is True  # 战利品到顶
    assert sp.is_done((0, "c0", 4, 5)) is False  # 两边都差一点


def test_best_index_picks_the_largest_gain_among_the_usable() -> None:
    """输入：全部局面。输出：无。

    预期行为：能用的动作里收益最大的那个；一个能用的都没有给 None。收益互不相同，
    所以答案唯一——这条也是 `best` / `cost` 两族「出不出」的判据。
    """
    for board in (sp.board_of(name) for name in sp.BOARDS):
        usable = [i for i in range(len(sp.ACTIONS)) if sp.is_legal(board, i)]
        got = sp.best_index(board)
        if not usable:
            assert got is None
            assert sp.cost_level(board) is None
            continue
        assert got in usable
        assert sp.GAIN_VALUE[sp.ACTIONS[got]] == max(
            sp.GAIN_VALUE[sp.ACTIONS[i]] for i in usable
        )
        assert sp.cost_level(board) == sp.COST_VALUE[sp.ACTIONS[got]]


def test_best_index_is_none_when_nothing_is_usable() -> None:
    """输入：资源 0、冷却组又让仅剩的动作进了冷却。输出：无。

    预期行为：一个能用的都没有时给 None（调用方据此不出 `best` / `cost`）——这条也是
    「没结束但卡住」那种局面的答案形态。
    """
    board = (0, "c1", 3, 4)
    assert not any(sp.is_legal(board, index) for index in range(len(sp.ACTIONS)))
    assert sp.best_index(board) is None
    assert sp.cost_level(board) is None


def test_gains_are_pairwise_distinct() -> None:
    """输入：收益表。输出：无。

    预期行为：六个收益恰好铺满 0..5 六档——`gain` 一族的答案空间因此是六档，
    而且「该用哪个」永远有唯一解（并列时也不需要真的走并列规则）。
    """
    assert sorted(sp.GAIN_VALUE.values()) == list(range(len(sp.GAINS)))
    assert set(sp.GAIN_VALUE) == set(sp.ACTIONS)
    assert sp.gain_levels(sp.board_of("b0")) == tuple(
        sp.GAIN_VALUE[action] for action in sp.ACTIONS
    )


def test_the_twenty_four_boards_are_balanced_by_construction() -> None:
    """输入：二十四张局面。输出：无。

    预期行为：四族的分布都是**构造出来**的，不靠事后过滤——合法动作数合计恰好 72
    （占 144 的一半）、结束的恰好 12 张、没结束的 12 张都有合法动作、该用的那个动作的耗
    四档各 3 次。逐位置看还各占一半，`best` 的位置与 `gain` 的档位都被拉丁方铺匀。
    """
    boards = [sp.board_of(name) for name in sp.BOARDS]
    assert len(boards) == 24 and len(set(boards)) == 24
    legal = sum(sum(sp.is_legal(board, i) for i in range(len(sp.ACTIONS))) for board in boards)
    assert legal == 72
    done = [sp.is_done(board) for board in boards]
    assert sum(done) == 12
    picks = [sp.best_index(board) for board in boards if not sp.is_done(board)]
    assert len(picks) == 12 and None not in picks
    assert Counter(sp.COST_VALUE[sp.ACTIONS[k]] for k in picks) == {0: 3, 1: 3, 2: 3, 3: 3}
    # 逐位置：24 张 × 6 排布 = 144 个位置实例，合法 72 次（一半）。
    for slot in range(1, len(sp.ACTIONS) + 1):
        hits = sum(
            1
            for board in boards
            for layout in sp.LAYOUTS
            if sp.legal_slots(board, layout)[slot - 1]
        )
        assert hits == 72
    # `best` 的位置：只有**没结束**的 12 张才出这道题，× 6 排布 = 72 道，六个位置各 12 道。
    slots = Counter(
        sp.best_slot(board, layout)
        for board in boards
        if not sp.is_done(board)
        for layout in sp.LAYOUTS
    )
    assert set(slots.values()) == {12}
    # `gain` 的档位：每个位置上六档各 24 道。
    for slot in range(1, len(sp.ACTIONS) + 1):
        seen = Counter(
            sp.GAIN_VALUE[sp.ACTIONS[sp.action_at(layout, slot)]]
            for _board in boards
            for layout in sp.LAYOUTS
        )
        assert set(seen.values()) == {24}


def test_the_tiny_prefix_is_balanced_by_construction() -> None:
    """输入：二十四张里的前六张。输出：无。

    预期行为：合法动作数合计 18（占 36 的一半）、结束 3 张、没结束的 3 张都有合法动作、
    该用的那个动作的耗落在三个不同档——小规格跑一遍也能验四族的均衡。
    """
    boards = [sp.board_of(name) for name in sp.BOARDS[:6]]
    legal = sum(sum(sp.is_legal(board, i) for i in range(len(sp.ACTIONS))) for board in boards)
    assert legal == 18
    assert sum(sp.is_done(board) for board in boards) == 3
    picks = [sp.best_index(board) for board in boards if not sp.is_done(board)]
    assert len(picks) == 3 and None not in picks
    assert len({sp.COST_VALUE[sp.ACTIONS[k]] for k in picks}) == 3


def test_layouts_are_latin_squares() -> None:
    """输入：六种排布 × 六个位置。输出：无。

    预期行为：每种排布都是六个动作的一个排列，而且 `action_at` / `slot_of` 互为逆——
    逐位置的合法率与 `best` 的位置分布均匀全靠这条。
    """
    for layout in sp.LAYOUTS:
        assert sorted(sp.action_at(layout, slot) for slot in range(1, 7)) == list(range(6))
        for index in range(6):
            assert sp.action_at(layout, sp.slot_of(layout, index)) == index


def test_layout_and_action_lookups_report_bad_values() -> None:
    """输入：坏排布名 / 坏位置 / 坏动作下标。输出：无。预期行为：都报人话。"""
    with pytest.raises(KeyError, match="layout"):
        sp.layout_of("nope")
    with pytest.raises(ValueError, match="位置要在"):
        sp.action_at("l0", 0)
    with pytest.raises(ValueError, match="位置要在"):
        sp.action_at("l0", 7)
    with pytest.raises(ValueError, match="动作下标"):
        sp.slot_of("l0", 6)
    with pytest.raises(ValueError, match="动作下标"):
        sp.cooldown_key(sp.board_of("b0"), 6)


def test_board_and_key_helpers_report_unknown_values() -> None:
    """输入：坏局面名 / 坏动作 / 坏动作下标。输出：无。预期行为：都报人话并点名字段。"""
    with pytest.raises(KeyError, match="board"):
        sp.board_of("nope")
    with pytest.raises(KeyError, match="action"):
        sp.cost_key("nope")
    with pytest.raises(KeyError, match="action"):
        sp.gain_key("nope")


def test_rows_of_follows_the_layout() -> None:
    """输入：同一张局面的六种排布。输出：无。

    预期行为：行序由**排布**定——`l0` 是动作原序，环移之后每一行换动作，四格跟着换。
    """
    board = sp.board_of("b6")
    plain = sp.rows_of(board, "l0")
    assert [row.action for row in plain] == list(sp.ACTIONS)
    for layout in sp.LAYOUTS:
        rows = sp.rows_of(board, layout)
        assert [row.action for row in rows] == [
            sp.ACTIONS[sp.action_at(layout, slot)] for slot in range(1, 7)
        ]
        for row in rows:
            assert row.cost == sp.cost_key(row.action)
            assert row.gain == sp.gain_key(row.action)
        assert len({row.cooldown for row in rows}) >= 1
    shifted = sp.rows_of(board, "l1")
    assert [row.action for row in shifted] == list(sp.ACTIONS[-1:]) + list(sp.ACTIONS[:-1])


def test_legal_slots_and_best_slot_agree_with_the_action_level_rules() -> None:
    """输入：全部局面 × 六种排布。输出：无。

    预期行为：位置版的合法表 / 答案与动作版的规则换算得一致——排布只换摆放，不换答案。
    """
    for board in (sp.board_of(name) for name in sp.BOARDS):
        for layout in sp.LAYOUTS:
            slots = sp.legal_slots(board, layout)
            assert len(slots) == len(sp.ACTIONS)
            for slot, legal in enumerate(slots, start=1):
                assert legal is sp.is_legal(board, sp.action_at(layout, slot))
            index = sp.best_index(board)
            want = None if index is None else sp.slot_of(layout, index)
            assert sp.best_slot(board, layout) == want


def test_key_helpers_recompute_from_the_cells_only() -> None:
    """输入：规格里的六行与抬头。输出：无。

    预期行为：只吃行里的耗 / 收益 / 冷却三格与抬头的资源格就能重算出全部答案——
    判别器与审计走的就是这条（真值规格与反解结果都满足 `RowLike` / `HeadLike`）。
    """
    for case in sp.build_cases(sp.SimConfig(layouts=("l0", "l3"), phrasings=(0,))):
        spec = sp.sample_spec(case, "train", 0, 0, None)
        assert [sp.legal_of_row(row, spec.resources) for row in spec.rows] == list(spec.legal)
        assert sp.best_slot_of_rows(spec.rows, spec.resources) == spec.best_slot
        assert sp.cost_level_of_rows(spec.rows, spec.resources) == spec.cost_level
        assert [sp.gain_level_of_row(row) for row in spec.rows] == list(spec.gains)
        assert sp.done_of_head(spec) is spec.done


def test_key_helpers_report_unknown_keys() -> None:
    """输入：耗 / 收益 / 冷却三格各喂一个坏键。输出：无。预期行为：都报 KeyError。"""
    good = sp.SimRow("advance", "c1", "g2", "k0")
    with pytest.raises(KeyError):
        sp.legal_of_row(sp.SimRow("advance", "c9", "g2", "k0"), 4)
    with pytest.raises(KeyError):
        sp.legal_of_row(sp.SimRow("advance", "c1", "g2", "k9"), 4)
    with pytest.raises(KeyError):
        sp.gain_level_of_row(sp.SimRow("advance", "c1", "g9", "k0"))
    assert sp.best_slot_of_rows((good,), 4) == 1
    assert sp.best_slot_of_rows((good,), 0) is None


def test_goals_of_the_row_helpers() -> None:
    """输入：一行的四格 / 一行描述。输出：无。预期行为：摊平与描述与键一一对应。"""
    row = sp.SimRow("strike", "c3", "g5", "k2")
    assert row.as_dict() == {"action": "strike", "cost": "c3", "gain": "g5", "cooldown": "k2"}
    assert row.describe() == "strike/c3/g5/k2"


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
    """输入：三个维度各喂一个坏值。输出：无。预期行为：都报人话并点名字段。"""
    with pytest.raises(KeyError, match="board"):
        _make_case(board="nope")
    with pytest.raises(KeyError, match="layout"):
        _make_case(layout="nope")
    with pytest.raises(KeyError, match="句面"):
        _make_case(phrasing=2)


def test_request_id_is_deterministic_and_leaks_no_split() -> None:
    """输入：同一个 case 的两次取号 / 换个切分的取号。输出：无。

    预期行为：同一素材必得同一请求号；请求号只看素材指纹，**与切分无关**（否则同一条样本
    在不同切分里拿到不同请求号，等于把切分信息写进 state）。格式是 `S-` + 五位数字。
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

    预期行为：合法表、收益档、结没结束、该用哪个、该用的耗都由规则函数算出来，
    与行里的四格对得上——规格里没有一处是掷骰子来的。
    """
    for case in sp.build_cases(sp.SimConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        board = sp.board_of(case.board)
        assert spec.resources == board[0] and spec.turn == board[2] and spec.haul == board[3]
        assert spec.legal == sp.legal_slots(board, case.layout)
        assert spec.gains == tuple(
            sp.GAIN_VALUE[sp.ACTIONS[sp.action_at(case.layout, slot)]]
            for slot in range(1, 7)
        )
        assert spec.done is sp.is_done(board)
        assert spec.best_slot == sp.best_slot(board, case.layout)
        assert spec.cost_level == sp.cost_level(board)




def test_build_cases_enumerates_the_whole_grid() -> None:
    """输入：全默认配置。输出：无。

    预期行为：24 张局面 × 6 排布 × 2 句面 = 288 格，顺序稳定、无重复。
    """
    cases = sp.build_cases(sp.SimConfig())
    assert len(cases) == 24 * 6 * 2 == 288
    assert len({sp.material_of(case) for case in cases}) == len(cases)
    assert cases == sp.build_cases(sp.SimConfig())


def test_ood_only_keeps_exactly_the_cases_with_an_unseen_value() -> None:
    """输入：`ood_only` 配置。输出：无。

    预期行为：留下的每一条都至少含一个留出取值，滤掉的正是不含留出取值的那些；
    288 − 24 × 5 × 1 = 168。
    """
    kept = sp.build_cases(sp.SimConfig(ood_only=True))
    assert kept and all(case.unseen_dims() for case in kept)
    assert len(kept) == 288 - 24 * 5 * 1 == 168
    id_materials = {
        sp.material_of(case)
        for case in sp.build_cases(sp.SimConfig())
        if not case.unseen_dims()
    }
    assert not id_materials & {sp.material_of(case) for case in kept}


def test_config_reports_empty_pools() -> None:
    """输入：三张取值表各清空一次。输出：无。预期行为：构造期报人话并点名字段。"""
    for field in ("boards", "layouts", "phrasings"):
        with pytest.raises(ValueError, match=field):
            sp.SimConfig(**{field: ()})
