"""真值规格的单元测试：规则函数、六行清单的构造、case 校验与屏号。"""

from __future__ import annotations

import pytest

from tests.workflows.elementpick.conftest import _make_case
from workflows.elementpick import spec as sp


def test_position_and_role_follow_the_layout() -> None:
    """输入：六种排布 / 六个行号。输出：无。

    预期行为：位置是「标签 + 排布」的环形移位——固定标签时六种排布给出 1..6 各一次，
    固定排布时六个标签也给出 1..6 各一次（两个方向都是双射，所以留出任何一维，
    位置分布都还铺满六行）；角色按行号循环且三分类相邻成对。这两条一起保证
    `target` 的六个位置与 `role` 的三档都是均匀的。
    """
    for label in sp.LABELS:
        assert sorted(sp.position_of(name, label) for name in sp.LAYOUTS) == [1, 2, 3, 4, 5, 6]
    for name in sp.LAYOUTS:
        assert sorted(sp.position_of(name, label) for label in sp.LABELS) == [1, 2, 3, 4, 5, 6]
    with pytest.raises(KeyError, match="layout"):
        sp.position_of("l9", sp.LABELS[0])
    with pytest.raises(KeyError, match="label"):
        sp.position_of(sp.LAYOUTS[0], "nope")
    assert [sp.role_at(index) for index in range(1, 7)] == list(sp.ROLES)
    assert [sp.role_class_of(sp.role_at(index)) for index in range(1, 7)] == [
        "button_like",
        "button_like",
        "link_like",
        "link_like",
        "input_like",
        "input_like",
    ]
    with pytest.raises(ValueError, match="行号"):
        sp.role_at(0)
    with pytest.raises(KeyError, match="role"):
        sp.role_class_of("nope")


def test_labels_are_pairwise_substring_free() -> None:
    """输入：六个标签。输出：无。

    预期行为：没有哪个标签是另一个标签的子串——无关行的文本因此不会把动作标签
    「当子串含在里面」，`shadow` 那一族的「有 / 没有」只由近似元素决定，不靠巧合。
    """
    for label in sp.LABELS:
        for other in sp.LABELS:
            if other != label:
                assert label not in other, (label, other)


def test_matching_is_exact_and_near_is_the_shadow_prefix() -> None:
    """输入：标签键 / 近似键 / 无关键。输出：无。

    预期行为：完全一致才算匹配（近似键不算）；近似只认 `shadow:<标签>`。
    """
    assert sp.text_matches("save", "save") is True
    assert sp.text_matches("shadow:save", "save") is False
    assert sp.text_matches("cancel", "save") is False
    assert sp.text_is_near("shadow:save", "save") is True
    assert sp.text_is_near("save", "save") is False
    assert sp.text_is_near("shadow:cancel", "save") is False


def test_blocked_style_alternates_between_the_two_ways() -> None:
    """输入：四个序号。输出：无。

    预期行为：偶数号「隐藏」、奇数号「可见但不可点」——两种挡法在同一批数据里都出现，
    判据里那两条都不是摆设。
    """
    assert sp.blocked_style(0) == ("hidden", "yes")
    assert sp.blocked_style(1) == ("visible", "no")
    assert sp.blocked_style(2) == ("hidden", "yes")
    assert sp.blocked_style(3) == ("visible", "no")


def test_hit_has_exactly_one_usable_same_name_row() -> None:
    """输入：全开网格里的每个 `hit` case。输出：无。

    预期行为：同名行恰好一行可见且可点（落点），其余同名行都被挡住——判据里
    「不用比谁更靠上」这句话就是靠这条构造成立的。
    """
    for case in sp.build_cases(sp.ElementConfig()):
        rows = sp.rows_of(case)
        assert len(rows) == sp.SIZE
        hits = [row for row in rows if sp.text_matches(row.text, case.label)]
        usable = [row for row in hits if row.usable()]
        if case.shape == "hit":
            assert len(usable) == 1, case.describe()
            assert sp.target_slot(rows, case.label) == sp.position_of(case.layout, case.label)
            assert sp.blocked_count(rows, case.label) == case.extras
        else:
            assert usable == [], case.describe()
            assert sp.target_slot(rows, case.label) is None
            assert sp.blocked_count(rows, case.label) == case.extras + 1


def test_shadow_row_carries_the_near_key() -> None:
    """输入：全开网格。输出：无。预期行为：近似元素在不在屏幕上，与 `case.shadow` 一致。"""
    for case in sp.build_cases(sp.ElementConfig()):
        rows = sp.rows_of(case)
        assert sp.shadow_present(rows, case.label) is case.shadow
        near = [row for row in rows if sp.is_shadow_text(row.text)]
        assert len(near) == (1 if case.shadow else 0)


def test_unrelated_rows_never_repeat_a_text() -> None:
    """输入：全开网格。输出：无。

    预期行为：一行里同一个文本最多出现一次（同名行除外）——无关行的文本两两不同，
    否则「有几个同名元素」会变成数不清的糊涂账。
    """
    for case in sp.build_cases(sp.ElementConfig()):
        rows = sp.rows_of(case)
        plain = [row.text for row in rows if not sp.text_matches(row.text, case.label)]
        assert len(set(plain)) == len(plain), case.describe()


def test_every_case_yields_a_distinct_layout_of_rows() -> None:
    """输入：全开网格。输出：无。

    预期行为：六维不同 ⇒ 六行清单不同（`phrasing` 除外，它只影响题面）——
    「同 case 维度不同 ⇒ 文本不同」是构造不变量，破了它就会出现两条一样的样本。
    """
    seen: dict[tuple[object, ...], str] = {}
    # 只留第一套句面：句面**只影响题面**，两套句面的六行清单本来就该一模一样。
    for case in sp.build_cases(sp.ElementConfig(phrasings=(sp.PHRASINGS[0],))):
        # 动作那一行也要进 key：换一个动作标签、同时把同名行挪一行，六行文本可能撞车，
        # 但动作那一行不同 —— state 仍然不同（`test_pipeline` 里另有逐字节的断言）。
        key = (
            case.label,
            *((row.role, row.visibility, row.clickable, row.text) for row in sp.rows_of(case)),
        )
        assert key not in seen, (case.describe(), seen.get(key))
        seen[key] = case.describe()


def test_case_rejects_unknown_values() -> None:
    """输入：六维各一个坏取值。输出：无。预期行为：都报人话并点名字段。"""
    with pytest.raises(KeyError, match="label"):
        _make_case(label="nope")
    with pytest.raises(KeyError, match="layout"):
        _make_case(layout="l9")
    with pytest.raises(KeyError, match="shape"):
        _make_case(shape="maybe")
    with pytest.raises(KeyError, match="extras"):
        _make_case(extras=9)
    with pytest.raises(KeyError, match="shadow"):
        _make_case(shadow="yes")
    with pytest.raises(KeyError, match="phrasing"):
        _make_case(phrasing=7)


def test_element_row_rejects_unknown_cells() -> None:
    """输入：四格各一个坏取值。输出：无。预期行为：都报人话（渲染与反解都靠它）。"""
    with pytest.raises(KeyError, match="role"):
        sp.ElementRow("nope", "visible", "yes", "save")
    with pytest.raises(KeyError, match="visibility"):
        sp.ElementRow("button", "nope", "yes", "save")
    with pytest.raises(KeyError, match="clickable"):
        sp.ElementRow("button", "visible", "nope", "save")
    with pytest.raises(KeyError, match="text"):
        sp.ElementRow("button", "visible", "yes", "nope")
    assert sp.ElementRow("button", "visible", "yes", "shadow:save").usable() is True


def test_unseen_dims_only_names_the_held_out_ones() -> None:
    """输入：三种取值组合。输出：无。预期行为：只有留出的两维会被点名。"""
    assert _make_case().unseen_dims() == ()
    assert _make_case(layout="l5").unseen_dims() == ("layout",)
    assert _make_case(phrasing=1).unseen_dims() == ("phrasing",)
    assert _make_case(layout="l5", phrasing=1).unseen_dims() == ("layout", "phrasing")


def test_screen_id_is_deterministic_and_split_free() -> None:
    """输入：同一 case 的两次取样、以及跨切分的同一 case。输出：无。

    预期行为：屏号只由素材决定——换切分不换屏号（否则等于把切分信息写进 state）。
    """
    case = _make_case()
    first = sp.sample_spec(case, "train", 0, 0, None)
    second = sp.sample_spec(case, "test_ood", 7, 0, None)
    assert first.screen_id == second.screen_id == sp.screen_id_of(case)
    assert first.screen_id.startswith(sp.SCREEN_PREFIX)
    assert first.screen_id != sp.screen_id_of(_make_case(label="cancel"))




def test_build_cases_covers_the_whole_grid() -> None:
    """输入：默认配置。输出：无。预期行为：六张表取全组合，顺序稳定且可复现。"""
    cases = sp.build_cases(sp.ElementConfig())
    expected = (
        len(sp.LABELS)
        * len(sp.LAYOUTS)
        * len(sp.SHAPES)
        * len(sp.EXTRA_COUNTS)
        * len(sp.SHADOWS)
        * len(sp.PHRASINGS)
    )
    assert len(cases) == expected
    assert cases == sp.build_cases(sp.ElementConfig())


def test_ood_only_keeps_cases_with_an_unseen_value() -> None:
    """输入：`ood_only` 配置。输出：无。预期行为：每条都至少含一个留出取值。"""
    cases = sp.build_cases(sp.ElementConfig(ood_only=True))
    assert cases and all(case.unseen_dims() for case in cases)


def test_empty_pool_is_reported_by_name() -> None:
    """输入：六张表各清空一次。输出：无。预期行为：构造期报人话并点名字段。"""
    with pytest.raises(ValueError, match="labels"):
        sp.ElementConfig(labels=())
    with pytest.raises(ValueError, match="layouts"):
        sp.ElementConfig(layouts=())
    with pytest.raises(ValueError, match="shapes"):
        sp.ElementConfig(shapes=())
    with pytest.raises(ValueError, match="extras"):
        sp.ElementConfig(extras=())
    with pytest.raises(ValueError, match="shadows"):
        sp.ElementConfig(shadows=())
    with pytest.raises(ValueError, match="phrasings"):
        sp.ElementConfig(phrasings=())
