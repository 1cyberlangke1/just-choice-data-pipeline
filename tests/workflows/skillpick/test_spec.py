"""真值规格的单元测试：规则函数、四张表、六行技能表的构造、case 校验与请求号。"""

from __future__ import annotations

import pytest

from tests.workflows.skillpick.conftest import _make_case, _make_spec
from workflows.skillpick import spec as sp


def test_entry_of_needs_an_exact_match() -> None:
    """输入：六个技能标签 / 六个干扰标签 / 一个不存在的标签。输出：无。

    预期行为：完全一致才匹配（干扰标签给 None）；认不出的标签直接报错——
    「按子串匹配」的读者会在干扰标签上选错技能，这条是 `fits` 那一族的前提。
    """
    for entry in sp.CATALOG:
        assert sp.entry_of(sp.TAG_OF[entry]) == entry
    for orphan in sp.ORPHANS:
        assert sp.entry_of(orphan) is None
    with pytest.raises(KeyError, match="tag"):
        sp.entry_of("nope")


def test_every_decoy_contains_a_skill_tag_as_a_substring() -> None:
    """输入：六个干扰标签。输出：无。

    预期行为：每个干扰标签都把某个技能标签当**子串**含在里面——陷阱必须真的存在，
    否则「完全一致才算匹配」这条判据就没有对手，`fits` 也退化成一道白送的题。
    """
    for orphan in sp.ORPHANS:
        parents = [tag for tag in sp.TAGS if tag != orphan and tag in orphan]
        assert parents, f"干扰标签 {orphan!r} 里没有技能标签当子串"


def test_allowed_is_a_rank_comparison() -> None:
    """输入：三档权限范围 × 两种授权。输出：无。

    预期行为：范围不高于授权才给用——管理员级技能在两种授权下都**不给用**，
    所以这条表不是「两档对两档」的巧合，`allow` 的答案真的要看这一格。
    """
    want = {
        ("read", "read"): True,
        ("read", "write"): True,
        ("write", "read"): False,
        ("write", "write"): True,
        ("admin", "read"): False,
        ("admin", "write"): False,
    }
    for (scope, grant), expected in want.items():
        assert sp.allowed(scope, grant) is expected, (scope, grant)
    with pytest.raises(KeyError, match="scope"):
        sp.allowed("nope", "read")
    with pytest.raises(KeyError, match="grant"):
        sp.allowed("read", "nope")


def test_share_resource_is_key_equality() -> None:
    """输入：三对资源键。输出：无。预期行为：同一类就冲突；认不出的键报人话。"""
    assert sp.share_resource("code", "code") is True
    assert sp.share_resource("code", "net") is False
    assert sp.share_resource("data", "data") is True
    with pytest.raises(KeyError, match="resource"):
        sp.share_resource("code", "nope")


def test_scope_and_resource_lookups() -> None:
    """输入：六个条目。输出：无。预期行为：三档范围各两个、三类资源各两个；认不出报人话。"""
    assert {sp.scope_of(entry) for entry in sp.CATALOG} == set(sp.SCOPES)
    assert {sp.resource_of(entry) for entry in sp.CATALOG} == set(sp.RESOURCES)
    for scope in sp.SCOPES:
        assert sum(1 for entry in sp.CATALOG if sp.scope_of(entry) == scope) == 2
    for resource in sp.RESOURCES:
        assert sum(1 for entry in sp.CATALOG if sp.resource_of(entry) == resource) == 2
    with pytest.raises(KeyError, match="entry"):
        sp.scope_of("nope")
    with pytest.raises(KeyError, match="entry"):
        sp.resource_of("nope")


def test_layouts_are_a_latin_square() -> None:
    """输入：六种排布。输出：无。

    预期行为：每个条目在六个位置上各出现一次——这正是 `pick` 答案位置均匀的原因，
    破了它答案位置就会偏（均衡闸当场标红）。
    """
    seen: dict[str, dict[int, int]] = {entry: {} for entry in sp.CATALOG}
    for layout in sp.LAYOUTS:
        placed = [sp.entry_at(layout, slot) for slot in range(1, len(sp.CATALOG) + 1)]
        assert sorted(placed) == sorted(sp.CATALOG)
        for position, entry in enumerate(placed, start=1):
            seen[entry][position] = seen[entry].get(position, 0) + 1
    for entry, counts in seen.items():
        assert counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, (entry, counts)


def test_slot_of_is_the_inverse_of_entry_at() -> None:
    """输入：六种排布 × 六个条目。输出：无。预期行为：互为反函数；位置越界报人话。"""
    for layout in sp.LAYOUTS:
        for slot in range(1, len(sp.CATALOG) + 1):
            assert sp.slot_of(layout, sp.entry_at(layout, slot)) == slot
    with pytest.raises(ValueError, match="位置"):
        sp.entry_at("l0", 7)
    with pytest.raises(KeyError, match="entry"):
        sp.slot_of("l0", "nope")


def test_rows_are_the_catalog_in_layout_order() -> None:
    """输入：六种排布。输出：无。

    预期行为：六行、位置顺序稳定，每行四格都从条目键查出来；一行里的适用场景与条目一一对应。
    """
    for layout in sp.LAYOUTS:
        rows = sp.rows_of(layout)
        assert len(rows) == len(sp.CATALOG)
        assert [row.entry for row in rows] == [
            sp.entry_at(layout, slot) for slot in range(1, len(sp.CATALOG) + 1)
        ]
        for row in rows:
            assert row.tag == sp.TAG_OF[row.entry]
            assert row.scope == sp.SCOPE_OF[row.entry]
            assert row.resource == sp.RESOURCE_OF[row.entry]


def test_inuse_is_the_neighbour_above_or_below() -> None:
    """输入：六个条目 × 两种占用配对。输出：无。

    预期行为：`above` / `below` 就是表里上下挪一行；而**恰好一个**邻居与目标同资源——
    这条是 `conflict` 的 50/50 能靠构造保证的全部依据。
    """
    size = len(sp.CATALOG)
    for index, entry in enumerate(sp.CATALOG):
        above = sp.inuse_entry(sp.TAG_OF[entry], "above")
        below = sp.inuse_entry(sp.TAG_OF[entry], "below")
        assert above == sp.CATALOG[(index - 1) % size]
        assert below == sp.CATALOG[(index + 1) % size]
        same = [
            sp.share_resource(sp.RESOURCE_OF[entry], sp.RESOURCE_OF[other])
            for other in (above, below)
        ]
        assert sorted(same) == [False, True], (entry, above, below)


def test_inuse_on_a_decoy_tag_still_changes_the_state() -> None:
    """输入：一个干扰标签、两种占用配对。输出：无。

    预期行为：干扰标签没有基准条目，就以第 0 个条目为基准——两种配对仍给出**不同**的
    已占用技能，不会造出两条一模一样的样本。
    """
    above = sp.inuse_entry(sp.ORPHANS[0], "above")
    below = sp.inuse_entry(sp.ORPHANS[0], "below")
    assert above != below
    assert above == sp.CATALOG[-1] and below == sp.CATALOG[1]
    with pytest.raises(KeyError, match="inuse"):
        sp.inuse_entry(sp.ORPHANS[0], "nope")


def test_unknown_keys_report_the_field() -> None:
    """输入：五个认不出的键。输出：无。预期行为：都报 KeyError 且点名字段。"""
    with pytest.raises(KeyError, match="layout"):
        sp.layout_of("nope")
    with pytest.raises(KeyError, match="tag"):
        _make_case(tag="nope")
    with pytest.raises(KeyError, match="layout"):
        _make_case(layout="nope")
    with pytest.raises(KeyError, match="grant"):
        _make_case(grant="nope")
    with pytest.raises(KeyError, match="inuse"):
        _make_case(inuse="nope")
    with pytest.raises(KeyError, match="句面"):
        _make_case(phrasing=9)


def test_case_dict_and_describe() -> None:
    """输入：一个 case。输出：无。预期行为：五个维度全摊平；描述与素材指纹同形。"""
    case = _make_case()
    assert set(case.as_dict()) == {"tag", "layout", "grant", "inuse", "phrasing"}
    assert case.describe() == sp.material_of(case)


@pytest.mark.parametrize(
    ("layout", "phrasing", "want"),
    [
        ("l0", 0, ()),
        ("l0", 1, ("phrasing",)),
        ("l5", 0, ("layout",)),
        ("l5", 1, ("layout", "phrasing")),
    ],
)
def test_unseen_dims(layout: str, phrasing: int, want: tuple[str, ...]) -> None:
    """输入：排布 × 句面四种组合。输出：无。预期行为：点出训练侧没见过的维度。"""
    assert _make_case(layout=layout, phrasing=phrasing).unseen_dims() == want


def test_request_id_is_deterministic_and_leaks_no_split() -> None:
    """输入：同一个 case 的两条规格（不同切分 / 序号）。输出：无。

    预期行为：请求号只由素材指纹派生——用序号或切分名拼请求号会把切分信息写进 state。
    """
    case = _make_case()
    first = sp.sample_spec(case, "train", 0, 0, None)
    second = sp.sample_spec(case, "test_ood", 7, 0, None)
    assert first.request_id == second.request_id == sp.request_id_of(case)
    assert first.request_id.startswith(sp.REQUEST_PREFIX)
    other = sp.sample_spec(_make_case(tag="edit_code"), "train", 0, 0, None)
    assert other.request_id != first.request_id


def test_sample_spec_carries_the_derived_truth() -> None:
    """输入：一个 case。输出：无。

    预期行为：规格里带上算好的目标位置、已占用标签与两条判据——`allow` / `conflict`
    只对**匹配到技能**的请求有意义，没匹配上时两者都是 False（那一族本来也不出）。
    """
    spec = _make_spec()
    assert spec.split == "train" and spec.seq == 0 and spec.phrasing == 0
    assert spec.target_entry == "repo" and spec.target_slot == 1
    assert spec.inuse_tag == sp.TAG_OF[sp.CATALOG[-1]]
    assert spec.allowed is True and spec.conflicts is False

    orphan = _make_spec(tag="read_code_notes")
    assert orphan.target_entry is None and orphan.target_slot is None
    assert orphan.allowed is False and orphan.conflicts is False


def test_spec_to_dict_is_json_shaped() -> None:
    """输入：一条规格。输出：无。预期行为：摘要里的每一项都能直接进 manifest。"""
    payload = sp.spec_to_dict(_make_spec())
    assert set(payload) == {
        "case",
        "unseen_dims",
        "request_id",
        "target_slot",
        "target_entry",
        "inuse_tag",
        "allowed",
        "conflicts",
        "rows",
    }
    assert len(payload["rows"]) == len(sp.CATALOG)
    assert isinstance(payload["unseen_dims"], list)


def test_row_helpers() -> None:
    """输入：技能表的一行。输出：无。预期行为：能摊平成 dict，也能拼成一行描述。"""
    row = sp.rows_of("l0")[0]
    assert row.as_dict() == {
        "entry": "repo",
        "tag": "read_code",
        "scope": "read",
        "resource": "code",
    }
    assert row.describe() == "read_code/read/code"


def test_build_cases_enumerates_the_whole_grid() -> None:
    """输入：全开配置。输出：无。

    预期行为：五张表的全组合（576 = 12 × 6 × 2 × 2 × 2），顺序稳定，五维两两不矛盾。
    """
    cases = sp.build_cases(sp.SkillConfig())
    expected = 12 * 6 * 2 * 2 * 2
    assert len(cases) == expected == 576
    assert len({sp.material_of(case) for case in cases}) == expected
    assert cases == sp.build_cases(sp.SkillConfig())


def test_ood_only_keeps_exactly_the_cases_with_an_unseen_value() -> None:
    """输入：全开 + `ood_only`。输出：无。预期行为：留下的都含留出取值，且与同分布侧互补。"""
    everything = sp.build_cases(sp.SkillConfig())
    ood = sp.build_cases(sp.SkillConfig(ood_only=True))
    id_side = [case for case in everything if not case.unseen_dims()]
    assert len(ood) + len(id_side) == len(everything)
    assert all(case.unseen_dims() for case in ood)
    assert not {sp.material_of(case) for case in ood} & {
        sp.material_of(case) for case in id_side
    }
    seen = sp.SkillConfig(
        layouts=tuple(sorted(sp.SEEN_LAYOUTS)), phrasings=tuple(sorted(sp.SEEN_PHRASINGS))
    )
    assert len(sp.build_cases(seen)) == len(id_side) == 12 * 5 * 2 * 2 * 1
