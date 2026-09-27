"""真值规格的单元测试：规则函数、元素表构造、case 校验、素材指纹与窗口号。"""

from __future__ import annotations

from typing import Any

import pytest

from tests.workflows.browser.conftest import _make_case, _make_spec
from workflows.browser import spec as sp


def test_intent_table_covers_every_task_once() -> None:
    """输入：无。输出：无。

    预期行为：四个任务各映射到一个**不同**的角色与一个**不同**的动作——重复的话
    「按任务认角色」这条判据就有歧义，`action` 那一族也会塌成常数。
    """
    assert set(sp.INTENT) == set(sp.TASKS)
    roles = [sp.target_role(task) for task in sp.TASKS]
    actions = [sp.action_of(task) for task in sp.TASKS]
    assert sorted(roles) == sorted(sp.ROLES)
    assert sorted(actions) == sorted(sp.ACTIONS)


def test_unknown_keys_report_the_field() -> None:
    """输入：四个认不出的键。输出：无。预期行为：都报 KeyError 且点名字段。"""
    with pytest.raises(KeyError, match="task"):
        sp.target_role("nope")
    with pytest.raises(KeyError, match="task"):
        sp.action_of("nope")
    with pytest.raises(KeyError, match="task"):
        sp.object_key("nope", "plain")
    with pytest.raises(KeyError, match="form"):
        sp.object_key("fill", "nope")
    with pytest.raises(KeyError, match="排布"):
        sp.order_of("nope")


def test_object_key_switches_on_the_form() -> None:
    """输入：四个任务的两种对象档。输出：无。

    预期行为：普通档与敏感档给**不同**的键，且敏感档真的落在敏感清单里——两档一样的话
    `risk` 那一族会塌成常数。
    """
    for task in sp.TASKS:
        plain = sp.object_key(task, "plain")
        sensitive = sp.object_key(task, "sensitive")
        assert plain != sensitive
        assert not sp.is_sensitive(plain)
        assert sp.is_sensitive(sensitive)


def test_aux_roles_are_the_complement() -> None:
    """输入：四个任务。输出：无。预期行为：陪衬角色 = 四个角色去掉目标角色，顺序稳定。"""
    for task in sp.TASKS:
        roles = sp.aux_roles(task)
        assert len(roles) == 3
        assert sp.target_role(task) not in roles
        assert set(roles) | {sp.target_role(task)} == set(sp.ROLES)
        assert list(roles) == sorted(roles, key=sp.ROLES.index)


def test_orders_are_a_latin_square() -> None:
    """输入：五种排布。输出：无。

    预期行为：每个槽在五个位置上各出现一次——这正是 `target` 的答案位置均匀的原因，
    破了它答案位置就会偏（均衡闸当场标红）。
    """
    seen: dict[str, dict[int, int]] = {slot: {} for slot in sp.SLOTS}
    for order in sp.ORDERS:
        placed = sp.order_of(order)
        assert sorted(placed) == sorted(sp.SLOTS)
        for position, slot in enumerate(placed, start=1):
            seen[slot][position] = seen[slot].get(position, 0) + 1
    for slot, counts in seen.items():
        assert counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, (slot, counts)


def test_rows_are_five_distinct_elements() -> None:
    """输入：全部 480 个 case。输出：无。

    预期行为：五行元素的（角色, 对象, 状态）两两不同——撞车会渲染出两行一模一样的元素，
    位置题的选项也会跟着重复。
    """
    for case in sp.build_cases(sp.ScreenConfig()):
        rows = sp.rows_of(case)
        assert len(rows) == len(sp.SLOTS)
        assert len({(row.role, row.object, row.state) for row in rows}) == len(rows)


def test_rows_put_the_target_and_the_decoy_on_the_same_role() -> None:
    """输入：全部 480 个 case。输出：无。

    预期行为：目标槽与干扰槽**同角色、不同对象**——这是「只按角色找」这条捷径失效的原因；
    另外三个角色各一个、状态一律可用。
    """
    for case in sp.build_cases(sp.ScreenConfig()):
        rows = sp.rows_of(case)
        role = sp.target_role(case.task)
        same_role = [row for row in rows if row.role == role]
        assert len(same_role) == 2
        assert same_role[0].object != same_role[1].object
        assert {row.role for row in rows} == set(sp.ROLES)
        for row in rows:
            if row.role == role:
                continue
            assert row.state == "ready" and row.object == sp.NEUTRALS[row.role]


def test_target_slot_state_follows_the_done_flag() -> None:
    """输入：同一 case 的两档完成档。输出：无。预期行为：目标槽状态在可用 / 已是当前之间切。"""
    for done, state in ((False, "ready"), (True, "current")):
        spec = _make_spec(done=done)
        row = sp.object_row(spec.rows, spec.task, spec.object)
        assert row is not None and row.state == state


def test_object_row_reports_ambiguity() -> None:
    """输入：手工拼的坏元素表。输出：无。

    预期行为：一行都找不到、或者找到两行，都给 None——静默取第一行会把「表坏了」变成
    一条答案看起来正常的坏样本。
    """
    rows = sp.rows_of(_make_case())
    assert sp.object_row(rows, "fill", "nope") is None
    doubled = (rows[0], rows[0], *rows[2:])
    assert sp.object_row(doubled, "fill", rows[0].object) is None


def test_is_done_ignores_a_decoy_that_is_current() -> None:
    """输入：干扰槽是「已是当前」、目标槽可用的 case。输出：无。

    预期行为：任务**没有**完成——完成判据要求「已是当前」的那一行的标签就是任务要操作的
    那个对象，干扰槽挂的是别的对象。
    """
    spec = _make_spec(done=False, decoy_state="current")
    assert spec.done is False
    assert spec.target_position is not None


def test_target_position_is_none_when_the_target_is_not_operable() -> None:
    """输入：任务已完成的 case。输出：无。

    预期行为：目标槽是「已是当前」时没有可操作的行，定位 / 动作 / 风险三族都不该出。
    """
    spec = _make_spec(done=True)
    assert spec.target_position is None
    assert sp.target_position(spec.rows, spec.task, "nope") is None


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("task", "nope", "task"),
        ("form", "nope", "form"),
        ("decoy_state", "nope", "decoy_state"),
        ("order", "nope", "order"),
        ("phrasing", 9, "句面"),
        ("done", "yes", "完成档"),
    ],
)
def test_case_rejects_bad_dimensions(field: str, value: Any, pattern: str) -> None:
    """输入：六个坏维度各一个。输出：无。预期行为：构造期报 KeyError 且点名那一维。"""
    with pytest.raises(KeyError, match=pattern):
        _make_case(**{field: value})


def test_case_as_dict_and_describe() -> None:
    """输入：一个 case。输出：无。预期行为：六个维度全摊平；描述与素材指纹同形。"""
    case = _make_case()
    assert set(case.as_dict()) == {
        "task",
        "form",
        "done",
        "decoy_state",
        "order",
        "phrasing",
    }
    assert case.describe() == sp.material_of(case)


@pytest.mark.parametrize(
    ("task", "phrasing", "want"),
    [
        ("fill", 0, ()),
        ("fill", 1, ("phrasing",)),
        ("switch", 0, ("task",)),
        ("switch", 1, ("phrasing", "task")),
    ],
)
def test_unseen_dims(task: str, phrasing: int, want: tuple[str, ...]) -> None:
    """输入：任务意图 × 句面四种组合。输出：无。预期行为：点出训练侧没见过的维度。"""
    assert _make_case(task=task, phrasing=phrasing).unseen_dims() == want


def test_screen_id_is_deterministic_and_leaks_no_split() -> None:
    """输入：同一个 case 的两条规格（不同切分 / 序号）。输出：无。

    预期行为：窗口号只由素材指纹派生——用序号或切分名拼窗口号会把切分信息写进 state。
    """
    case = _make_case()
    first = sp.sample_spec(case, "train", 0, 0, None)
    second = sp.sample_spec(case, "test_ood", 7, 0, None)
    assert first.screen_id == second.screen_id == sp.screen_id_of(case)
    assert first.screen_id.startswith("W-")
    other = sp.sample_spec(_make_case(task="open"), "train", 0, 0, None)
    assert other.screen_id != first.screen_id


def test_sample_spec_carries_the_derived_truth() -> None:
    """输入：一个 case。输出：无。预期行为：规格里带上算好的完成度与目标位置。"""
    spec = _make_spec()
    assert spec.split == "train" and spec.seq == 0 and spec.phrasing == 0
    assert spec.done is False
    assert spec.target_position == 1
    assert sp.object_key(spec.task, spec.case.form) == spec.object


def test_spec_to_dict_is_json_shaped() -> None:
    """输入：一条规格。输出：无。预期行为：摘要里的每一项都能直接进 manifest。"""
    payload = sp.spec_to_dict(_make_spec())
    assert set(payload) == {
        "case",
        "unseen_dims",
        "screen_id",
        "task",
        "object",
        "done",
        "target_position",
        "rows",
    }
    assert len(payload["rows"]) == len(sp.SLOTS)
    assert isinstance(payload["unseen_dims"], list)


def test_build_cases_enumerates_the_whole_grid() -> None:
    """输入：全开配置。输出：无。

    预期行为：六张表的全组合（480 = 4 × 2 × 2 × 3 × 5 × 2），顺序稳定，六维两两不矛盾。
    """
    cases = sp.build_cases(sp.ScreenConfig())
    expected = 4 * 2 * 2 * 3 * 5 * 2
    assert len(cases) == expected
    assert len({sp.material_of(case) for case in cases}) == expected
    assert cases == sp.build_cases(sp.ScreenConfig())


def test_ood_only_keeps_exactly_the_cases_with_an_unseen_value() -> None:
    """输入：全开 + `ood_only`。输出：无。预期行为：留下的都含留出取值，且与同分布侧互补。"""
    everything = sp.build_cases(sp.ScreenConfig())
    ood = sp.build_cases(sp.ScreenConfig(ood_only=True))
    id_side = [case for case in everything if not case.unseen_dims()]
    assert len(ood) + len(id_side) == len(everything)
    assert all(case.unseen_dims() for case in ood)
    assert not {sp.material_of(case) for case in ood} & {
        sp.material_of(case) for case in id_side
    }
    # 训练侧取值表就是「留出维只留一个取值」的那张网格：3 × 2 × 2 × 3 × 5 × 1。
    seen = sp.ScreenConfig(
        tasks=tuple(sorted(sp.SEEN_TASKS)),
        phrasings=tuple(sorted(sp.SEEN_PHRASINGS)),
    )
    assert len(sp.build_cases(seen)) == len(id_side) == 3 * 2 * 2 * 3 * 5 * 1


def test_element_helpers() -> None:
    """输入：一个元素。输出：无。预期行为：三格能摊平成 dict，也能拼成一行描述。"""
    element = sp.Element("field", "note", "ready")
    assert element.as_dict() == {"role": "field", "object": "note", "state": "ready"}
    assert element.describe() == "field/note/ready"


def test_object_tables_share_no_keys_and_no_role_slots() -> None:
    """输入：四张对象表。输出：无。

    预期行为：一共 16 个对象键两两不同（4 陪衬 + 4 干扰 + 8 目标），而且干扰槽挂的对象
    与目标槽的两个对象都不同——撞了会渲染出两行一模一样的元素。
    """
    used: set[str] = set(sp.NEUTRALS.values()) | set(sp.DECOYS.values())
    for task in sp.TASKS:
        used |= set(sp.OBJECTS[task])
    assert len(used) == len(sp.NEUTRALS) + len(sp.DECOYS) + 2 * len(sp.TASKS) == 16
    for task in sp.TASKS:
        assert sp.DECOYS[task] not in sp.OBJECTS[task]
        assert sp.DECOYS[task] not in sp.NEUTRALS.values()
