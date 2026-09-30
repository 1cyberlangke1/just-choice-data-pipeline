"""浏览器 / GUI 操作的特征层测试：五维表、元素表、风险表与三档门控。"""

from __future__ import annotations

from collections import Counter

import pytest

from workflows.browser import features
from workflows.browser.features import TagSet


def all_tags() -> list[TagSet]:
    """输入：无。输出：全空间里所有特征组合（192 组，走全枚举）。"""
    return features.build_cases(1024)


def tag(**overrides: str) -> TagSet:
    """输入：要覆盖的维度。输出：其余取默认值的一组特征（构造反例用）。"""
    base = {
        "intent": "fill",
        "state": "ready",
        "sensitive": "no",
        "decoy": "none",
        "screen": "form",
    }
    base.update(overrides)
    return TagSet(**base)


def test_full_space_size_and_determinism() -> None:
    """输入：维度表与全空间枚举。输出：无。

    预期行为：4×4×2×2×3 = 192 组，一对禁用对都没有，所以全枚举就是 192 组；
    同一入参两次枚举逐组相同（case 库要能逐字节重建）。
    """
    total = 1
    for values in features.DIMS.values():
        total *= len(values)
    assert total == 192
    assert len(all_tags()) == 192
    assert [case.as_dict() for case in all_tags()] == [case.as_dict() for case in all_tags()]


def test_every_dimension_value_is_reachable() -> None:
    """输入：全空间组合。输出：无。预期行为：五维的每一档都出现过。"""
    for name, values in features.DIMS.items():
        assert {getattr(case, name) for case in all_tags()} == set(values)


def test_every_answer_level_is_reachable() -> None:
    """输入：全空间组合。输出：无。预期行为：七个题族每一档都到得了。

    偏斜来自判据本身（门控表与风险表就是这么写的），照实标出不掩盖；
    要的是**每一档都到得了**，不是每档一样多。
    """
    labels = [case.label() for case in all_tags()]
    assert {label.intent for label in labels} == set(features.INTENTS)
    assert {label.action for label in labels} == set(features.ACTIONS)
    assert {label.done for label in labels} == {False, True}
    assert {label.blocked for label in labels} == {False, True}
    assert {label.confirm for label in labels} == {False, True}
    assert {label.risk for label in labels} == set(range(features.RISK_LEVELS))
    assert features.NONE_TARGET in {label.target for label in labels}


def test_every_role_can_be_the_target_and_be_named() -> None:
    """输入：全空间组合。输出：无。

    预期行为：四种角色都可能成为目标（`target` 那一族每种角色都出得到），
    而且「可操作时」的答案一定落在该角色的标签池里。
    """
    seen: Counter[str] = Counter()
    for case in all_tags():
        label = case.label()
        if label.target != features.NONE_TARGET:
            role = features.role_for_intent(case.intent)
            assert label.target in features.LABELS[(role, case.sensitive)]
            seen[role] += 1
    assert set(seen) == set(features.ROLES)


def test_labels_are_sensitive_exactly_when_they_should_be() -> None:
    """输入：元素标签词表。输出：无。

    预期行为：同一格里每个标签命中 `SENSITIVE_WORDS` **当且仅当**那一格是 `yes`——
    `is_sensitive` 是 `confirm` 那一族的判据，词表与判据必须同源。
    """
    for (role, sensitive), pool in features.LABELS.items():
        assert role in features.ROLES
        assert sensitive in features.SENSITIVES
        assert len(pool) >= 3, f"{role}/{sensitive} 的标签太少"
        for label in pool:
            assert features.is_sensitive(label) is (sensitive == "yes"), label


def test_label_pools_never_overlap() -> None:
    """输入：元素标签词表。输出：无。

    预期行为：跨组的字符串两两不同——同一张元素表里不许出现两个同名选项
    （选项名重复会被契约拒掉，而且反解会认不出用户指的是哪一个）。
    """
    flat = [label for pool in features.LABELS.values() for label in pool]
    assert len(flat) == len(set(flat))


def test_pick_label_differs_for_adjacent_salts() -> None:
    """输入：角色、敏感档、相邻的两个 salt。输出：无。

    预期行为：干扰行与目标行不能同名——`pick_label(..., salt)` 与 `pick_label(..., salt + 1)`
    必须给出不同的标签。
    """
    for (role, sensitive), pool in features.LABELS.items():
        for salt in range(len(pool) * 2):
            assert features.pick_label(role, sensitive, salt) != features.pick_label(
                role, sensitive, salt + 1
            )


def test_filler_rows_never_steal_the_target_role() -> None:
    """输入：每种屏型 × 每种目标角色。输出：无。

    预期行为：填充行一共三行、角色两两不同、而且**绝不含目标角色**——否则
    「该操作哪个元素」就有两个候选。标签一律取不敏感那一格（敏感只有目标行与干扰行带）。
    """
    for screen in features.SCREENS:
        for role in features.ROLES:
            rows = features.filler_rows(screen, role, 7)
            assert len(rows) == features.FILLER_LIMIT
            assert len({row.role for row in rows}) == len(rows)
            assert role not in {row.role for row in rows}
            for row in rows:
                assert row.label in features.LABELS[(row.role, "no")]


def test_element_table_has_unique_labels_and_one_ready_target() -> None:
    """输入：全空间组合。输出：无。

    预期行为：一张表里标签两两不同（选项名不能重复）；目标行的状态就是注入的 `state`；
    `decoy=same_role` 时多出来的那一行与目标行同角色、但一定不可用。
    """
    for case in all_tags():
        rows = features.rows_for(case)
        assert len({row.label for row in rows}) == len(rows)
        target = features.wanted_row(case)
        assert target.state == case.state
        same_role = [row for row in rows if row.role == target.role]
        expected = 2 if case.decoy == "same_role" else 1
        assert len(same_role) == expected
        if case.decoy == "same_role":
            other = next(row for row in same_role if row != target)
            assert other.state in {"disabled", "hidden"}
            assert other.label != target.label


def test_rows_for_is_a_rotation_of_the_logical_table() -> None:
    """输入：全空间组合。输出：无。

    预期行为：显示顺序只是把逻辑顺序旋转一下（同一组特征必得同一张表），
    而且目标行**不是**老在第一行——否则「用户指的是第几行」这种反解问题会退化成常数。
    """
    positions: set[int] = set()
    for case in all_tags():
        logical = features.logical_rows(case)
        shown = features.rows_for(case)
        assert sorted(row.label for row in shown) == sorted(row.label for row in logical)
        assert shown.index(features.wanted_row(case)) + 1 == features.wanted_index(case)
        positions.add(features.wanted_index(case))
    # 一屏最多 1 目标行 + 1 干扰行 + FILLER_LIMIT 行填充
    assert positions == set(range(1, features.FILLER_LIMIT + 3))


def test_salt_depends_only_on_the_tags() -> None:
    """输入：同一组特征。输出：无。预期行为：salt 可复算——元素表与答案才能逐字节重建。"""
    case = tag(intent="submit", state="hidden", sensitive="yes", decoy="same_role", screen="list")
    assert features._salt(case) == features._salt(tag(**case.as_dict()))
    assert len({features._salt(case) for case in all_tags()}) > 1


def test_three_gates_are_mutually_exclusive_and_exhaustive() -> None:
    """输入：全空间组合。输出：无。

    预期行为：`done`（已是当前）/ `blocked`（置灰或不可见）/ 「可操作」三者互斥且穷尽；
    可操作时 `target` 是元素名、`action` 按意图表；不可操作时 `target` 是 `none`、`action` 是 `none`。
    """
    for case in all_tags():
        label = case.label()
        assert sum([label.done, label.blocked, label.target != features.NONE_TARGET]) == 1
        assert (label.action != "none") is (label.target != features.NONE_TARGET)
        assert label.done is (case.state == "current")
        assert label.blocked is (case.state in {"disabled", "hidden"})
        assert label.confirm is (case.state == "ready" and case.sensitive == "yes")


def test_risk_comes_from_the_table_only() -> None:
    """输入：全空间组合。输出：无。预期行为：风险档恒等于意图 × 敏感那一格。"""
    for case in all_tags():
        assert case.label().risk == features.RISK_TABLE[(case.intent, case.sensitive)]


def test_risk_table_covers_every_intent_and_reaches_every_level() -> None:
    """输入：风险表。输出：无。预期行为：四种意图 × 两种敏感都有格，0~3 四档都到得了。"""
    assert set(features.RISK_TABLE) == {
        (intent, sensitive)
        for intent in features.INTENTS
        for sensitive in features.SENSITIVES
    }
    assert set(features.RISK_TABLE.values()) == set(range(features.RISK_LEVELS))


def test_action_table_matches_the_action_options() -> None:
    """输入：意图表与动作选项表。输出：无。

    预期行为：`ACTION_OF_INTENT` 的值域去掉 `none` 正好是 `ACTIONS` 去掉 `none`——
    题面里那份动作选项与判据表不能各写各的。
    """
    assert set(features.ACTION_OF_INTENT.values()) == set(features.ACTIONS) - {"none"}
    assert set(features.ROLE_OF_INTENT) == set(features.INTENTS)
    assert set(features.ROLE_OF_INTENT.values()) == set(features.ROLES)


def test_lookup_helpers_raise_on_unknown_tokens() -> None:
    """输入：不认识的意图。输出：无。预期行为：抛 KeyError，不静默兜底。"""
    with pytest.raises(KeyError):
        features.role_for_intent("bogus")
    with pytest.raises(KeyError):
        features.action_for_intent("bogus")
    with pytest.raises(KeyError):
        features.pick_label("bogus", "no", 0)
    with pytest.raises(KeyError):
        features.filler_rows("bogus", "field", 0)


def test_no_forbidden_pairs_so_every_combination_is_compatible() -> None:
    """输入：全空间组合。输出：无。

    预期行为：五维是「意图 × 界面快照」，任意组合都写得出来——所以禁用表是空的、
    `compatible` 恒为真，`allow_pair` 与顺序无关（留着它是为了与其它 workflow 同形）。
    """
    assert features.FORBIDDEN == frozenset()
    for case in all_tags():
        assert features.compatible(case)
    assert features.allow_pair("intent", "fill", "state", "ready")
    assert features.allow_pair("state", "ready", "intent", "fill")


def test_build_cases_can_restrict_the_dimension_table() -> None:
    """输入：剔除一档的维度表。输出：无。预期行为：只产出留下的那一档，不改动 `DIMS`。"""
    dims = {**features.DIMS, "screen": ("form", "list")}
    cases = features.build_cases(1024, dims=dims)
    assert len(cases) == 128
    assert {case.screen for case in cases} == {"form", "list"}
    assert features.DIMS["screen"] == features.SCREENS


def test_choice_options_cover_the_fixed_vocabularies() -> None:
    """输入：选项表。输出：无。预期行为：意图与动作两族的选项就是各自那一维的取值。"""
    assert features.CHOICE_OPTIONS == {"intent": features.INTENTS, "action": features.ACTIONS}
    assert features.NONE_TARGET in features.CHOICE_OPTIONS["action"]
    assert features.ACTIONS[-1] == features.NONE_TARGET


def test_label_for_is_pure() -> None:
    """输入：同一组特征算两次。输出：无。预期行为：结果相同——标签先行、只看特征。"""
    case = tag(intent="switch", state="disabled", sensitive="yes", decoy="same_role", screen="settings")
    assert case.label() == features.label_for(case)


def test_target_row_only_exists_when_ready() -> None:
    """输入：可用 / 不可用的目标行。输出：无。

    预期行为：只有「可用」时才有可操作的元素——已完成（`current`）或被拦住
    （`disabled` / `hidden`）时没有可操作的元素。
    """
    ready = tag(state="ready")
    assert features.target_row(ready) == features.wanted_row(ready)
    for state in ("current", "disabled", "hidden"):
        assert features.target_row(tag(state=state)) is None


def test_compatible_returns_false_when_a_pair_is_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：临时塞进一对禁用对。输出：无。

    预期行为：`compatible` 当场为假、`allow_pair` 与顺序无关——本 workflow 现在一对禁用对
    都没有，但这条防线将来加上取值对时要立刻生效。
    """
    monkeypatch.setattr(
        features, "FORBIDDEN", frozenset({("intent", "fill", "state", "ready")})
    )
    assert not features.compatible(tag(intent="fill", state="ready"))
    assert features.compatible(tag(intent="fill", state="current"))
    assert not features.allow_pair("intent", "fill", "state", "ready")
    assert not features.allow_pair("state", "ready", "intent", "fill")
