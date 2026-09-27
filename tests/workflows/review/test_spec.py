"""真值规格的单元测试：规则函数、三张表、六处改动的构造、case 校验与改动号。"""

from __future__ import annotations

from typing import Any

import pytest

from tests.workflows.review.conftest import _make_case, _make_spec
from workflows.review import spec as sp


def test_matches_switches_on_the_rule() -> None:
    """输入：三条规则各自的判据。输出：无。

    预期行为：`bulk` 只看规模格；`secret` / `debug` 走正则搜正文；规则名不认识直接报错。
    """
    assert sp.matches(sp.RULE_BULK, "big", "随便什么正文")
    assert not sp.matches(sp.RULE_BULK, "small", "随便什么正文")
    assert sp.matches(sp.RULE_SECRET, "small", 'password = "hunter2"')
    assert not sp.matches(sp.RULE_SECRET, "small", "改从环境变量读，不写死")
    assert sp.matches(sp.RULE_DEBUG, "small", "print(中间结果)")
    assert not sp.matches(sp.RULE_DEBUG, "small", "什么都不打印")
    with pytest.raises(KeyError, match="规则"):
        sp.matches("nope", "small", "正文")


def test_rule_from_rejects_a_body_that_hits_two_rules() -> None:
    """输入：同时踩两条规则的正文。输出：无。

    预期行为：报人话——静默取第一条会让「该不该提」跟着掷骰子。
    """
    with pytest.raises(ValueError, match="同时命中多条规则"):
        sp.rule_from("small", 'password = "hunter2" 然后 print(它)')


def test_rule_from_is_none_for_a_clean_body() -> None:
    """输入：什么都没命中的正文。输出：无。预期行为：给 None；规模格是「大」时给 `bulk`。"""
    assert sp.rule_from("small", "把注释补全了") is None
    assert sp.rule_from("big", "把注释补全了") == sp.RULE_BULK


def test_raise_table_is_the_core_judgement() -> None:
    """输入：三条规则 × 四种文件。输出：无。

    预期行为：命中规则**不等于**该提——凭据在文档里不提、调试在测试里不提、规模只在源码提。
    """
    for rule in sp.RULES:
        for file in sp.FILES:
            finding = sp.Finding(rule, file, "add", "small", "x")
            assert sp.raises_(finding) is (file in sp.RAISE[rule]), (rule, file)
    assert sp.raises_(sp.Finding(None, "src", "add", "small", "clean_clean")) is False


def test_severity_and_gate_tables() -> None:
    """输入：三条规则 / 九种（严重度 × 变更）组合。输出：无。

    预期行为：严重度是规则 → 档的固定映射；门控是**另一张表**，覆盖九种组合；没命中规则时
    两者都给 None。
    """
    assert sp.SEVERITY == {sp.RULE_SECRET: 3, sp.RULE_DEBUG: 2, sp.RULE_BULK: 1}
    assert set(sp.GATE) == {
        (severity, change) for severity in (1, 2, 3) for change in sp.CHANGES
    }
    assert set(sp.GATE.values()) == set(sp.GATES)
    for rule, severity in sp.SEVERITY.items():
        for change in sp.CHANGES:
            finding = sp.Finding(rule, "src", change, "small", "x")
            assert sp.severity_of(finding) == severity
            assert sp.gate_of(finding) == sp.GATE[(severity, change)]
    clean = sp.Finding(None, "src", "add", "small", "clean_clean")
    assert sp.severity_of(clean) is None and sp.gate_of(clean) is None


def test_gate_is_not_a_rename_of_severity() -> None:
    """输入：中等严重度在「新增」与「删除」两种变更下。输出：无。

    预期行为：门控比严重度细一档——同样是「中」，新增走请求修改、删除走拦下。
    这条是 `gate` 那一族不塌成常数的前提。
    """
    add = sp.Finding(sp.RULE_DEBUG, "src", "add", "small", "x")
    remove = sp.Finding(sp.RULE_DEBUG, "src", "remove", "small", "x")
    assert sp.severity_of(add) == sp.severity_of(remove) == 2
    assert sp.gate_of(add) == "request" and sp.gate_of(remove) == "block"


def test_orders_are_a_latin_square() -> None:
    """输入：六种排布。输出：无。

    预期行为：每个槽在六个位置上各出现一次——这正是 `top` 的答案位置均匀的原因，
    破了它答案位置就会偏（均衡闸当场标红）。
    """
    seen: dict[str, dict[int, int]] = {slot: {} for slot in sp.SLOTS}
    for order in sp.ORDERS:
        placed = sp.order_of(order)
        assert sorted(placed) == sorted(sp.SLOTS)
        for position, slot in enumerate(placed, start=1):
            seen[slot][position] = seen[slot].get(position, 0) + 1
    for slot, counts in seen.items():
        assert counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, (slot, counts)


def test_unknown_keys_report_the_field() -> None:
    """输入：五个认不出的键。输出：无。预期行为：都报 KeyError 且点名字段。"""
    with pytest.raises(KeyError, match="排布"):
        sp.order_of("nope")
    with pytest.raises(KeyError, match="triple"):
        _make_case(triple="nope")
    with pytest.raises(KeyError, match="noise"):
        _make_case(noise="nope")
    with pytest.raises(KeyError, match="change"):
        _make_case(change="nope")
    with pytest.raises(KeyError, match="句面"):
        _make_case(phrasing=9)


def test_rows_have_three_raised_and_three_not() -> None:
    """输入：全开网格。输出：无。

    预期行为：六处改动里恰好三处该提（严重度 3 / 2 / 1 各一个）、两处命中但不提、一处干净；
    最严重的那一处唯一，就是严重度 3 的那一处。
    """
    for case in sp.build_cases(sp.ReviewConfig()):
        rows = sp.rows_of(case)
        assert len(rows) == len(sp.SLOTS)
        raised = [row for row in rows if sp.raises_(row)]
        assert len(raised) == 3
        assert sorted(sp.severity_of(row) for row in raised) == [1, 2, 3]
        assert sum(1 for row in rows if row.rule is not None) == 5
        assert sum(1 for row in rows if row.rule is None) == 1
        top = sp.top_position(rows)
        assert top is not None and sp.severity_of(rows[top - 1]) == 3
        assert sp.raised_positions(rows) == tuple(
            index for index, row in enumerate(rows, start=1) if sp.raises_(row)
        )


def test_top_position_is_none_when_tied_or_when_nothing_is_raised() -> None:
    """输入：并列最高 / 一处都不提 的两种改动。输出：无。

    预期行为：都给 None——「最严重的是哪一处」这时没有唯一解，出这种题就是坏数据。
    """
    filler = tuple(
        sp.Finding(None, "src", "add", "small", f"clean_{name}")
        for name in ("a", "b", "c", "d")
    )
    tied = (
        sp.Finding(sp.RULE_SECRET, "src", "add", "small", "secret_heavy"),
        sp.Finding(sp.RULE_SECRET, "src", "edit", "small", "secret_heavy"),
        *filler,
    )
    assert sp.top_position(tied) is None
    assert sp.raised_positions(tied) == (1, 2)
    quiet = tuple(
        sp.Finding(None, "doc", "add", "small", f"clean_{slot}") for slot in sp.SLOTS
    )
    assert sp.top_position(quiet) is None
    assert sp.raised_positions(quiet) == ()


def test_case_dict_and_describe() -> None:
    """输入：一个 case。输出：无。预期行为：五个维度全摊平；描述与素材指纹同形。"""
    case = _make_case()
    assert set(case.as_dict()) == {"triple", "noise", "change", "order", "phrasing"}
    assert case.describe() == sp.material_of(case)


@pytest.mark.parametrize(
    ("noise", "phrasing", "want"),
    [
        ("n0", 0, ()),
        ("n0", 1, ("phrasing",)),
        ("n2", 0, ("noise",)),
        ("n2", 1, ("noise", "phrasing")),
    ],
)
def test_unseen_dims(noise: str, phrasing: int, want: tuple[str, ...]) -> None:
    """输入：噪声搭配 × 句面四种组合。输出：无。预期行为：点出训练侧没见过的维度。"""
    assert _make_case(noise=noise, phrasing=phrasing).unseen_dims() == want


def test_change_id_is_deterministic_and_leaks_no_split() -> None:
    """输入：同一个 case 的两条规格（不同切分 / 序号）。输出：无。

    预期行为：改动号只由素材指纹派生——用序号或切分名拼改动号会把切分信息写进 state。
    """
    case = _make_case()
    first = sp.sample_spec(case, "train", 0, 0, None)
    second = sp.sample_spec(case, "test_ood", 7, 0, None)
    assert first.change_id == second.change_id == sp.change_id_of(case)
    assert first.change_id.startswith("R-")
    other = sp.sample_spec(_make_case(triple="t1"), "train", 0, 0, None)
    assert other.change_id != first.change_id


def test_sample_spec_carries_the_derived_truth() -> None:
    """输入：一个 case。输出：无。预期行为：规格里带上算好的该提处与最严重处。"""
    spec = _make_spec()
    assert spec.split == "train" and spec.seq == 0 and spec.phrasing == 0
    assert spec.raised == sp.raised_positions(spec.rows) == (1, 2, 3)
    assert spec.top_position == sp.top_position(spec.rows) == 1


def test_spec_to_dict_is_json_shaped() -> None:
    """输入：一条规格。输出：无。预期行为：摘要里的每一项都能直接进 manifest。"""
    payload = sp.spec_to_dict(_make_spec())
    assert set(payload) == {
        "case",
        "unseen_dims",
        "change_id",
        "raised",
        "top_position",
        "rows",
    }
    assert len(payload["rows"]) == len(sp.SLOTS)
    assert isinstance(payload["unseen_dims"], list)


def test_finding_helpers() -> None:
    """输入：两处改动。输出：无。预期行为：能摊平成 dict，也能拼成一行描述。"""
    finding = sp.Finding(sp.RULE_SECRET, "src", "add", "small", "secret_heavy")
    assert finding.as_dict() == {
        "rule": "secret",
        "file": "src",
        "change": "add",
        "size": "small",
        "body": "secret_heavy",
    }
    assert finding.describe() == "secret/src/add/small"
    clean = sp.Finding(None, "doc", "remove", "small", "clean_clean")
    assert clean.describe() == "clean/doc/remove/small"


def test_build_cases_enumerates_the_whole_grid() -> None:
    """输入：全开配置。输出：无。

    预期行为：五张表的全组合（432 = 6 × 3 × 2 × 6 × 2），顺序稳定，五维两两不矛盾。
    """
    cases = sp.build_cases(sp.ReviewConfig())
    expected = 6 * 3 * 2 * 6 * 2
    assert len(cases) == expected == 432
    assert len({sp.material_of(case) for case in cases}) == expected
    assert cases == sp.build_cases(sp.ReviewConfig())


def test_ood_only_keeps_exactly_the_cases_with_an_unseen_value() -> None:
    """输入：全开 + `ood_only`。输出：无。预期行为：留下的都含留出取值，且与同分布侧互补。"""
    everything = sp.build_cases(sp.ReviewConfig())
    ood = sp.build_cases(sp.ReviewConfig(ood_only=True))
    id_side = [case for case in everything if not case.unseen_dims()]
    assert len(ood) + len(id_side) == len(everything)
    assert all(case.unseen_dims() for case in ood)
    assert not {sp.material_of(case) for case in ood} & {
        sp.material_of(case) for case in id_side
    }
    seen = sp.ReviewConfig(
        noises=tuple(sorted(sp.SEEN_NOISES)), phrasings=tuple(sorted(sp.SEEN_PHRASINGS))
    )
    assert len(sp.build_cases(seen)) == len(id_side) == 6 * 2 * 2 * 6 * 1
