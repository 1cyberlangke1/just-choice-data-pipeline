"""case 矩阵与计划：留出、指纹、骨架、旋钮。"""

from __future__ import annotations

import random

import pytest

from tests.workflows.autoformat.conftest import _make_case
from workflows.autoformat import spec as sp
from workflows.autoformat.workflow import ID_CONFIG, OOD_CONFIG, AutoformatWorkflow


def test_material_is_stable_and_sensitive() -> None:
    """输入：同一个 case 两次、只差一维的两个 case。输出：无。

    预期行为：指纹是确定性的；任何一维不同指纹就不同（否则切分会把两份文档混在一起）。
    """
    case = _make_case()
    assert sp.material_of(case) == sp.material_of(_make_case())
    for field, value in (
        ("topic", "release"),
        ("wrap", 56),
        ("shape", "list"),
        ("variant", "ordered"),
        ("code_family", "build"),
    ):
        assert sp.material_of(case) != sp.material_of(_make_case(**{field: value}))


def test_default_config_covers_every_shape_and_variant() -> None:
    """输入：默认配置。输出：无。预期行为：六种形状、六种变体、五种命令族全都出现过。"""
    cases = sp.build_cases(sp.DocConfig())
    assert {case.shape for case in cases} == set(sp.SHAPES)
    assert {case.variant for case in cases} == set(sp.ALL_VARIANTS)
    assert {case.code_family for case in cases} == set(sp.ALL_CODE_FAMILIES)
    assert {case.topic for case in cases} == set(sp.TOPICS)
    assert {case.wrap for case in cases} == set(sp.WRAPS)


def test_id_side_has_no_unseen_value_and_ood_side_has_one() -> None:
    """输入：训练侧配置与 OOD 配置。输出：无。

    预期行为：同分布侧**一个没见过取值都没有**；OOD 侧**每条都至少含一个**——
    这就是「值留出」在本格式里的落地形态。
    """
    id_cases = sp.build_cases(ID_CONFIG)
    ood_cases = sp.build_cases(OOD_CONFIG)
    assert id_cases and ood_cases
    assert all(case.unseen_dims() == () for case in id_cases)
    assert all(case.unseen_dims() for case in ood_cases)
    assert not set(map(sp.material_of, id_cases)) & set(map(sp.material_of, ood_cases))


def test_unseen_dims_names_the_right_dimensions() -> None:
    """输入：四个各自只差一维的 case。输出：无。预期行为：点名点到具体维度上。"""
    assert _make_case(topic="incident").unseen_dims() == ("topic",)
    assert _make_case(wrap=56).unseen_dims() == ("wrap",)
    assert _make_case(shape="callout", variant="tip").unseen_dims() == ("variant",)
    assert _make_case(shape="code", code_family="container").unseen_dims() == ("code_family",)
    assert _make_case(topic="migration", wrap=56).unseen_dims() == ("topic", "wrap")


def test_ood_only_skips_fully_seen_cases() -> None:
    """输入：全取值表 + `ood_only`。输出：无。预期行为：留下的每条都至少含一个没见过取值。"""
    config = sp.DocConfig(ood_only=True)
    cases = sp.build_cases(config)
    assert cases
    assert all(case.unseen_dims() for case in cases)
    assert len(cases) < len(sp.build_cases(sp.DocConfig()))


def test_plan_skeleton_starts_with_a_heading_then_a_paragraph() -> None:
    """输入：六种形状各一个 case。输出：无。

    预期行为：第一块是标题、第二块是正文——标题**紧贴**正文是构造约定，
    反解侧正是靠「短行后面跟着长行」把标题切出来的。
    """
    for shape in sp.SHAPES:
        case = _make_case(shape=shape, variant=_variant_for(shape), code_family=_family_for(shape))
        plan = sp.build_plan(case)
        assert plan[0].kind == sp.KIND_HEADING
        assert plan[1].kind == sp.KIND_PARAGRAPH
        assert len(plan[1].parts) >= 2, "正文必须由多句拼成，否则永远不会被折行"


def test_sample_spec_is_deterministic() -> None:
    """输入：同一 case 造两次规格。输出：无。预期行为：计划逐字段一致（同素材同题同答）。"""
    first = sp.sample_spec(_make_case(shape="mixed", variant="warning", code_family="vcs"), "a", 0, 0)
    second = sp.sample_spec(_make_case(shape="mixed", variant="warning", code_family="vcs"), "b", 9, 1)
    assert first.plan == second.plan
    assert (first.split, first.seq, first.phrasing) != (second.split, second.seq, second.phrasing)


def test_unknown_slot_in_a_template_reports_the_name() -> None:
    """输入：带未知槽位的模板。输出：无。预期行为：报 KeyError 并点名槽位。"""
    with pytest.raises(KeyError, match="bogus"):
        sp._part("sent", "build", 0, "这里有 {bogus} 槽位", random.Random(0))


def test_empty_knobs_are_rejected() -> None:
    """输入：五张取值表各清空一次。输出：无。预期行为：构造期 ValueError 点名字段。"""
    for field in ("topics", "wraps", "shapes", "variants", "code_families", "phrasings"):
        with pytest.raises(ValueError, match=field):
            sp.DocConfig(**{field: ()})


def test_workflow_declares_four_splits_and_no_backend() -> None:
    """输入：workflow 实例。输出：无。预期行为：四切分、零后端、两个判别器。"""
    workflow = AutoformatWorkflow()
    assert workflow.split_names() == ("train", "val", "test_iid", "test_ood")
    assert workflow.needs == frozenset()
    assert workflow.kinds == frozenset({"choice", "noul"})
    assert set(workflow.judges) == {"roundtrip", "label"}


def _variant_for(shape: str) -> str:
    """输入：形状名。输出：该形状第一个合法变体（测试里只用来凑出一个合法 case）。"""
    return sp.SHAPE_VARIANTS[shape][0]


def _family_for(shape: str) -> str:
    """输入：形状名。输出：该形状第一个合法命令族（`code` / `mixed` 才不是 `na`）。"""
    return sp.SHAPE_CODE_FAMILIES[shape][0]
