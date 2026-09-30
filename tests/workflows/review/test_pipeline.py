"""改动审查的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod

from workflows.review import features, locales, render, workflow as review_workflow
from workflows.review.spec import ReviewSpec, spec_to_dict

from tests.workflows.review.conftest import FakeBackend, body_for

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train", lang: str = "zh"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="review",
        split=split,
        locales=(lang,),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=review_workflow.ReviewConfig(split=split, case_limit=CASE_LIMIT),
        backend=backend,
    )


def test_audit_accepts_the_other_locales_tables(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：一份 **en** 产物，审计时故意传 zh。输出：(报告, 缺失项)。

    预期行为：零缺失——审查规则表的「条件」那一列是随语言渲染的文案，审计必须按
    「等于任一语言的规范表」判定，而不是拿调用方给的那一种语言去比。反面实测过：
    混语言的产物按 zh 口径审计，8 条里 4 条被误判成「审查规则表与规范表对不上」；
    这份全 en 的产物更是 8 条全中。
    """
    result = pipeline.run(make_config(tmp_path, "loc", fake_backend(), lang="en"))
    assert result.accepted == CASE_LIMIT
    report, missing = workflow_mod.load("review").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["rules_bad"] == 0


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，
    而且清单行、正文形状、三张表、严重度与门控全都自洽。
    """
    result = pipeline.run(make_config(tmp_path, "c1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("review").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    for counter in (
        "manifest_bad",
        "body_shape_bad",
        "rules_bad",
        "severity_table_bad",
        "gate_table_bad",
        "severity_bad",
        "gate_bad",
    ):
        assert report["train"][counter] == 0, counter


@pytest.mark.parametrize(
    ("switch", "code"),
    [
        ("trace_lies_secret", "E_JUDGE_TRACE"),
        ("trace_lies_debug", "E_JUDGE_TRACE"),
        ("table_body", "E_ASSEMBLE"),
        ("no_prefix_body", "E_ASSEMBLE"),
        ("fence_body", "E_ASSEMBLE"),
        ("label_body", "E_ASSEMBLE"),
        ("empty_body", "E_ASSEMBLE"),
    ],
)
def test_bad_drafts_and_bad_traces_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：各种作弊开关。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是该有的那一个——反解说反话是 `E_JUDGE_TRACE`，
    形状 / 内容不合规是渲染期的 `E_ASSEMBLE`（重试耗尽后仍被拒，不静默放行）。
    """
    result = pipeline.run(make_config(tmp_path, f"c3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：同分布三个切分互不相交、`test_ood` 与它们不相交，四份合起来正好是全空间
    （同分布侧剔除留出取值后是 60 组，留出侧 48 组，合计 108 组）。
    """
    keys: dict[tuple[tuple[str, str], ...], str] = {}
    for split in ("train", "val", "test_iid", "test_ood"):
        cases = review_workflow.cases_for(review_workflow.ReviewConfig(split=split))
        assert cases
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in keys, f"{key} 同时出现在 {keys[key]} 与 {split}"
            keys[key] = split
    assert len(keys) == 108


def test_id_splits_never_carry_the_held_out_value() -> None:
    """输入：同分布三个切分。输出：无。预期行为：一条都不带留出的变更类型。"""
    for split in ("train", "val", "test_iid"):
        cases = review_workflow.cases_for(review_workflow.ReviewConfig(split=split))
        assert cases
        for case in cases:
            assert case.change_kind not in review_workflow.OOD_CHANGE_KINDS


def test_ood_side_only_carries_the_held_out_value() -> None:
    """输入：`test_ood` 的 case 列表。输出：无。预期行为：每一条都带留出的那个取值。"""
    cases = review_workflow.cases_for(review_workflow.ReviewConfig(split="test_ood"))
    assert cases
    for case in cases:
        assert case.change_kind in review_workflow.OOD_CHANGE_KINDS


def test_ood_side_still_reaches_every_option() -> None:
    """输入：`test_ood` 的 case 列表。输出：无。

    预期行为：留出 `change_kind` 不杀任何一档——五族每一档在留出侧都还到得了。
    """
    labels = [
        case.label()
        for case in review_workflow.cases_for(review_workflow.ReviewConfig(split="test_ood"))
    ]
    assert {label.secret for label in labels} == set(features.SECRETS)
    assert {label.debug for label in labels} == set(features.DEBUGS)
    assert {label.rule for label in labels} == set(features.REVIEW_RULES)
    assert {label.severity for label in labels} == set(range(features.SEVERITY_LEVELS))
    assert {label.gate for label in labels} == set(features.GATES)


def test_id_side_still_reaches_every_option() -> None:
    """输入：同分布三个切分的 case 列表。输出：无。

    预期行为：留出「新增」不杀任何一档——五族每一档在训练侧都还到得了。
    """
    labels = [
        case.label()
        for split in ("train", "val", "test_iid")
        for case in review_workflow.cases_for(review_workflow.ReviewConfig(split=split))
    ]
    assert {label.rule for label in labels} == set(features.REVIEW_RULES)
    assert {label.severity for label in labels} == set(range(features.SEVERITY_LEVELS))
    assert {label.gate for label in labels} == set(features.GATES)


def test_cover_limit_forces_full_enumeration() -> None:
    """输入：默认 `cover_limit`。输出：无。

    预期行为：同分布空间（60 组）与全空间（108 组）都在 512 之下，所以走的是**全枚举**
    ——成对覆盖在这里只会白扔组合。
    """
    id_cases = review_workflow.cases_for(review_workflow.ReviewConfig(split="train"))
    assert review_workflow.ReviewConfig().cover_limit >= 108
    assert len(features.build_cases(review_workflow.ReviewConfig().cover_limit)) == 108
    assert id_cases


def test_bad_knobs_are_rejected() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话。"""
    with pytest.raises(ValueError):
        review_workflow.ReviewConfig(split="train", case_limit=-1)
    with pytest.raises(ValueError):
        review_workflow.ReviewConfig(split="train", phrasing_pool=())
    with pytest.raises(KeyError):
        review_workflow.ReviewWorkflow().config_for("nope")


def test_render_without_backend_raises() -> None:
    """输入：没注入后端的 workflow。输出：无。

    预期行为：直接报错，绝不悄悄降级成模板——那会让「正文由模型写」这条前提失效。
    """
    wf = review_workflow.ReviewWorkflow()
    spec = ReviewSpec(tags=features.build_cases(2048)[0], phrasing=0, split="train", seq=0)
    with pytest.raises(RuntimeError):
        wf.render(spec, random.Random(0), "zh")


def test_unknown_judge_name_raises() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：直接报错。"""
    with pytest.raises(KeyError):
        review_workflow.ReviewWorkflow().make_judges(("nope",), "zh")


def test_workflow_declares_what_it_needs() -> None:
    """输入：workflow 的类属性。输出：无。

    预期行为：`needs` 非空（正文由后端写）、三种题型都声明、判别器名单与默认一致。
    """
    wf = review_workflow.ReviewWorkflow()
    assert wf.name == "review"
    assert wf.needs == frozenset({"text"})
    assert wf.kinds == frozenset({"choice", "noul", "score"})
    assert wf.judges == ("hygiene", "frame", "trace")
    assert wf.available_locales() == ["en", "zh"]
    assert wf.split_names() == ("train", "val", "test_iid", "test_ood")


def test_config_for_and_reparse_wire_up() -> None:
    """输入：切分名与一段 state。输出：无。

    预期行为：`config_for` 给出该切分的采样参数、`reparse` 走纯代码反解——
    这两条是通用层与 workflow 的接线点，接错了整条管线就哑了。
    """
    wf = review_workflow.ReviewWorkflow()
    assert wf.config_for("train").split == "train"
    assert wf.config_for("test_ood").split == "test_ood"
    tags = features.build_cases(2048)[0]
    state = render.to_state({"body": body_for(tags.as_dict())}, tags, locales.load("zh"))
    parsed = wf.reparse(state)
    assert parsed.body == body_for(tags.as_dict())
    assert parsed.manifest is not None


def test_case_label_and_stats_dimensions_cover_every_dimension() -> None:
    """输入：一个 case。输出：无。预期行为：可读描述与统计维度覆盖全部五维。"""
    wf = review_workflow.ReviewWorkflow()
    case = features.build_cases(2048)[0]
    label = wf.case_label(case)
    for dim, value in case.as_dict().items():
        assert f"{dim}={value}" in label
    assert set(wf.stats_dimensions(case)) == set(features.DIMS)
    assert set(wf.dimension_titles()) == set(features.DIMS)


def test_spec_to_dict_and_task_ids() -> None:
    """输入：一条真值规格。输出：无。

    预期行为：`spec_to_dict` 可 JSON 序列化且字段齐全；渲染 / 反解任务 id 按同一条
    样本可重算（断点续跑按它查进度）。
    """
    tags = features.build_cases(2048)[0]
    spec = ReviewSpec(tags=tags, phrasing=2, split="train", seq=5)
    assert spec_to_dict(spec) == {
        "tags": tags.as_dict(),
        "phrasing": 2,
        "split": "train",
        "seq": 5,
    }
    assert spec.labels == tags.label()
    assert spec.render_task_id == "review-train-0000005-render"
    assert spec.trace_task_id == "review-train-0000005-trace"
