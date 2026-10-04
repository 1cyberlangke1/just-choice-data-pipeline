"""会话上下文压缩的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import pipeline

from workflows.compact import features, locales, render, workflow as compact_workflow
from workflows.compact.spec import CompactSpec

from tests.workflows._scaffold import run_config
from tests.workflows.compact.conftest import BASE, FakeBackend

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="compact",
        split=split,
        seed=7,
        override=compact_workflow.CompactConfig(split=split, case_limit=CASE_LIMIT),
    )


def test_trace_judge_rejects_a_misread_fact(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解把「有没有承载持久事实」说反。输出：RunResult。预期行为：同样被拒。"""
    result = pipeline.run(make_config(tmp_path, "c4", fake_backend(trace_lies_fact=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_render_rejects_a_where_word_in_the_body(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：把位置词写进正文的假后端。输出：RunResult。

    预期行为：渲染期就拒掉，拒绝码是 `E_ASSEMBLE`（生成阶段的统一拒绝码）——
    这类文本拼进 state 就洗不掉了，而且会与「本块位置」那一行打架。
    """
    result = pipeline.run(make_config(tmp_path, "c5", fake_backend(where_body=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_render_rejects_a_multi_line_body(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：正文写成两行的假后端。输出：RunResult。预期行为：重试耗尽后仍被拒，不静默放行。"""
    result = pipeline.run(make_config(tmp_path, "c6", fake_backend(multi_line_body=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：同分布三个切分互不相交、`test_ood` 与它们不相交，四者并起来等于全空间——
    一个相容组合都不丢、也不重复。
    """
    wf = compact_workflow.CompactWorkflow()
    parts: dict[str, set[tuple[tuple[str, str], ...]]] = {}
    for split in wf.split_names():
        cases = wf.build_cases(compact_workflow.CompactConfig(split=split))
        parts[split] = {tuple(case.as_dict().items()) for case in cases}
    assert parts["train"] & parts["val"] == set()
    assert parts["train"] & parts["test_iid"] == set()
    assert parts["val"] & parts["test_iid"] == set()
    for split in ("train", "val", "test_iid"):
        assert parts[split] & parts["test_ood"] == set()
    union = set().union(*parts.values())
    assert len(union) == sum(len(part) for part in parts.values())
    assert union == {tuple(case.as_dict().items()) for case in features.build_cases(1024)}


def test_id_side_has_no_dead_options() -> None:
    """输入：同分布侧的 case 列表。输出：无。

    预期行为：`test_ood` 留出的取值在同分布侧一次都不出现，而且**每一族每一档都还到得了**
    ——留出一个取值连带杀死别的族的好几档，就是死选项。
    """
    cases = [
        case
        for split in ("train", "val", "test_iid")
        for case in compact_workflow.cases_for(compact_workflow.CompactConfig(split=split))
    ]
    assert cases
    for case in cases:
        assert case.age not in compact_workflow.OOD_AGES
    labels = [case.label() for case in cases]
    assert {label.rule for label in labels} == set(features.RULES)
    assert {label.disposition for label in labels} == set(features.DISPOSITIONS)
    assert {label.priority for label in labels} == set(range(features.PRIORITY_LEVELS))
    assert {label.kind for label in labels} == set(features.KINDS)
    assert {label.fact for label in labels} == set(features.FACTS)
    assert {label.pin_overrode for label in labels} == {False, True}


def test_every_train_option_appears_in_train() -> None:
    """输入：train 切分的 case 列表。输出：无。

    预期行为：训练侧每一族每一档都到得了——只靠「同分布侧整体」到得了还不够，
    训练集本身要能教到每一个答案。
    """
    labels = [
        case.label()
        for case in compact_workflow.cases_for(compact_workflow.CompactConfig(split="train"))
    ]
    assert {label.rule for label in labels} == set(features.RULES)
    assert {label.disposition for label in labels} == set(features.DISPOSITIONS)
    assert {label.priority for label in labels} == set(range(features.PRIORITY_LEVELS))
    assert {label.kind for label in labels} == set(features.KINDS)
    assert {label.fact for label in labels} == set(features.FACTS)
    assert {label.pin_overrode for label in labels} == {False, True}


def test_ood_side_only_carries_the_held_out_value() -> None:
    """输入：`test_ood` 的 case 列表。输出：无。预期行为：每一条都带留出的那个取值。"""
    cases = compact_workflow.cases_for(compact_workflow.CompactConfig(split="test_ood"))
    assert cases
    for case in cases:
        assert case.age in compact_workflow.OOD_AGES


def test_cover_limit_forces_full_enumeration() -> None:
    """输入：默认 `cover_limit`。输出：无。

    预期行为：同分布空间（108 组）与全空间（162 组）都在 256 之下，所以走的是**全枚举**
    ——成对覆盖在这里只会白扔组合。
    """
    id_cases = compact_workflow.cases_for(compact_workflow.CompactConfig(split="train"))
    assert compact_workflow.CompactConfig().cover_limit >= 144
    assert len(features.build_cases(compact_workflow.CompactConfig().cover_limit)) == 144
    assert id_cases


def test_bad_knobs_are_rejected() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话。"""
    with pytest.raises(ValueError):
        compact_workflow.CompactConfig(split="train", case_limit=-1)
    with pytest.raises(ValueError):
        compact_workflow.CompactConfig(split="train", phrasing_pool=())
    with pytest.raises(KeyError):
        compact_workflow.CompactWorkflow().config_for("nope")


def test_config_for_and_reparse_wire_up() -> None:
    """输入：切分名与一段 state。输出：无。

    预期行为：`config_for` 给出该切分的采样参数、`reparse` 走纯代码反解——
    这两条是通用层与 workflow 的接线点，接错了整条管线就哑了。
    """
    wf = compact_workflow.CompactWorkflow()
    assert wf.config_for("train").split == "train"
    assert wf.config_for("test_ood").split == "test_ood"
    tags = features.build_cases(1024)[0]
    state = render.to_state({"content": BASE[tags.kind]}, tags, locales.load("zh"))
    parsed = wf.reparse(state)
    assert parsed.content == BASE[tags.kind]
    assert parsed.where is not None


def test_task_ids() -> None:
    """输入：一条真值规格。输出：无。

    预期行为：渲染 / 反解任务 id 按同一条
    样本可重算（断点续跑按它查进度）。
    """
    tags = features.build_cases(1024)[0]
    spec = CompactSpec(tags=tags, phrasing=2, split="train", seq=5)
    assert spec.labels == tags.label()
    assert spec.render_task_id == "compact-train-0000005-render"
    assert spec.trace_task_id == "compact-train-0000005-trace"
