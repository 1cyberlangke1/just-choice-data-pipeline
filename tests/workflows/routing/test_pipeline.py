"""工单分诊的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod

from workflows.routing import features, locales, render, workflow as routing_workflow
from workflows.routing.spec import RoutingSpec, spec_to_dict

from tests.workflows.routing.conftest import FakeBackend, body_for

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train", lang: str = "zh"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="routing",
        split=split,
        locales=(lang,),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=routing_workflow.RoutingConfig(split=split, case_limit=CASE_LIMIT),
        backend=backend,
    )


def test_audit_accepts_the_other_locales_tables(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：一份 **en** 产物，审计时故意传 zh。输出：(报告, 缺失项)。

    预期行为：零缺失——派单表的「条件」列与附加规则的「说明」列都是随语言渲染的文案，
    审计必须按「等于任一语言的规范表」判定，而不是拿调用方给的那一种语言去比。反面实测过：
    混语言的产物按 zh 口径审计，8 条里 4 条被误判成「派单表与规范表对不上」、4 条被误判成
    「附加规则与规范表对不上」；这份全 en 的产物更是两张表各 8 条全中。
    """
    result = pipeline.run(make_config(tmp_path, "loc", fake_backend(), lang="en"))
    assert result.accepted == CASE_LIMIT
    report, missing = workflow_mod.load("routing").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["queue_table_bad"] == 0
    assert report["train"]["flags_bad"] == 0


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，
    而且队列与「命中规则」、优先级与「紧急度 × 客户等级」、人工与队列全都自洽。
    """
    result = pipeline.run(make_config(tmp_path, "c1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("routing").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    assert report["train"]["attrs_bad"] == 0
    assert report["train"]["queue_table_bad"] == 0
    assert report["train"]["priority_table_bad"] == 0
    assert report["train"]["flags_bad"] == 0
    assert report["train"]["queue_bad"] == 0
    assert report["train"]["priority_bad"] == 0
    assert report["train"]["human_bad"] == 0


def test_two_runs_are_byte_identical(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：同种子跑两次。输出：两份产物。预期行为：逐字节一致。"""
    first = pipeline.run(make_config(tmp_path, "c2a", fake_backend()))
    second = pipeline.run(make_config(tmp_path, "c2b", fake_backend()))
    assert (first.run_dir / "train.jsonl").read_bytes() == (
        second.run_dir / "train.jsonl"
    ).read_bytes()


def test_trace_judge_rejects_a_misread_issue(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解说反话的假后端。输出：RunResult。

    预期行为：这些样本被拒，拒绝码是 `E_JUDGE_TRACE`——判别器抓的是模型真会犯的错。
    """
    result = pipeline.run(make_config(tmp_path, "c3", fake_backend(trace_lies_issue=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_trace_judge_rejects_a_misread_urgency(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解把「有多急」说反。输出：RunResult。预期行为：同样被拒。"""
    result = pipeline.run(make_config(tmp_path, "c4", fake_backend(trace_lies_urgency=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_render_rejects_attribute_words_in_the_body(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：把属性词写进正文的假后端。输出：RunResult。

    预期行为：渲染期就拒掉，拒绝码是 `E_ASSEMBLE`（生成阶段的统一拒绝码）——
    这类文本拼进 state 就洗不掉了，而且会与「工单属性」那一行打架。
    """
    result = pipeline.run(make_config(tmp_path, "c5", fake_backend(attrs_body=True)))
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
    wf = routing_workflow.RoutingWorkflow()
    parts: dict[str, set[tuple[tuple[str, str], ...]]] = {}
    for split in wf.split_names():
        cases = wf.build_cases(routing_workflow.RoutingConfig(split=split))
        parts[split] = {tuple(case.as_dict().items()) for case in cases}
    names = list(parts)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert parts[left].isdisjoint(parts[right]), f"{left} 与 {right} 相交"
    union = set().union(*parts.values())
    assert union == {tuple(case.as_dict().items()) for case in features.build_cases(2048)}


def test_every_train_option_appears_in_train() -> None:
    """输入：train 切分的 case 列表。输出：无。

    预期行为：训练侧每一族每一档都到得了——只靠「同分布侧整体」到得了还不够，
    训练集本身要能教到每一个答案。
    """
    labels = [
        case.label()
        for case in routing_workflow.cases_for(routing_workflow.RoutingConfig(split="train"))
    ]
    assert {label.issue for label in labels} == set(features.ISSUES)
    assert {label.urgency for label in labels} == set(features.URGENCIES)
    assert {label.rule for label in labels} == set(features.QUEUE_RULES)
    assert {label.queue for label in labels} == set(features.QUEUES)
    assert {label.priority for label in labels} == set(range(features.PRIORITY_LEVELS))
    assert {label.needs_human for label in labels} == {True, False}


def test_ood_side_only_carries_the_held_out_value() -> None:
    """输入：`test_ood` 的 case 列表。输出：无。预期行为：每一条都带留出的那个取值。"""
    cases = routing_workflow.cases_for(routing_workflow.RoutingConfig(split="test_ood"))
    assert cases
    for case in cases:
        assert case.area in routing_workflow.OOD_AREAS


def test_ood_side_still_reaches_every_option() -> None:
    """输入：`test_ood` 的 case 列表。输出：无。

    预期行为：留出 `area` 不杀任何一档——六族每一档在留出侧都还到得了。
    """
    labels = [
        case.label()
        for case in routing_workflow.cases_for(routing_workflow.RoutingConfig(split="test_ood"))
    ]
    assert {label.issue for label in labels} == set(features.ISSUES)
    assert {label.urgency for label in labels} == set(features.URGENCIES)
    assert {label.rule for label in labels} == set(features.QUEUE_RULES)
    assert {label.queue for label in labels} == set(features.QUEUES)
    assert {label.priority for label in labels} == set(range(features.PRIORITY_LEVELS))
    assert {label.needs_human for label in labels} == {True, False}


def test_cover_limit_forces_full_enumeration() -> None:
    """输入：默认 `cover_limit`。输出：无。

    预期行为：同分布空间（324 组）与全空间（432 组）都在 1024 之下，所以走的是**全枚举**
    ——成对覆盖在这里只会白扔组合。
    """
    id_cases = routing_workflow.cases_for(routing_workflow.RoutingConfig(split="train"))
    assert routing_workflow.RoutingConfig().cover_limit >= 432
    assert len(features.build_cases(routing_workflow.RoutingConfig().cover_limit)) == 432
    assert id_cases


def test_bad_knobs_are_rejected() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话。"""
    with pytest.raises(ValueError):
        routing_workflow.RoutingConfig(split="train", case_limit=-1)
    with pytest.raises(ValueError):
        routing_workflow.RoutingConfig(split="train", phrasing_pool=())
    with pytest.raises(KeyError):
        routing_workflow.RoutingWorkflow().config_for("nope")


def test_render_without_backend_raises() -> None:
    """输入：没注入后端的 workflow。输出：无。

    预期行为：直接报错，绝不悄悄降级成模板——那会让「正文由模型写」这条前提失效。
    """
    wf = routing_workflow.RoutingWorkflow()
    spec = RoutingSpec(tags=features.build_cases(2048)[0], phrasing=0, split="train", seq=0)
    with pytest.raises(RuntimeError):
        wf.render(spec, random.Random(0), "zh")


def test_unknown_judge_name_raises() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：直接报错。"""
    with pytest.raises(KeyError):
        routing_workflow.RoutingWorkflow().make_judges(("nope",), "zh")


def test_workflow_declares_what_it_needs() -> None:
    """输入：workflow 的类属性。输出：无。

    预期行为：`needs` 非空（正文由后端写）、三种题型都声明、判别器名单与默认一致。
    """
    wf = routing_workflow.RoutingWorkflow()
    assert wf.name == "routing"
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
    wf = routing_workflow.RoutingWorkflow()
    assert wf.config_for("train").split == "train"
    assert wf.config_for("test_ood").split == "test_ood"
    tags = features.build_cases(2048)[0]
    state = render.to_state({"body": body_for(tags.as_dict())}, tags, locales.load("zh"))
    parsed = wf.reparse(state)
    assert parsed.body == body_for(tags.as_dict())
    assert parsed.attrs is not None


def test_spec_to_dict_and_task_ids() -> None:
    """输入：一条真值规格。输出：无。

    预期行为：`spec_to_dict` 可 JSON 序列化且字段齐全；渲染 / 反解任务 id 按同一条
    样本可重算（断点续跑按它查进度）。
    """
    tags = features.build_cases(2048)[0]
    spec = RoutingSpec(tags=tags, phrasing=2, split="train", seq=5)
    assert spec_to_dict(spec) == {
        "tags": tags.as_dict(),
        "phrasing": 2,
        "split": "train",
        "seq": 5,
    }
    assert spec.labels == tags.label()
    assert spec.render_task_id == "routing-train-0000005-render"
    assert spec.trace_task_id == "routing-train-0000005-trace"
