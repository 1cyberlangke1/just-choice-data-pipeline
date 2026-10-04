"""模型路由 workflow 的端到端测试：假后端跑通整条管线，并验证判别器真的会拦错。"""

from __future__ import annotations

import json
import random
from pathlib import Path

from jc import jsonio, pipeline, workflow as workflow_mod
from jc.backend.types import GenerationBackend
from jc.jsontypes import JsonValue
from workflows.modelroute import features, locales, render, workflow as route_workflow
from workflows.modelroute.features import CATALOG, TagSet
from workflows.modelroute.spec import RouteSpec
from tests.workflows._scaffold import run_config

CASE_LIMIT = 8

#: 一条「四族答案都不退化」的特征：选中 cinder-mini（价位 1 档、窗口有富余）、
#: 第一行卡在预算上。审计那几条用它，答案才好写。
CASE = TagSet("code_gen", "small", "batch", "tight", "all_ok")


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="modelroute",
        split=split,
        seed=7,
        override=route_workflow.RouteConfig(split=split, case_limit=CASE_LIMIT),
    )


def cases_of(split: str = "train") -> list[TagSet]:
    """输入：切分名。输出：该切分实际会跑的 case（与产物逐条对得上）。"""
    return route_workflow.cases_for(
        route_workflow.RouteConfig(split=split, case_limit=CASE_LIMIT)
    )


def test_bind_backend_stores_the_backend(fake_backend: GenerationBackend) -> None:
    """输入：一个后端。输出：无。预期行为：存下来给 `render` 与 `trace` 判别器用。"""
    workflow = route_workflow.ModelRouteWorkflow()
    backend = fake_backend()
    workflow.bind_backend(backend)
    assert workflow.backend is backend


def test_make_judges_builds_the_declared_set(fake_backend: GenerationBackend) -> None:
    """输入：声明的判别器名单。输出：无。预期行为：名字与 `judges` 声明逐项对上。

    `trace` 那一档要后端（没有后端时 `make_judges` 当场报错），所以先绑一个假后端。
    """
    workflow = route_workflow.ModelRouteWorkflow()
    workflow.bind_backend(fake_backend())
    built = workflow.make_judges(workflow.judges, "zh")
    assert [judge.name for judge in built] == list(workflow.judges)


def test_reparse_and_case_label() -> None:
    """输入：一段文本、一个 case。输出：反解结果、可读描述。

    预期行为：`reparse` 只反解纯代码能查的部分（请求 / 候选表 / 门槛行），非文本输入回空；
    `case_label` 列出全部维度。
    """
    workflow = route_workflow.ModelRouteWorkflow()
    state = render.to_state({"request": "帮我改代码"}, CASE, locales.load("zh"))
    assert workflow.reparse(state).catalog == CATALOG
    assert workflow.reparse({"any": "object"}).request == ""
    label = workflow.case_label(features.build_cases(48)[0])
    for dim in features.DIMS:
        assert f"{dim}=" in label


def test_task_ids() -> None:
    """输入：一条真值规格。输出：无。预期行为：渲染 / 反解任务 id 按同一条样本可重算。"""
    spec = RouteSpec(tags=CASE, phrasing=1, split="train", seq=3)
    assert spec.render_task_id == "modelroute-train-0000003-render"
    assert spec.trace_task_id == "modelroute-train-0000003-trace"


def test_split_sizes_are_sane() -> None:
    """输入：无。输出：无。

    预期行为：同分布侧全枚举 120 组按 60/20/20 切开；`test_ood` 是值留出的并集。
    比例写成断言是因为「桶边界被改错」会让切分悄悄失衡，而它平时没人看。
    """
    workflow = route_workflow.ModelRouteWorkflow()
    counts = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in workflow.split_names()
    }
    assert counts == {"train": 108, "val": 36, "test_iid": 36, "test_ood": 36}


# ---------- 泛化留出 ----------


def test_ood_holds_out_exactly_the_unseen_kind() -> None:
    """输入：无。输出：无。

    预期行为：`test_ood` 的每条都是「看图」，同分布侧一个都不许出现——这才是「值留出」，
    不是「更大的数字」。
    """
    workflow = workflow_mod.load("modelroute")
    ood = workflow.build_cases(workflow.config_for("test_ood"))
    assert ood
    for case in ood:
        assert case.task_kind in route_workflow.OOD_TASK_KINDS
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            assert case.task_kind not in route_workflow.OOD_TASK_KINDS


def test_id_side_has_no_dead_options() -> None:
    """输入：同分布三个切分的并集。输出：无。

    预期行为：`kind` 之外的四族在 ID 侧覆盖**全部**取值；`kind` 覆盖除留出值以外的全部
    取值。留出某个取值时若不留神，会把 ID 侧弄出一个永远取不到的选项（例如留出
    `interactive` 就再也选不上 `atlas-lite`），那是拿「死选项」换「更难 OOD」。

    `kind` 的 `vision` 是**故意**留出的：它在 ID 侧的选项表里出现，但不该是答案
    （见下一条）。
    """
    workflow = route_workflow.ModelRouteWorkflow()
    seen: dict[str, set[str]] = {name: set() for name in ("kind", "route", "blocked", "cost", "slack")}
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            labels = case.label()
            for name in seen:
                seen[name].add(str(getattr(labels, name)))
    assert seen["kind"] == set(features.TASK_KINDS) - route_workflow.OOD_TASK_KINDS
    assert seen["route"] == set(features.CHOICE_OPTIONS["route"])
    assert seen["blocked"] == set(features.BLOCKED_REASONS)
    assert seen["cost"] == {"0", "1", "2", "3"}
    assert seen["slack"] == {"True", "False"}


def test_every_train_option_appears_in_train() -> None:
    """输入：`train` 切分。输出：无。

    预期行为：各族的取值在 **train 一个切分里**就都出现过（`kind` 同样除留出值外）——
    真跑默认只跑 train，选项要靠 train 自己撑起来。
    """
    workflow = route_workflow.ModelRouteWorkflow()
    cases = workflow.build_cases(workflow.config_for("train"))
    seen: dict[str, set[str]] = {name: set() for name in ("kind", "route", "blocked", "cost", "slack")}
    for case in cases:
        labels = case.label()
        for name in seen:
            seen[name].add(str(getattr(labels, name)))
    assert seen["kind"] == set(features.TASK_KINDS) - route_workflow.OOD_TASK_KINDS
    assert seen["route"] == set(features.CHOICE_OPTIONS["route"])
    assert seen["blocked"] == set(features.BLOCKED_REASONS)
    assert seen["cost"] == {"0", "1", "2", "3"}
    assert seen["slack"] == {"True", "False"}


def test_held_out_kind_is_still_offered_as_an_option_on_the_id_side() -> None:
    """输入：`train` 侧的一条样本。输出：无。

    预期行为：留出的 `vision` 仍然出现在 ID 侧的选项表里，只是永远不是答案——OOD 考的
    就是「碰到训练侧没见过的取值还能不能答对」，选项缺席会让那道题变成送分题。
    """
    workflow = route_workflow.ModelRouteWorkflow()
    cases = workflow.build_cases(workflow.config_for("train"))
    questions_map = {item.family: item for item in workflow.ask(workflow.sample(
        random.Random(0),
        workflow.plan_sample(random.Random(0), workflow.config_for("train"), cases[0], 0),
    ), random.Random(0), "zh")}
    assert set(questions_map["kind"].question["criteria"]) == set(features.TASK_KINDS)
    assert questions_map["kind"].answer != "vision"


# ---------- 审计 ----------


def _row(
    state: str, questions_map: dict[str, JsonValue], answers: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """造一条最小样本行（审计只读 state / questions / answers）。"""
    return {"id": "x", "state": state, "questions": questions_map, "answers": answers}


def _questions(route: str, blocked: str) -> dict[str, JsonValue]:
    """输入：期望的路由与原因（各自放在选项表第 0 位）。输出：五个题族的题面。

    预期行为：审计拿「答案编号 → 选项名」换回可读值再比对，所以把期望值放第 0 位，
    下面 `_answers` 统一写编号 0。
    """
    return {
        "kind": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in features.TASK_KINDS},
        },
        "route": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in (route, *features.CHOICE_OPTIONS["route"])},
        },
        "blocked": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in (blocked, *features.BLOCKED_REASONS)},
        },
        "cost": {"type": "score", "instructions": "x", "criteria": ["a", "b", "c", "d"]},
        "slack": {"type": "noul", "instructions": "x", "criteria": {"true": "y", "false": "n"}},
    }


def _answers(cost: int, slack: bool) -> dict[str, JsonValue]:
    """输入：价格档与富余两个 noul / score 答案。输出：五个题族的答案（choice 全取编号 0）。"""
    return {
        "kind": {"type": "choice", "choice": 0},
        "route": {"type": "choice", "choice": 0},
        "blocked": {"type": "choice", "choice": 0},
        "cost": {"type": "score", "score": cost},
        "slack": {"type": "noul", "noul": slack},
    }


def _state() -> str:
    """输入：无。输出：一条完整的 state（审计要从它重算四族答案）。"""
    return render.to_state({"request": "帮我把这段函数改成异步的"}, CASE, locales.load("zh"))


def test_audit_passes_a_consistent_artifact(tmp_path: Path) -> None:
    """输入：一条自洽的产物行。输出：(报告, 缺失项)。预期行为：零缺失，四族各记到 1。"""
    labels = CASE.label()
    jsonio.write_jsonl(
        tmp_path / "train.jsonl",
        [_row(_state(), _questions(labels.route, labels.blocked), _answers(labels.cost, labels.slack))],
    )
    report, missing = workflow_mod.load("modelroute").audit(tmp_path, ("train",), "zh")
    assert missing == []
    for family in ("route", "blocked", "cost", "slack"):
        assert report["train"][f"{family}_ok"] == 1


def test_audit_reports_a_broken_frame(tmp_path: Path) -> None:
    """输入：候选表被删掉一行的产物行。输出：(报告, 缺失项)。

    预期行为：框架解析不出来就单独记一笔（`frame_bad`），四族重算整体跳过——
    拿解不出来的表去重算只会产出一堆假矛盾。
    """
    loc = locales.load("zh")
    state = _state().replace(f"{render.catalog_line(CATALOG[3], loc)}\n", "")
    jsonio.write_jsonl(
        tmp_path / "train.jsonl",
        [_row(state, _questions("cinder-mini", "budget"), _answers(1, True))],
    )
    report, missing = workflow_mod.load("modelroute").audit(tmp_path, ("train",), "zh")
    assert report["train"]["frame_bad"] == 1
    assert any("门槛行或候选表解析不出来" in item for item in missing)
