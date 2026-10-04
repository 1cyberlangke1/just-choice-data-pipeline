"""完成声明核验 workflow 的端到端测试：假后端跑通整条管线，并验证判别器真的会拦错。"""

from __future__ import annotations

import json
import random
from pathlib import Path

from jc import jsonio, pipeline, workflow as workflow_mod
from jc.backend.types import GenerationBackend
from jc.jsontypes import JsonValue
from workflows.verify import features, locales, render, workflow as verify_workflow
from workflows.verify.features import TagSet
from workflows.verify.spec import VerifySpec
from tests.workflows._scaffold import run_config

CASE_LIMIT = 8

#: 一条「各族答案都不退化」的特征：检查只过了一部分、覆盖完整、还没落盘。
#: 审计那几条用它，答案才好写：partial / fix_the_failing_checks / 1 档 / 没落盘。
CASE = TagSet("bug_fixed", "hedged", "partial_pass", "covered", "uncommitted")
CLEAN = "把订单列表偶发的空指针修掉了"


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="verify",
        split=split,
        seed=7,
        override=verify_workflow.VerifyConfig(split=split, case_limit=CASE_LIMIT),
    )


def cases_of(split: str = "train") -> list[TagSet]:
    """输入：切分名。输出：该切分实际会跑的 case（与产物逐条对得上）。"""
    return verify_workflow.cases_for(
        verify_workflow.VerifyConfig(split=split, case_limit=CASE_LIMIT)
    )


def test_trace_judge_rejects_a_misread_tone(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：反解时把「口气」说反的假后端。输出：RunResult。

    预期行为：同样拒掉——声明要同时承载两维，只对一维不算过关。
    """
    result = pipeline.run(make_config(tmp_path, "s3b", fake_backend(trace_lies_tone=True)))
    assert result.stats["reject_codes"] == {"E_JUDGE_TRACE": CASE_LIMIT}
    assert result.accepted == 0


def test_bind_backend_stores_the_backend(fake_backend: GenerationBackend) -> None:
    """输入：一个后端。输出：无。预期行为：存下来给 `render` 与 `trace` 判别器用。"""
    workflow = verify_workflow.VerifyWorkflow()
    backend = fake_backend()
    workflow.bind_backend(backend)
    assert workflow.backend is backend


def test_make_judges_builds_the_declared_set(fake_backend: GenerationBackend) -> None:
    """输入：声明的判别器名单。输出：无。预期行为：名字与 `judges` 声明逐项对上。

    `trace` 那一档要后端（没有后端时 `make_judges` 当场报错），所以先绑一个假后端。
    """
    workflow = verify_workflow.VerifyWorkflow()
    workflow.bind_backend(fake_backend())
    built = workflow.make_judges(workflow.judges, "zh")
    assert [judge.name for judge in built] == list(workflow.judges)


def test_reparse_and_case_label() -> None:
    """输入：一段文本、一个 case。输出：反解结果、可读描述。

    预期行为：`reparse` 只反解纯代码能查的部分（声明 / 事实行 / 门槛行），非文本输入回空；
    `case_label` 列出全部维度。
    """
    workflow = verify_workflow.VerifyWorkflow()
    state = render.to_state({"claim": CLEAN}, CASE, locales.load("zh"))
    assert workflow.reparse(state).facts == CASE.facts
    assert workflow.reparse({"any": "object"}).claim == ""
    label = workflow.case_label(features.build_cases(48)[0])
    for dim in features.DIMS:
        assert f"{dim}=" in label


def test_task_ids() -> None:
    """输入：一条真值规格。输出：无。预期行为：渲染 / 反解任务 id 按同一条样本可重算。"""
    spec = VerifySpec(tags=CASE, phrasing=1, split="train", seq=3)
    assert spec.render_task_id == "verify-train-0000003-render"
    assert spec.trace_task_id == "verify-train-0000003-trace"


def test_split_sizes_are_sane() -> None:
    """输入：无。输出：无。

    预期行为：同分布侧 160 组按 60/20/20 切开；`test_ood` 是值留出的并集（140 组）。
    比例写成断言是因为「桶边界被改错」会让切分悄悄失衡，而它平时没人看。
    """
    workflow = verify_workflow.VerifyWorkflow()
    counts = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in workflow.split_names()
    }
    assert counts == {"train": 96, "val": 32, "test_iid": 32, "test_ood": 140}


# ---------- 泛化留出 ----------


def test_ood_holds_out_exactly_the_unseen_values() -> None:
    """输入：无。输出：无。

    预期行为：`test_ood` 的每条都落在两个留出取值上，同分布侧一个都不许出现——这才是
    「值留出」，不是「更大的数字」。
    """
    workflow = workflow_mod.load("verify")
    ood = workflow.build_cases(workflow.config_for("test_ood"))
    assert ood
    for case in ood:
        assert (
            case.claim_kind in verify_workflow.OOD_CLAIM_KINDS
            or case.tone in verify_workflow.OOD_TONES
        )
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            assert case.claim_kind not in verify_workflow.OOD_CLAIM_KINDS
            assert case.tone not in verify_workflow.OOD_TONES


def test_id_and_ood_together_cover_every_compatible_case() -> None:
    """输入：四个切分。输出：无。

    预期行为：ID 侧 160 组 + OOD 侧 140 组**恰好**等于全部 300 个相容组合——
    留出不是「少给组合」，是把同一批组合换个切分，一个都不丢。
    """
    workflow = workflow_mod.load("verify")
    seen: set[tuple[tuple[str, str], ...]] = set()
    for split in workflow.split_names():
        for case in workflow.build_cases(workflow.config_for(split)):
            seen.add(tuple(case.as_dict().items()))
    assert seen == {tuple(case.as_dict().items()) for case in features.build_cases(1000)}


def test_id_side_has_no_dead_options() -> None:
    """输入：同分布三个切分的并集。输出：无。

    预期行为：六个题族在 ID 侧覆盖**全部**取值，只有被留出的那两个取值例外。留出某个
    取值时若不留神，会把 ID 侧弄出一个永远取不到的选项（例如留出 `all_pass` 就再也取不到
    `verified`），那是拿「死选项」换「更难 OOD」。
    """
    workflow = verify_workflow.VerifyWorkflow()
    names = ("kind", "tone", "verdict", "gap", "confidence", "committed")
    seen: dict[str, set[str]] = {name: set() for name in names}
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            labels = case.label()
            for name in seen:
                seen[name].add(str(getattr(labels, name)))
    assert seen["kind"] == set(features.CLAIM_KINDS) - verify_workflow.OOD_CLAIM_KINDS
    assert seen["tone"] == set(features.TONES) - verify_workflow.OOD_TONES
    assert seen["verdict"] == set(features.VERDICTS)
    assert seen["gap"] == set(features.GAPS)
    assert seen["confidence"] == {"0", "1", "2", "3"}
    assert seen["committed"] == {"True", "False"}


def test_every_train_option_appears_in_train() -> None:
    """输入：`train` 切分。输出：无。

    预期行为：各族的取值在 **train 一个切分里**就都出现过（只有留出值例外）——
    真跑默认只跑 train，选项要靠 train 自己撑起来。
    """
    workflow = verify_workflow.VerifyWorkflow()
    cases = workflow.build_cases(workflow.config_for("train"))
    seen: dict[str, set[str]] = {name: set() for name in ("kind", "tone", "verdict", "gap", "confidence", "committed")}
    for case in cases:
        labels = case.label()
        for name in seen:
            seen[name].add(str(getattr(labels, name)))
    assert seen["kind"] == set(features.CLAIM_KINDS) - verify_workflow.OOD_CLAIM_KINDS
    assert seen["tone"] == set(features.TONES) - verify_workflow.OOD_TONES
    assert seen["verdict"] == set(features.VERDICTS)
    assert seen["gap"] == set(features.GAPS)
    assert seen["confidence"] == {"0", "1", "2", "3"}
    assert seen["committed"] == {"True", "False"}


def test_held_out_values_are_still_offered_as_options_on_the_id_side() -> None:
    """输入：`train` 侧的一条样本。输出：无。

    预期行为：留出的两个取值仍然出现在 ID 侧的选项表里，只是永远不是答案——OOD 考的
    就是「碰到训练侧没见过的取值还能不能答对」，选项缺席会让那道题变成送分题。
    """
    workflow = verify_workflow.VerifyWorkflow()
    config = workflow.config_for("train")
    case = workflow.build_cases(config)[0]
    spec = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(0), config, case, 0)
    )
    items = {item.family: item for item in workflow.ask(spec, random.Random(0), "zh")}
    assert set(items["kind"].question["criteria"]) == set(features.CLAIM_KINDS)
    assert set(items["tone"].question["criteria"]) == set(features.TONES)
    assert items["kind"].answer != "api_added"
    assert items["tone"].answer != "absolute"


def test_held_out_values_can_still_be_answered_on_the_ood_side() -> None:
    """输入：`test_ood` 侧的一条样本。输出：无。

    预期行为：留出侧真的会出这两个取值当答案（不是把选项摆着好看）。
    """
    workflow = verify_workflow.VerifyWorkflow()
    config = workflow.config_for("test_ood")
    cases = workflow.build_cases(config)
    kinds = {case.claim_kind for case in cases}
    tones = {case.tone for case in cases}
    assert kinds == set(features.CLAIM_KINDS)
    assert tones == set(features.TONES)
    assert any(case.claim_kind in verify_workflow.OOD_CLAIM_KINDS for case in cases)
    assert any(case.tone in verify_workflow.OOD_TONES for case in cases)


# ---------- 审计 ----------


def _row(
    state: str, questions_map: dict[str, JsonValue], answers: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """造一条最小样本行（审计只读 state / questions / answers）。"""
    return {"id": "x", "state": state, "questions": questions_map, "answers": answers}


def _questions(verdict: str, gap: str) -> dict[str, JsonValue]:
    """输入：期望的结论与缺口（各自放在选项表第 0 位）。输出：六个题族的题面。

    预期行为：审计拿「答案编号 → 选项名」换回可读值再比对，所以把期望值放第 0 位，
    下面 `_answers` 统一写编号 0。
    """
    return {
        "kind": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in features.CLAIM_KINDS},
        },
        "tone": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in features.TONES},
        },
        "verdict": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in (verdict, *features.VERDICTS)},
        },
        "gap": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in (gap, *features.GAPS)},
        },
        "confidence": {"type": "score", "instructions": "x", "criteria": ["a", "b", "c", "d"]},
        "committed": {"type": "noul", "instructions": "x", "criteria": {"true": "y", "false": "n"}},
    }


def _answers(confidence: int, committed: bool) -> dict[str, JsonValue]:
    """输入：把握档与落盘两个 noul / score 答案。输出：六个题族的答案（choice 全取编号 0）。"""
    return {
        "kind": {"type": "choice", "choice": 0},
        "tone": {"type": "choice", "choice": 0},
        "verdict": {"type": "choice", "choice": 0},
        "gap": {"type": "choice", "choice": 0},
        "confidence": {"type": "score", "score": confidence},
        "committed": {"type": "noul", "noul": committed},
    }


def _state() -> str:
    """输入：无。输出：一条完整的 state（审计要从它重算四族答案）。"""
    return render.to_state({"claim": CLEAN}, CASE, locales.load("zh"))


def test_audit_passes_a_consistent_artifact(tmp_path: Path) -> None:
    """输入：一条自洽的产物行。输出：(报告, 缺失项)。预期行为：零缺失，四族各记到 1。"""
    labels = CASE.label()
    jsonio.write_jsonl(
        tmp_path / "train.jsonl",
        [_row(_state(), _questions(labels.verdict, labels.gap), _answers(labels.confidence, labels.committed))],
    )
    report, missing = workflow_mod.load("verify").audit(tmp_path, ("train",), "zh")
    assert missing == []
    for family in ("verdict", "gap", "confidence", "committed"):
        assert report["train"][f"{family}_ok"] == 1


def test_audit_reports_a_broken_facts_row(tmp_path: Path) -> None:
    """输入：事实行被删掉的产物行。输出：(报告, 缺失项)。

    预期行为：事实行解析不出来就单独记一笔（`frame_bad`），四族重算整体跳过——
    拿解不出来的事实去重算只会产出一堆假矛盾。
    """
    loc = locales.load("zh")
    broken = _state().replace(f"{render.facts_line(CASE.facts, loc)}\n", "")
    jsonio.write_jsonl(
        tmp_path / "train.jsonl",
        [_row(broken, _questions("partial", "fix_the_failing_checks"), _answers(1, False))],
    )
    report, missing = workflow_mod.load("verify").audit(tmp_path, ("train",), "zh")
    assert report["train"]["frame_bad"] == 1
    assert any("事实行解析不出来" in item for item in missing)


def test_audit_reports_a_gates_mismatch(tmp_path: Path) -> None:
    """输入：门槛行与事实行对不上的产物行。输出：(报告, 缺失项)。

    预期行为：单独记一笔（`gates_bad`）——两行是同一份事实的两个视图，对不上说明渲染
    或组装串味了，这时题面自相矛盾，比「答案错」更早暴露。
    """
    loc = locales.load("zh")
    other = TagSet("bug_fixed", "hedged", "all_pass", "covered", "committed")
    broken = _state().replace(
        render.gates_line(CASE.facts, loc), render.gates_line(other.facts, loc)
    )
    jsonio.write_jsonl(
        tmp_path / "train.jsonl",
        [_row(broken, _questions("partial", "fix_the_failing_checks"), _answers(1, False))],
    )
    report, missing = workflow_mod.load("verify").audit(tmp_path, ("train",), "zh")
    assert report["train"]["gates_bad"] == 1
    assert any("门槛行与事实行对不上" in item for item in missing)
