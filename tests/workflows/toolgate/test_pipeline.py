"""工具调用门控 workflow 的端到端测试：假后端跑通整条管线，并验证判别器真的会拦错。"""

from __future__ import annotations

import json
import random
from pathlib import Path

from jc import pipeline, workflow as workflow_mod
from jc.jsontypes import JsonValue
from workflows.toolgate import audit as audit_mod, features, locales, render
from workflows.toolgate import workflow as gate_workflow
from workflows.toolgate.features import TagSet
from workflows.toolgate.spec import ToolGateSpec

from tests.workflows._scaffold import run_config
from tests.workflows.toolgate.conftest import FakeBackend

CASE_LIMIT = 8

#: 一条「各族答案都不退化」的特征：外网 + 敏感内容 + 居中档 + 已拿审批。
#: 审计那几条用它，答案才好写：命中 sensitive_payload、最终放行、审批起了作用、风险 3 档。
CASE = TagSet(tool="http_request", reach="external", payload="sensitive", policy="balanced", approval="granted")
CLEAN = "调一下库存服务的查询接口，看看还有多少货"


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="toolgate",
        split=split,
        seed=7,
        override=gate_workflow.ToolGateConfig(split=split, case_limit=CASE_LIMIT),
    )


def cases_of(split: str = "train") -> list[TagSet]:
    """输入：切分名。输出：该切分实际会跑的 case（与产物逐条对得上）。"""
    return gate_workflow.cases_for(
        gate_workflow.ToolGateConfig(split=split, case_limit=CASE_LIMIT)
    )


def test_trace_judge_rejects_a_misread_payload(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解时把「内容性质」说反的假后端。输出：RunResult。

    预期行为：同样拒掉——意图要同时承载两维，只对一维不算过关。
    """
    result = pipeline.run(make_config(tmp_path, "s3b", fake_backend(trace_lies_payload=True)))
    assert result.stats["reject_codes"] == {"E_JUDGE_TRACE": CASE_LIMIT}
    assert result.accepted == 0


def test_bind_backend_stores_the_backend(fake_backend: type[FakeBackend]) -> None:
    """输入：一个后端。输出：无。预期行为：存下来给 `render` 与 `trace` 判别器用。"""
    workflow = gate_workflow.ToolGateWorkflow()
    backend = fake_backend()
    workflow.bind_backend(backend)
    assert workflow.backend is backend


def test_make_judges_builds_the_declared_set(fake_backend: type[FakeBackend]) -> None:
    """输入：声明的判别器名单。输出：无。预期行为：名字与 `judges` 声明逐项对上。

    `trace` 那一档要后端（没有后端时 `make_judges` 当场报错），所以先绑一个假后端。
    """
    workflow = gate_workflow.ToolGateWorkflow()
    workflow.bind_backend(fake_backend())
    built = workflow.make_judges(workflow.judges, "zh")
    assert [judge.name for judge in built] == list(workflow.judges)


def test_reparse_and_case_label() -> None:
    """输入：一段文本、一个 case。输出：反解结果、可读描述。

    预期行为：`reparse` 只反解纯代码能查的部分（意图 / 状态行 / 策略表），非文本输入回空；
    `case_label` 列出全部维度。
    """
    workflow = gate_workflow.ToolGateWorkflow()
    state = render.to_state({"intent": CLEAN}, CASE, locales.load("zh"))
    assert workflow.reparse(state).call == render.CallState("external", "balanced", "granted")
    assert workflow.reparse({"any": "object"}).call is None
    label = workflow.case_label(features.build_cases(10_000)[0])
    for dim in features.DIMS:
        assert f"{dim}=" in label


def test_task_ids() -> None:
    """输入：一条真值规格。输出：无。预期行为：渲染 / 反解任务 id 按同一条样本可重算。"""
    spec = ToolGateSpec(tags=CASE, phrasing=1, split="train", seq=3)
    assert spec.render_task_id == "toolgate-train-0000003-render"
    assert spec.trace_task_id == "toolgate-train-0000003-trace"


def test_split_sizes_are_sane() -> None:
    """输入：无。输出：无。

    预期行为：同分布侧 100 组按 60/20/20 切开；`test_ood` 是值留出的并集（86 组）。
    比例写成断言是因为「桶边界被改错」会让切分悄悄失衡，而它平时没人看。
    """
    workflow = gate_workflow.ToolGateWorkflow()
    counts = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in workflow.split_names()
    }
    assert counts == {"train": 60, "val": 20, "test_iid": 20, "test_ood": 86}


# ---------- 泛化留出 ----------


def test_ood_holds_out_exactly_the_unseen_values() -> None:
    """输入：无。输出：无。

    预期行为：`test_ood` 的每条都落在两个留出取值上，同分布侧一个都不许出现——这才是
    「值留出」，不是「更大的数字」。
    """
    workflow = workflow_mod.load("toolgate")
    ood = workflow.build_cases(workflow.config_for("test_ood"))
    assert ood
    for case in ood:
        assert (
            case.tool in gate_workflow.OOD_TOOLS or case.policy in gate_workflow.OOD_POLICIES
        )
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            assert case.tool not in gate_workflow.OOD_TOOLS
            assert case.policy not in gate_workflow.OOD_POLICIES


def test_id_side_has_no_dead_options() -> None:
    """输入：同分布三个切分的并集。输出：无。

    预期行为：六个题族在 ID 侧覆盖**全部**取值，只有被留出的那两个取值例外。留出某个
    取值时若不留神，会把 ID 侧弄出一个永远取不到的选项（例如留出 `payload=harmful`
    就再也命中不了策略表第一条），那是拿「死选项」换「更难 OOD」。
    """
    workflow = gate_workflow.ToolGateWorkflow()
    names = ("tool", "payload", "rule", "decision", "overridden", "severity")
    seen: dict[str, set[str]] = {name: set() for name in names}
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            labels = case.label()
            for name in seen:
                seen[name].add(str(getattr(labels, name)))
    assert seen["tool"] == set(features.TOOLS) - gate_workflow.OOD_TOOLS
    assert seen["payload"] == set(features.PAYLOADS)
    assert seen["rule"] == set(features.RULES)
    assert seen["decision"] == set(features.DECISIONS)
    assert seen["overridden"] == {"True", "False"}
    assert seen["severity"] == {"0", "1", "2", "3"}


def test_every_train_option_appears_in_train() -> None:
    """输入：`train` 切分。输出：无。

    预期行为：各族的取值在 **train 一个切分里**就都出现过（只有留出值例外）——
    真跑默认只跑 train，选项要靠 train 自己撑起来。
    """
    workflow = gate_workflow.ToolGateWorkflow()
    cases = workflow.build_cases(workflow.config_for("train"))
    names = ("tool", "payload", "rule", "decision", "overridden", "severity")
    seen: dict[str, set[str]] = {name: set() for name in names}
    for case in cases:
        labels = case.label()
        for name in seen:
            seen[name].add(str(getattr(labels, name)))
    assert seen["tool"] == set(features.TOOLS) - gate_workflow.OOD_TOOLS
    assert seen["payload"] == set(features.PAYLOADS)
    assert seen["rule"] == set(features.RULES)
    assert seen["decision"] == set(features.DECISIONS)
    assert seen["overridden"] == {"True", "False"}
    assert seen["severity"] == {"0", "1", "2", "3"}


def test_held_out_values_are_still_offered_as_options_on_the_id_side() -> None:
    """输入：`train` 侧的一条样本。输出：无。

    预期行为：留出的两个取值仍然出现在 ID 侧的选项表里，只是永远不是答案——OOD 考的
    就是「碰到训练侧没见过的取值还能不能答对」，选项缺席会让那道题变成送分题。
    """
    workflow = gate_workflow.ToolGateWorkflow()
    config = workflow.config_for("train")
    case = workflow.build_cases(config)[0]
    spec = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(0), config, case, 0)
    )
    items = {item.family: item for item in workflow.ask(spec, random.Random(0), "zh")}
    assert set(items["tool"].question["criteria"]) == set(features.TOOLS)
    assert items["tool"].answer != "send_message"


def test_held_out_values_can_still_be_answered_on_the_ood_side() -> None:
    """输入：`test_ood` 侧的一条样本。输出：无。

    预期行为：留出侧真的会出这两个取值当答案（不是把选项摆着好看）。
    """
    workflow = gate_workflow.ToolGateWorkflow()
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    tools = {case.tool for case in cases}
    policies = {case.policy for case in cases}
    assert tools == set(features.TOOLS)
    assert policies == set(features.POLICIES)
    assert any(case.tool in gate_workflow.OOD_TOOLS for case in cases)
    assert any(case.policy in gate_workflow.OOD_POLICIES for case in cases)


# ---------- 审计 ----------


def _row(
    state: str, questions_map: dict[str, JsonValue], answers: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """造一条最小样本行（审计只读 state / questions / answers）。"""
    return {"id": "x", "state": state, "questions": questions_map, "answers": answers}


def _questions(rule: str, decision: str) -> dict[str, JsonValue]:
    """输入：期望的命中规则与处置（各自放在选项表第 0 位）。输出：六个题族的题面。

    预期行为：审计拿「答案编号 → 选项名」换回可读值再比对，所以把期望值放第 0 位，
    下面 `_answers` 统一写编号 0。
    """
    return {
        "tool": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in features.TOOLS},
        },
        "payload": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in features.PAYLOADS},
        },
        "rule": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in (rule, *features.RULES)},
        },
        "decision": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in (decision, *features.DECISIONS)},
        },
        "overridden": {
            "type": "noul",
            "instructions": "x",
            "criteria": {"true": "t", "false": "f"},
        },
        "severity": {"type": "score", "instructions": "x", "criteria": ["0", "1", "2", "3"]},
    }


def _answers(overridden: bool, severity: int = 0) -> dict[str, JsonValue]:
    """输入：期望的 overridden 与风险档。输出：六个题族的答案（choice 一律第 0 号）。"""
    return {
        "tool": {"type": "choice", "choice": 0},
        "payload": {"type": "choice", "choice": 0},
        "rule": {"type": "choice", "choice": 0},
        "decision": {"type": "choice", "choice": 0},
        "overridden": {"type": "noul", "noul": overridden},
        "severity": {"type": "score", "score": severity},
    }


def _write(run_dir: Path, rows: list[dict[str, JsonValue]]) -> None:
    """把样本行写成 `train.jsonl`。预期行为：一行一个 JSON 对象。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    (run_dir / "train.jsonl").write_text(text + "\n", encoding="utf-8")


def state_of(tags: TagSet = CASE) -> str:
    """输入：特征。输出：渲染好的 state。预期行为：走公开入口拼装。"""
    return render.to_state({"intent": CLEAN}, tags, locales.load("zh"))


def test_audit_passes_a_consistent_artifact(tmp_path: Path) -> None:
    """输入：一条自洽的产物。输出：无。

    预期行为：六个题族齐全、状态行与策略表都在、处置与 overridden 都与「规则 + 审批」
    自洽——审计一条缺失项都不报。
    """
    _write(tmp_path, [_row(state_of(), _questions("sensitive_payload", "allow"), _answers(True, 3))])
    report, missing = gate_workflow.ToolGateWorkflow().audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["decision_bad"] == 0
    assert report["train"]["overridden_bad"] == 0
    assert report["train"]["rules_bad"] == 0


def test_audit_catches_contradictions_and_unknown_families(tmp_path: Path) -> None:
    """输入：处置与规则打架、多了一个陌生题族的产物。输出：无。

    预期行为：`decision_bad` 与 `overridden_bad` 各记一笔、陌生题族被点名——
    审计不是走过场，它真的会把这些标出来。
    """
    questions = _questions("none", "deny")
    questions["made_up"] = {"type": "noul", "instructions": "x", "criteria": {"true": "t", "false": "f"}}
    # 命中 `none` 时处置只能是放行、审批也起不了作用；这里两处都故意写反。
    answers = _answers(True)
    answers["made_up"] = {"type": "noul", "noul": False}
    _write(tmp_path, [_row(state_of(), questions, answers)])
    report, missing = gate_workflow.ToolGateWorkflow().audit(tmp_path, ("train",), "zh")
    assert report["train"]["decision_bad"] == 1
    assert report["train"]["overridden_bad"] == 1
    assert report["train"]["unknown_families"]["made_up"] == 1
    assert any("不自洽" in item for item in missing)


def test_audit_reports_a_broken_state_row(tmp_path: Path) -> None:
    """输入：状态行被换成不认识的词元。输出：无。预期行为：`frame_bad` 记一笔。"""
    broken = state_of().replace("external | balanced | granted", "nowhere | balanced | granted")
    _write(tmp_path, [_row(broken, _questions("sensitive_payload", "allow"), _answers(True, 3))])
    report, missing = gate_workflow.ToolGateWorkflow().audit(tmp_path, ("train",), "zh")
    assert report["train"]["frame_bad"] == 1
    assert any("状态行" in item for item in missing)


def test_audit_reports_a_tampered_rule_table(tmp_path: Path) -> None:
    """输入：策略表少了一行。输出：无。预期行为：`rules_bad` 记一笔。"""
    loc = locales.load("zh")
    broken = state_of().replace(render.rule_lines(loc)[0] + "\n", "")
    _write(tmp_path, [_row(broken, _questions("sensitive_payload", "allow"), _answers(True, 3))])
    report, missing = gate_workflow.ToolGateWorkflow().audit(tmp_path, ("train",), "zh")
    assert report["train"]["rules_bad"] == 1
    assert any("策略表" in item for item in missing)


def test_audit_flags_a_family_that_never_shows_up(tmp_path: Path) -> None:
    """输入：缺了 `severity` 那一族的产物。输出：无。预期行为：报「一条都没出」。"""
    questions = _questions("sensitive_payload", "allow")
    answers = _answers(True, 3)
    questions.pop("severity")
    answers.pop("severity")
    _write(tmp_path, [_row(state_of(), questions, answers)])
    _, missing = gate_workflow.ToolGateWorkflow().audit(tmp_path, ("train",), "zh")
    assert any("severity" in item for item in missing)


def test_cases_of_matches_the_config() -> None:
    """输入：无。输出：无。预期行为：`cases_of` 与 `build_cases` 用同一套参数。"""
    workflow = gate_workflow.ToolGateWorkflow()
    assert cases_of() == workflow.build_cases(
        gate_workflow.ToolGateConfig(split="train", case_limit=CASE_LIMIT)
    )


def test_decision_and_overridden_helpers_reject_unknown_rules() -> None:
    """输入：不认识的规则名 / 不是字符串的规则名。输出：无。

    预期行为：一律给 None（推不出来），绝不猜一个处置——猜了会把「串版本」写成「自洽」。
    """
    assert audit_mod.decision_from("made_up", "granted") is None
    assert audit_mod.decision_from(None, "granted") is None
    assert audit_mod.overridden_from("made_up", "granted") is None
    assert audit_mod.overridden_from(7, "granted") is None
