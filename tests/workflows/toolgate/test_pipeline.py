"""工具调用门控 workflow 的端到端测试：假后端跑通整条管线，并验证判别器真的会拦错。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from jc import jsonio, pipeline, workflow as workflow_mod
from jc.backend.types import TransportError
from jc.jsontypes import JsonValue
from workflows.toolgate import audit as audit_mod, features, locales, render
from workflows.toolgate import workflow as gate_workflow
from workflows.toolgate.features import TagSet
from workflows.toolgate.spec import ToolGateSpec, spec_to_dict

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
    return pipeline.RunConfig(
        workflow="toolgate",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=gate_workflow.ToolGateConfig(split=split, case_limit=CASE_LIMIT),
        backend=backend,
    )


def cases_of(split: str = "train") -> list[TagSet]:
    """输入：切分名。输出：该切分实际会跑的 case（与产物逐条对得上）。"""
    return gate_workflow.cases_for(
        gate_workflow.ToolGateConfig(split=split, case_limit=CASE_LIMIT)
    )


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，
    而且 `decision` / `overridden` 两族**每一条**都与「规则 + 审批状态」自洽。
    """
    result = pipeline.run(make_config(tmp_path, "s1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("toolgate").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    assert report["train"]["frame_bad"] == 0
    assert report["train"]["rules_bad"] == 0
    assert report["train"]["decision_bad"] == 0
    assert report["train"]["overridden_bad"] == 0


def test_two_runs_are_byte_identical(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：同种子跑两次。输出：两份产物。预期行为：逐字节一致。"""
    first = pipeline.run(make_config(tmp_path, "s2a", fake_backend()))
    second = pipeline.run(make_config(tmp_path, "s2b", fake_backend()))
    assert (first.run_dir / "train.jsonl").read_bytes() == (
        second.run_dir / "train.jsonl"
    ).read_bytes()


def test_trace_judge_rejects_a_misread_tool(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解时把「工具」说反的假后端。输出：RunResult。

    预期行为：这些样本被拒，拒绝码是 `E_JUDGE_TRACE`——判别器抓的是**模型真会犯的错**，
    不是摆设；被拒的行必须带上 state，不然「模型写了什么」这条唯一线索就丢了。
    """
    result = pipeline.run(make_config(tmp_path, "s3", fake_backend(trace_lies=True)))
    assert result.stats["reject_codes"] == {"E_JUDGE_TRACE": CASE_LIMIT}
    assert result.accepted == 0
    assert all(row["state"] for row in result.rejected)
    assert all("注入" in row["reason"] for row in result.rejected)


def test_trace_judge_rejects_a_misread_payload(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解时把「内容性质」说反的假后端。输出：RunResult。

    预期行为：同样拒掉——意图要同时承载两维，只对一维不算过关。
    """
    result = pipeline.run(make_config(tmp_path, "s3b", fake_backend(trace_lies_payload=True)))
    assert result.stats["reject_codes"] == {"E_JUDGE_TRACE": CASE_LIMIT}
    assert result.accepted == 0


def test_a_fenced_intent_is_refused_at_render_time(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：把意图包进 Markdown 围栏的假后端。输出：RunResult。

    预期行为：`render.check_draft` 在渲染期就抛错，整批记成生成期拒绝（`E_ASSEMBLE`）——
    这种 state 的意图段本身就是坏文本，不该等到判别器。
    """
    result = pipeline.run(make_config(tmp_path, "s4", fake_backend(fence_intent=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("围栏" in row["reason"] for row in result.rejected)


def test_a_multi_line_intent_is_refused_at_render_time(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：把意图写成两行的假后端。输出：RunResult。

    预期行为：渲染期就拒（`E_ASSEMBLE`）——意图是**一句话**，多行拼进 state 之后
    `intent_of` 只取第一行，剩下的会成为洗不掉的孤儿文本。
    """
    result = pipeline.run(make_config(tmp_path, "s5", fake_backend(multi_line_intent=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("多行" in row["reason"] for row in result.rejected)


def test_an_empty_intent_is_refused_at_render_time(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：意图是空白的假后端。输出：RunResult。预期行为：渲染期拒，理由点名「空」。"""
    result = pipeline.run(make_config(tmp_path, "s6", fake_backend(empty_intent=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("空的" in row["reason"] for row in result.rejected)


def test_an_intent_carrying_the_scope_is_refused_at_render_time(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：把作用范围写进意图的假后端。输出：RunResult。

    预期行为：渲染期拒——意图与「策略与状态」那一行一旦各说各话，这道题就有歧义了。
    """
    result = pipeline.run(make_config(tmp_path, "s7", fake_backend(scope_intent=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("状态行" in row["reason"] for row in result.rejected)


def test_a_polluted_intent_label_is_refused_at_render_time(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：意图开头又写了一遍「调用意图：」的假后端。输出：RunResult。

    预期行为：渲染期拒——拼进 state 就是一行重复标记（实跑里真出现过这种写法）。
    """
    result = pipeline.run(make_config(tmp_path, "s8", fake_backend(label_intent=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("段落名" in row["reason"] for row in result.rejected)


def test_trace_judge_skips_an_endpoint_failure_instead_of_rejecting(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解时端点故障（重试与降级都试过）的假后端。输出：RunResult。

    预期行为：意图本身没问题，所以数据照收；`trace` 记成跳过（`judge-skipped.jsonl`）
    而不是拒绝——端点抖动不是数据质量问题。
    """
    result = pipeline.run(
        make_config(tmp_path, "s9", fake_backend(trace_raises=TransportError("网关 504")))
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert result.stats["reject_codes"] == {}
    assert result.stats["judges"]["trace"] == {"ok": 0, "fail": 0, "skipped": CASE_LIMIT}
    assert result.stats["judge_skipped"] == CASE_LIMIT
    skipped = [row for _, row in jsonio.iter_jsonl(result.run_dir / "judge-skipped.jsonl")]
    assert {row["judge"] for row in skipped} == {"trace"}
    assert all("端点故障" in row["reason"] for row in skipped)


def test_manifest_records_backend(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端。输出：manifest。

    预期行为：记下后端名字与 needs，绝不记 key——溯源够用，凭据不入库。
    """
    result = pipeline.run(make_config(tmp_path, "s10", fake_backend()))
    manifest = json.loads((result.run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == {"name": "fake", "starting_mode": ""}
    assert manifest["needs"] == ["text"]
    assert manifest["workflow"] == "toolgate"
    assert manifest["cases"]


def test_stats_report_lists_dimensions_and_judges(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：一次跑。输出：stats 报告。预期行为：五个维度的中文标题与判别器名都在。"""
    result = pipeline.run(make_config(tmp_path, "s11", fake_backend()))
    stats = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    for title in gate_workflow.DIMENSION_TITLES.values():
        assert title in stats
    for name in ("hygiene", "frame", "trace"):
        assert name in stats


def test_a_missing_backend_is_refused(tmp_path: Path) -> None:
    """输入：不给后端。输出：ValueError。预期行为：`needs` 非空就立刻报错，不静默降级。"""
    with pytest.raises(ValueError, match="需要后端"):
        pipeline.run(make_config(tmp_path, "s12", None))


def test_render_without_backend_raises() -> None:
    """输入：没有注入后端的 workflow。输出：异常。

    预期行为：直接报错，绝不悄悄降级成模板文本（那会让「意图由模型写」这个前提失效）。
    """
    workflow = gate_workflow.ToolGateWorkflow()
    spec = ToolGateSpec(tags=CASE, phrasing=0, split="train", seq=0)
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_bind_backend_stores_the_backend(fake_backend: type[FakeBackend]) -> None:
    """输入：一个后端。输出：无。预期行为：存下来给 `render` 与 `trace` 判别器用。"""
    workflow = gate_workflow.ToolGateWorkflow()
    backend = fake_backend()
    workflow.bind_backend(backend)
    assert workflow.backend is backend


def test_config_for_rejects_unknown_split() -> None:
    """输入：不存在的切分名。输出：KeyError。预期行为：报错并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        gate_workflow.ToolGateWorkflow().config_for("nope")


def test_config_rejects_a_negative_case_limit() -> None:
    """输入：负数 `case_limit`。输出：无。

    预期行为：报人话——负数会被 `cases[:case_limit]` 悄悄解释成「去掉尾部几条」。
    """
    with pytest.raises(ValueError, match="case_limit"):
        gate_workflow.ToolGateConfig(case_limit=-1)


def test_make_judges_rejects_unknown_name() -> None:
    """输入：不存在的判别器名。输出：KeyError。预期行为：绝不静默少跑一个判别器。"""
    with pytest.raises(KeyError, match="未知判别器"):
        gate_workflow.ToolGateWorkflow().make_judges(("nope",), "zh")


def test_make_judges_builds_the_declared_set() -> None:
    """输入：声明的判别器名单。输出：无。预期行为：名字与 `judges` 声明逐项对上。"""
    workflow = gate_workflow.ToolGateWorkflow()
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


def test_spec_to_dict_is_serializable() -> None:
    """输入：一个规格。输出：字典。预期行为：能直接 `json.dumps`，任务 id 可重算。"""
    spec = ToolGateSpec(tags=CASE, phrasing=1, split="train", seq=3)
    payload = spec_to_dict(spec)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["tags"] == CASE.as_dict()
    assert spec.render_task_id == "toolgate-train-0000003-render"
    assert spec.trace_task_id == "toolgate-train-0000003-trace"


def test_available_locales_lists_shipped_language_files() -> None:
    """输入：无。输出：语言列表。预期行为：与 locales/ 里的 toml 一一对应，且都读得出来。"""
    workflow = gate_workflow.ToolGateWorkflow()
    assert workflow.available_locales() == ["en", "zh"]
    for lang in workflow.available_locales():
        assert locales.load(lang)["lang"] == lang


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


def test_id_and_ood_together_cover_every_compatible_case() -> None:
    """输入：四个切分。输出：无。

    预期行为：ID 侧 100 组 + OOD 侧 86 组**恰好**等于全部 186 个相容组合——
    留出不是「少给组合」，是把同一批组合换个切分，一个都不丢。
    """
    workflow = workflow_mod.load("toolgate")
    seen: set[tuple[tuple[str, str], ...]] = set()
    for split in workflow.split_names():
        for case in workflow.build_cases(workflow.config_for(split)):
            seen.add(tuple(case.as_dict().items()))
    assert seen == {tuple(case.as_dict().items()) for case in features.build_cases(10_000)}


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


def test_audit_reports_a_missing_artifact(tmp_path: Path) -> None:
    """输入：空目录。输出：无。预期行为：报「产物不存在」。"""
    _, missing = gate_workflow.ToolGateWorkflow().audit(tmp_path, ("train",), "zh")
    assert any("产物不存在" in item for item in missing)


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


def test_audit_rejects_an_unknown_locale(tmp_path: Path) -> None:
    """输入：不存在的语言。输出：无。预期行为：当场报错，别拖到解析阶段。"""
    with pytest.raises(FileNotFoundError):
        gate_workflow.ToolGateWorkflow().audit(tmp_path, ("train",), "xx")


def test_cases_of_matches_the_config() -> None:
    """输入：无。输出：无。预期行为：`cases_of` 与 `build_cases` 用同一套参数。"""
    workflow = gate_workflow.ToolGateWorkflow()
    assert cases_of() == workflow.build_cases(
        gate_workflow.ToolGateConfig(split="train", case_limit=CASE_LIMIT)
    )


def test_recorded_maps_every_answer_type() -> None:
    """输入：三种题型的答案。输出：无。

    预期行为：choice 换成选项名（越界给 None）、score / noul 原样返回——审计只靠它把
    产物里的编号换回可读值，换错了后面几条交叉校验就全在比错东西。
    """
    question = {"type": "choice", "criteria": {"a": "A", "b": "B"}}
    assert audit_mod._recorded(question, {"type": "choice", "choice": 1}) == "b"
    assert audit_mod._recorded(question, {"type": "choice", "choice": 9}) is None
    assert audit_mod._recorded({"type": "score"}, {"type": "score", "score": 2}) == 2
    assert audit_mod._recorded({"type": "noul"}, {"type": "noul", "noul": True}) is True


def test_decision_and_overridden_helpers_reject_unknown_rules() -> None:
    """输入：不认识的规则名 / 不是字符串的规则名。输出：无。

    预期行为：一律给 None（推不出来），绝不猜一个处置——猜了会把「串版本」写成「自洽」。
    """
    assert audit_mod.decision_from("made_up", "granted") is None
    assert audit_mod.decision_from(None, "granted") is None
    assert audit_mod.overridden_from("made_up", "granted") is None
    assert audit_mod.overridden_from(7, "granted") is None
