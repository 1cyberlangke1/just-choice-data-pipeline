"""完成声明核验 workflow 的端到端测试：假后端跑通整条管线，并验证判别器真的会拦错。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from jc import jsonio, pipeline, workflow as workflow_mod
from jc.backend.types import GenerationBackend, TransportError
from jc.jsontypes import JsonValue
from workflows.verify import features, locales, render, workflow as verify_workflow
from workflows.verify.features import TagSet
from workflows.verify.spec import VerifySpec, spec_to_dict

CASE_LIMIT = 8

#: 一条「各族答案都不退化」的特征：检查只过了一部分、覆盖完整、还没落盘。
#: 审计那几条用它，答案才好写：partial / fix_the_failing_checks / 1 档 / 没落盘。
CASE = TagSet("bug_fixed", "hedged", "partial_pass", "covered", "uncommitted")
CLEAN = "把订单列表偶发的空指针修掉了"


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="verify",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=verify_workflow.VerifyConfig(split=split, case_limit=CASE_LIMIT),
        backend=backend,
    )


def cases_of(split: str = "train") -> list[TagSet]:
    """输入：切分名。输出：该切分实际会跑的 case（与产物逐条对得上）。"""
    return verify_workflow.cases_for(
        verify_workflow.VerifyConfig(split=split, case_limit=CASE_LIMIT)
    )


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: GenerationBackend) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，
    而且四个可重算的题族**每一条**都能从 state 里那两行重算出来。
    """
    result = pipeline.run(make_config(tmp_path, "s1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("verify").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    assert report["train"]["frame_bad"] == 0
    assert report["train"]["gates_bad"] == 0
    for family in ("verdict", "gap", "confidence", "committed"):
        assert report["train"][f"{family}_bad"] == 0
        assert report["train"][f"{family}_ok"] == CASE_LIMIT


def test_two_runs_are_byte_identical(tmp_path: Path, fake_backend: GenerationBackend) -> None:
    """输入：同种子跑两次。输出：两份产物。预期行为：逐字节一致。"""
    first = pipeline.run(make_config(tmp_path, "s2a", fake_backend()))
    second = pipeline.run(make_config(tmp_path, "s2b", fake_backend()))
    assert (first.run_dir / "train.jsonl").read_bytes() == (
        second.run_dir / "train.jsonl"
    ).read_bytes()


def test_trace_judge_rejects_a_misread_claim(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：反解时把「声称做完的事」说反的假后端。输出：RunResult。

    预期行为：这些样本被拒，拒绝码是 `E_JUDGE_TRACE`——判别器抓的是**模型真会犯的错**，
    不是摆设；被拒的行必须带上 state，不然「模型写了什么」这条唯一线索就丢了。
    """
    result = pipeline.run(make_config(tmp_path, "s3", fake_backend(trace_lies=True)))
    assert result.stats["reject_codes"] == {"E_JUDGE_TRACE": CASE_LIMIT}
    assert result.accepted == 0
    assert all(row["state"] for row in result.rejected)
    assert all("注入" in row["reason"] for row in result.rejected)


def test_trace_judge_rejects_a_misread_tone(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：反解时把「口气」说反的假后端。输出：RunResult。

    预期行为：同样拒掉——声明要同时承载两维，只对一维不算过关。
    """
    result = pipeline.run(make_config(tmp_path, "s3b", fake_backend(trace_lies_tone=True)))
    assert result.stats["reject_codes"] == {"E_JUDGE_TRACE": CASE_LIMIT}
    assert result.accepted == 0


def test_a_fenced_claim_is_refused_at_render_time(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：把声明包进 Markdown 围栏的假后端。输出：RunResult。

    预期行为：`render.check_draft` 在渲染期就抛错，整批记成生成期拒绝（`E_ASSEMBLE`）——
    这种 state 的声明段本身就是坏文本，不该等到判别器。
    """
    result = pipeline.run(make_config(tmp_path, "s4", fake_backend(fence_claim=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("围栏" in row["reason"] for row in result.rejected)


def test_a_multi_line_claim_is_refused_at_render_time(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：把声明写成两行的假后端。输出：RunResult。

    预期行为：渲染期就拒（`E_ASSEMBLE`）——声明是**一句话**，多行拼进 state 之后
    `claim_of` 只取第一行，剩下的会成为洗不掉的孤儿文本。
    """
    result = pipeline.run(make_config(tmp_path, "s5", fake_backend(multi_line_claim=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("多行" in row["reason"] for row in result.rejected)


def test_an_empty_claim_is_refused_at_render_time(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：声明是空白的假后端。输出：RunResult。预期行为：渲染期拒，理由点名「空」。"""
    result = pipeline.run(make_config(tmp_path, "s6", fake_backend(empty_claim=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("空的" in row["reason"] for row in result.rejected)


def test_a_claim_carrying_evidence_is_refused_at_render_time(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：把证据写进声明的假后端。输出：RunResult。

    预期行为：渲染期拒——声明与代码渲染的事实行一旦各说各话，这道题就有歧义了。
    """
    result = pipeline.run(make_config(tmp_path, "s7", fake_backend(evidence_claim=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("证据" in row["reason"] for row in result.rejected)


def test_a_polluted_claim_label_is_refused_at_render_time(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：声明开头又写了一遍「声明：」的假后端。输出：RunResult。

    预期行为：渲染期拒——拼进 state 就是一行重复标记（实跑里真出现过这种写法）。
    """
    result = pipeline.run(make_config(tmp_path, "s8", fake_backend(label_claim=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("声明" in row["reason"] for row in result.rejected)


def test_trace_judge_skips_an_endpoint_failure_instead_of_rejecting(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：反解时端点故障（重试与降级都试过）的假后端。输出：RunResult。

    预期行为：声明本身没问题，所以数据照收；`trace` 记成跳过（`judge-skipped.jsonl`）
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


def test_manifest_records_backend(tmp_path: Path, fake_backend: GenerationBackend) -> None:
    """输入：假后端。输出：manifest。

    预期行为：记下后端名字与 needs，绝不记 key——溯源够用，凭据不入库。
    """
    result = pipeline.run(make_config(tmp_path, "s10", fake_backend()))
    manifest = json.loads((result.run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == {"name": "fake", "starting_mode": ""}
    assert manifest["needs"] == ["text"]
    assert manifest["workflow"] == "verify"
    assert manifest["cases"]


def test_stats_report_lists_dimensions_and_judges(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：一次跑。输出：stats 报告。预期行为：五个维度的中文标题与判别器名都在。"""
    result = pipeline.run(make_config(tmp_path, "s11", fake_backend()))
    stats = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    for title in verify_workflow.DIMENSION_TITLES.values():
        assert title in stats
    for name in ("hygiene", "frame", "trace"):
        assert name in stats


def test_a_missing_backend_is_refused(tmp_path: Path) -> None:
    """输入：不给后端。输出：ValueError。预期行为：`needs` 非空就立刻报错，不静默降级。"""
    with pytest.raises(ValueError, match="声明需要后端"):
        pipeline.run(make_config(tmp_path, "s12", None))


def test_render_without_backend_raises() -> None:
    """输入：没有注入后端的 workflow。输出：异常。

    预期行为：直接报错，绝不悄悄降级成模板文本（那会让「声明由模型写」这个前提失效）。
    """
    workflow = verify_workflow.VerifyWorkflow()
    spec = VerifySpec(tags=CASE, phrasing=0, split="train", seq=0)
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_bind_backend_stores_the_backend(fake_backend: GenerationBackend) -> None:
    """输入：一个后端。输出：无。预期行为：存下来给 `render` 与 `trace` 判别器用。"""
    workflow = verify_workflow.VerifyWorkflow()
    backend = fake_backend()
    workflow.bind_backend(backend)
    assert workflow.backend is backend


def test_config_for_rejects_unknown_split() -> None:
    """输入：不存在的切分名。输出：KeyError。预期行为：报错并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        verify_workflow.VerifyWorkflow().config_for("nope")


def test_config_rejects_a_negative_case_limit() -> None:
    """输入：负数 `case_limit`。输出：无。

    预期行为：报人话——以前负数会被 `cases[:case_limit]` 悄悄解释成「去掉尾部几条」。
    """
    with pytest.raises(ValueError, match="case_limit"):
        verify_workflow.VerifyConfig(case_limit=-1)


def test_make_judges_rejects_unknown_name() -> None:
    """输入：不存在的判别器名。输出：KeyError。预期行为：绝不静默少跑一个判别器。"""
    with pytest.raises(KeyError, match="未知判别器"):
        verify_workflow.VerifyWorkflow().make_judges(("nope",), "zh")


def test_make_judges_builds_the_declared_set() -> None:
    """输入：声明的判别器名单。输出：无。预期行为：名字与 `judges` 声明逐项对上。"""
    workflow = verify_workflow.VerifyWorkflow()
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


def test_spec_to_dict_is_serializable() -> None:
    """输入：一个规格。输出：字典。预期行为：能直接 `json.dumps`，任务 id 可重算。"""
    spec = VerifySpec(tags=CASE, phrasing=1, split="train", seq=3)
    payload = spec_to_dict(spec)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["tags"] == CASE.as_dict()
    assert spec.render_task_id == "verify-train-0000003-render"
    assert spec.trace_task_id == "verify-train-0000003-trace"


def test_available_locales_lists_shipped_language_files() -> None:
    """输入：无。输出：语言列表。预期行为：与 locales/ 里的 toml 一一对应，且都读得出来。"""
    workflow = verify_workflow.VerifyWorkflow()
    assert workflow.available_locales() == ["en", "zh"]
    for lang in workflow.available_locales():
        assert locales.load(lang)["lang"] == lang


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


def test_audit_reports_a_missing_artifact(tmp_path: Path) -> None:
    """输入：空的 run 目录。输出：(报告, 缺失项)。预期行为：点名产物不存在，计数全 0。"""
    report, missing = workflow_mod.load("verify").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


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


def test_audit_catches_contradictions_and_unknown_families(tmp_path: Path) -> None:
    """输入：一条四族答案都与 state 矛盾、且带未知题族的产物行。输出：(报告, 缺失项)。

    预期行为：四族矛盾各报一条、未知题族报一条、题族缺失也要报——审计的用处就是
    「文本与标签脱节」当场暴露，而不是靠人肉抽查。
    """
    questions_map = _questions("verified", "nothing_missing")
    questions_map["made_up_family"] = {"type": "noul", "instructions": "y"}
    answers = _answers(3, True)
    answers["made_up_family"] = {"type": "noul", "noul": True}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [_row(_state(), questions_map, answers)])
    _, missing = workflow_mod.load("verify").audit(tmp_path, ("train",), "zh")
    for family in ("verdict", "gap", "confidence", "committed"):
        assert any(f"{family} 答案与文本重算的结果矛盾" in item for item in missing)
    assert any("未知题族" in item for item in missing)


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