"""命令审批 workflow 的端到端测试：假后端跑通整条管线，并验证判别器真的会拦错。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from jc import jsonio, pipeline, workflow as workflow_mod
from jc.backend.types import GenerationBackend, TransportError
from jc.jsontypes import JsonValue
from workflows.shell import features, locales, render, workflow as shell_workflow
from workflows.shell.features import TagSet
from workflows.shell.spec import ShellSpec, spec_to_dict

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="shell",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=shell_workflow.ShellConfig(split=split, case_limit=CASE_LIMIT),
        backend=backend,
    )


def cases_of(split: str = "train") -> list[TagSet]:
    """输入：切分名。输出：该切分实际会跑的 case（与产物逐条对得上）。"""
    return shell_workflow.cases_for(shell_workflow.ShellConfig(split=split, case_limit=CASE_LIMIT))


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: GenerationBackend) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、五个判别器都记到通过；产物能过审计。
    """
    result = pipeline.run(make_config(tmp_path, "s1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {
        "dialect",
        "destructive",
        "elevated",
        "hygiene",
        "trace",
    }
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("shell").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    assert report["train"]["destructive_bad"] == 0
    assert report["train"]["elevated_bad"] == 0


def test_two_runs_are_byte_identical(tmp_path: Path, fake_backend: GenerationBackend) -> None:
    """输入：同种子跑两次。输出：两份产物。预期行为：逐字节一致（命令是字符串，不执行）。"""
    first = pipeline.run(make_config(tmp_path, "s2a", fake_backend()))
    second = pipeline.run(make_config(tmp_path, "s2b", fake_backend()))
    assert (first.run_dir / "train.jsonl").read_bytes() == (
        second.run_dir / "train.jsonl"
    ).read_bytes()


def test_dialect_judge_rejects_bash_written_for_powershell(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：给 PowerShell 样本写 bash 命令的假后端。输出：RunResult。

    预期行为：这些样本被拒，拒绝码是 `E_JUDGE_DIALECT`——判别器抓的是**模型真会犯的错**，
    不是摆设；而外部工具（`git` / `docker`）两种方言写法一样，不受影响。
    """
    result = pipeline.run(make_config(tmp_path, "s3", fake_backend(bash_in_powershell=True)))
    assert result.stats["reject_codes"].get("E_JUDGE_DIALECT", 0) > 0
    assert result.accepted < CASE_LIMIT
    # 被拒的行必须带上 state：不然「模型写了什么」这条唯一的诊断线索就丢了
    assert all(row["state"] for row in result.rejected)


def test_elevated_judge_rejects_a_missing_marker(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：该提权却不写提权写法的假后端。输出：RunResult。

    预期行为：整批里**每一条提权样本**都被拒（其余不提权的照收），拒绝码点名 `elevated`。
    """
    result = pipeline.run(make_config(tmp_path, "s4", fake_backend(skip_elevation=True)))
    elevated_cases = sum(1 for case in cases_of() if case.elevation == "elevated")
    assert elevated_cases > 0
    assert result.stats["reject_codes"] == {"E_JUDGE_ELEVATED": elevated_cases}
    assert result.accepted == CASE_LIMIT - elevated_cases


def test_hygiene_judge_rejects_a_fenced_command(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：把命令包进 Markdown 围栏的假后端。输出：RunResult。

    预期行为：`render.check_command` 在渲染期就抛错，整批记成生成期拒绝（`E_ASSEMBLE`）——
    这种 state 根本没法当「一条命令」审，不该等到判别器。
    """
    result = pipeline.run(make_config(tmp_path, "s5", fake_backend(fence=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}


def test_multi_line_command_is_refused_at_render_time(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：把命令写成两行的假后端。输出：RunResult。预期行为：同样记成 `E_ASSEMBLE`。"""
    result = pipeline.run(make_config(tmp_path, "s6", fake_backend(multi_line=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}


def test_a_polluted_goal_line_is_refused_at_render_time(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：把「用户目标」写成 `用户目标：用户目标：…` 的假后端。输出：RunResult。

    预期行为：整批在渲染期记成 `E_ASSEMBLE`——目标那一行会**原样**拼进 state，
    让它带着字段名进数据集，等于给每条样本塞一句重复标记（实跑里真出现过）。
    """
    result = pipeline.run(make_config(tmp_path, "s11", fake_backend(goal_line="用户目标：看看目录")))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": CASE_LIMIT}
    assert all("用户目标" in row["reason"] for row in result.rejected)


def test_trace_judge_skips_an_endpoint_failure_instead_of_rejecting(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：反解时端点故障（重试与降级都试过）的假后端。输出：RunResult。

    预期行为：命令本身没问题，所以数据照收；`trace` 记成跳过（`judge-skipped.jsonl`）
    而不是拒绝——端点抖动不是数据质量问题。
    """
    result = pipeline.run(
        make_config(tmp_path, "s7", fake_backend(trace_raises=TransportError("网关 504")))
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
    result = pipeline.run(make_config(tmp_path, "s8", fake_backend()))
    manifest = json.loads((result.run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == {"name": "fake", "starting_mode": ""}
    assert manifest["needs"] == ["text"]
    assert manifest["workflow"] == "shell"
    assert manifest["cases"]


def test_stats_report_lists_dimensions_and_judges(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：一次跑。输出：stats 报告。预期行为：四个维度的中文标题与判别器名都在。"""
    result = pipeline.run(make_config(tmp_path, "s9", fake_backend()))
    stats = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    for title in shell_workflow.DIMENSION_TITLES.values():
        assert title in stats
    for name in ("dialect", "destructive", "elevated", "hygiene", "trace"):
        assert name in stats


def test_a_missing_backend_is_refused(tmp_path: Path) -> None:
    """输入：不给后端。输出：ValueError。预期行为：`needs` 非空就立刻报错，不静默降级。"""
    with pytest.raises(ValueError, match="声明需要后端"):
        pipeline.run(make_config(tmp_path, "s10", None))


def test_render_without_backend_raises() -> None:
    """输入：没有注入后端的 workflow。输出：异常。

    预期行为：直接报错，绝不悄悄降级成模板文本（那会让「命令由模型写」这个前提失效）。
    """
    workflow = shell_workflow.ShellWorkflow()
    spec = ShellSpec(
        tags=TagSet("bash", "list_dir", "workdir", "none"), phrasing=0, split="train", seq=0
    )
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_bind_backend_stores_the_backend(fake_backend: GenerationBackend) -> None:
    """输入：一个后端。输出：无。预期行为：存下来给 `render` 与 `trace` 判别器用。"""
    workflow = shell_workflow.ShellWorkflow()
    backend = fake_backend()
    workflow.bind_backend(backend)
    assert workflow.backend is backend


def test_config_for_rejects_unknown_split() -> None:
    """输入：不存在的切分名。输出：KeyError。预期行为：报错并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        shell_workflow.ShellWorkflow().config_for("nope")


def test_make_judges_rejects_unknown_name() -> None:
    """输入：不存在的判别器名。输出：KeyError。预期行为：绝不静默少跑一个判别器。"""
    with pytest.raises(KeyError, match="未知判别器"):
        shell_workflow.ShellWorkflow().make_judges(("nope",), "zh")


def test_reparse_and_case_label() -> None:
    """输入：一段文本、一个 case。输出：反解结果、可读描述。

    预期行为：`reparse` 只反解纯代码能查的部分（命令那一行），非文本输入回空；
    `case_label` 列出全部维度。
    """
    workflow = shell_workflow.ShellWorkflow()
    assert workflow.reparse("拟执行命令：ls -la /home/dev").command == "ls -la /home/dev"
    assert workflow.reparse({"any": "object"}).command == ""
    label = workflow.case_label(features.build_cases(32)[0])
    for dim in features.DIMS:
        assert f"{dim}=" in label


def test_spec_to_dict_is_serializable() -> None:
    """输入：一个规格。输出：字典。预期行为：能直接 `json.dumps`，任务 id 可重算。"""
    case = TagSet("powershell", "restart_service", "system_path", "elevated")
    spec = ShellSpec(tags=case, phrasing=1, split="train", seq=3)
    payload = spec_to_dict(spec)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["tags"] == case.as_dict()
    assert spec.render_task_id == "shell-train-0000003-render"
    assert spec.trace_task_id == "shell-train-0000003-trace"


def test_available_locales_lists_shipped_language_files() -> None:
    """输入：无。输出：语言列表。预期行为：与 locales/ 里的 toml 一一对应，且都读得出来。"""
    workflow = shell_workflow.ShellWorkflow()
    assert workflow.available_locales() == ["en", "zh"]
    for lang in workflow.available_locales():
        assert locales.load(lang)["language_name"]


def test_id_splits_share_no_tag_combination() -> None:
    """输入：无。输出：无。

    预期行为：train / val / test_iid 的 tag 组合两两不相交——切分单位是组合，
    同一个组合不许同时出现在训练与测试里。
    """
    workflow = workflow_mod.load("shell")
    seen: dict[tuple[tuple[str, str], ...], str] = {}
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases, f"{split} 一个 case 都没有"
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in seen, f"组合同时出现在 {seen[key]} 与 {split}：{key}"
            seen[key] = split


def test_ood_holds_out_values_and_id_splits_never_see_them() -> None:
    """输入：无。输出：无。

    预期行为：`test_ood` 的每个组合都带一个留出取值（删表 / 删容器两个动作，或远端位置），
    而同分布切分里一个都不许出现——这才是「值留出」，不是「更大的数字」。
    """
    workflow = workflow_mod.load("shell")
    ood = workflow.build_cases(workflow.config_for("test_ood"))
    assert ood
    for case in ood:
        assert case.goal in shell_workflow.OOD_GOALS or case.target in shell_workflow.OOD_TARGETS
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            assert case.goal not in shell_workflow.OOD_GOALS
            assert case.target not in shell_workflow.OOD_TARGETS


def test_ood_values_are_compatible_with_their_other_dimension() -> None:
    """输入：无。输出：无。

    预期行为：`test_ood` 的每个组合都真的写得出来（不撞 `FORBIDDEN`）——留出取值时若
    不把另一维收窄，覆盖器会在某些取值上找不到可配的动作而直接抛错。
    """
    workflow = workflow_mod.load("shell")
    for case in workflow.build_cases(workflow.config_for("test_ood")):
        assert features.compatible(case), case.as_dict()


# ---------- 审计 ----------


def _row(
    state: str, questions_map: dict[str, JsonValue], answers: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """造一条最小样本行（审计只读 state / questions / answers）。"""
    return {"id": "x", "state": state, "questions": questions_map, "answers": answers}


def _full_questions() -> dict[str, JsonValue]:
    """输入：无。输出：五个题族齐全的题面（审计只数字段名与题型）。"""
    return {
        "verdict": {
            "type": "choice",
            "instructions": "x",
            "criteria": {"allow": "a", "confirm": "b", "deny": "c"},
        },
        "destructive": {"type": "noul", "instructions": "x"},
        "elevated": {"type": "noul", "instructions": "x"},
        "target_owner": {
            "type": "choice",
            "instructions": "x",
            "criteria": {name: name for name in features.TARGETS},
        },
        "blast_radius": {"type": "score", "instructions": "x", "criteria": ["a"] * 5},
    }


def _full_answers(destructive: bool, elevated: bool) -> dict[str, JsonValue]:
    """输入：两个 noul 答案。输出：五个题族的答案（审计只重算 noul 那两条）。"""
    return {
        "verdict": {"type": "choice", "choice": 0},
        "destructive": {"type": "noul", "noul": destructive},
        "elevated": {"type": "noul", "noul": elevated},
        "target_owner": {"type": "choice", "choice": 0},
        "blast_radius": {"type": "score", "score": 0},
    }


def _state(command: str) -> str:
    """输入：一条命令。输出：一条完整的 state（审计要读「拟执行命令」那一行）。"""
    return render.to_state(
        {"goal_line": "把这件事做掉", "command": command, "note": "客观描述"},
        locales.load("zh"),
        "bash",
    )


def test_audit_reports_a_missing_artifact(tmp_path: Path) -> None:
    """输入：空的 run 目录。输出：(报告, 缺失项)。预期行为：点名产物不存在，计数全 0。"""
    report, missing = workflow_mod.load("shell").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


def test_audit_passes_a_consistent_artifact(tmp_path: Path) -> None:
    """输入：一条自洽的产物行（五个题族齐全、答案与命令原文一致）。输出：(报告, 缺失项)。

    预期行为：零缺失，两个自洽计数各记到 1。
    """
    jsonio.write_jsonl(
        tmp_path / "train.jsonl",
        [_row(_state("rm -rf /home/dev/build"), _full_questions(), _full_answers(True, False))],
    )
    report, missing = workflow_mod.load("shell").audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["destructive_ok"] == 1
    assert report["train"]["elevated_ok"] == 1


def test_audit_catches_contradictions_and_unknown_families(tmp_path: Path) -> None:
    """输入：一条自相矛盾 + 带未知题族的产物行。输出：(报告, 缺失项)。

    预期行为：三件事都要报出来——答案与命令原文矛盾、未知题族、题族缺失。
    """
    questions_map = _full_questions()
    questions_map["made_up_family"] = {"type": "noul", "instructions": "y"}
    answers = _full_answers(True, True)  # 只读命令却答「破坏性」「带提权」
    answers["made_up_family"] = {"type": "noul", "noul": True}
    jsonio.write_jsonl(
        tmp_path / "train.jsonl", [_row(_state("ls -la /home/dev"), questions_map, answers)]
    )
    _, missing = workflow_mod.load("shell").audit(tmp_path, ("train",), "zh")
    assert any("destructive 答案与命令原文矛盾" in item for item in missing)
    assert any("elevated 答案与命令原文矛盾" in item for item in missing)
    assert any("未知题族" in item for item in missing)


def _state_in(command: str, lang: str) -> str:
    """输入：一条命令、语言。输出：该语言下的完整 state。

    预期行为：给下面那条「跨语言读法」的测试用——`_state` 只会造 zh 的 state。
    """
    return render.to_state(
        {"goal_line": "把这件事做掉", "command": command, "note": "客观描述"},
        locales.load(lang),
        "bash",
    )


def test_audit_reads_the_command_regardless_of_the_callers_locale(tmp_path: Path) -> None:
    """输入：一条 **en** 产物行，审计时故意传 zh。输出：(报告, 缺失项)。

    预期行为：零缺失——审计读「拟执行命令」那一行必须与判别器同一个口径（`command_of_any`，
    逐语言试），而不是拿调用方给的 `locale` 去认标记。反面实测过：en 产物按 zh 口径读，
    9 条里报出 4 条**假的**「答案与原文矛盾」（判别器读得对，审计读错了）。
    """
    jsonio.write_jsonl(
        tmp_path / "train.jsonl",
        [_row(_state_in("rm -rf /home/dev/build", "en"), _full_questions(), _full_answers(True, False))],
    )
    report, missing = workflow_mod.load("shell").audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["destructive_ok"] == 1
    assert report["train"]["elevated_ok"] == 1
