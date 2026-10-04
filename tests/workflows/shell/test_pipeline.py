"""命令审批 workflow 的端到端测试：假后端跑通整条管线，并验证判别器真的会拦错。"""

from __future__ import annotations

import json
from pathlib import Path

from jc import jsonio, language, pipeline, workflow as workflow_mod
from jc.backend.types import GenerationBackend, TransportError
from jc.jsontypes import JsonValue
from workflows.shell import features, locales, render, workflow as shell_workflow
from workflows.shell.features import TagSet
from workflows.shell.spec import ShellSpec
from tests.workflows._scaffold import run_config

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="shell",
        split=split,
        seed=7,
        override=shell_workflow.ShellConfig(split=split, case_limit=CASE_LIMIT),
    )


def cases_of(split: str = "train") -> list[TagSet]:
    """输入：切分名。输出：该切分实际会跑的 case（与产物逐条对得上）。"""
    return shell_workflow.cases_for(shell_workflow.ShellConfig(split=split, case_limit=CASE_LIMIT))


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


def test_bind_backend_stores_the_backend(fake_backend: GenerationBackend) -> None:
    """输入：一个后端。输出：无。预期行为：存下来给 `render` 与 `trace` 判别器用。"""
    workflow = shell_workflow.ShellWorkflow()
    backend = fake_backend()
    workflow.bind_backend(backend)
    assert workflow.backend is backend


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


def test_task_ids() -> None:
    """输入：一条真值规格。输出：无。预期行为：渲染 / 反解任务 id 按同一条样本可重算。"""
    case = TagSet("powershell", "restart_service", "system_path", "elevated")
    spec = ShellSpec(tags=case, phrasing=1, split="train", seq=3)
    assert spec.render_task_id == "shell-train-0000003-render"
    assert spec.trace_task_id == "shell-train-0000003-trace"


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


# ---------- 语言闸 ----------


def test_language_texts_reports_only_the_two_prose_lines() -> None:
    """输入：一条 zh 产物行。输出：无。

    预期行为：只报**模型写的两格自然语言正文**（「用户目标」「备注」），「拟执行命令」那一行
    不报——命令、路径与标识符按约定保持 ASCII，报上去只会把汉字占比稀释掉、把好数据判成坏数据。
    """
    state = render.to_state(
        {
            "goal_line": "把构建产物清掉",
            "command": "rm -rf /home/dev/build",
            "note": "递归删除该目录，不会动别的东西",
        },
        locales.load("zh"),
        "bash",
    )
    texts = shell_workflow.ShellWorkflow().language_texts({"state": state})
    assert sorted(texts) == ["备注", "用户目标"]
    assert texts["用户目标"] == "把构建产物清掉"
    assert language.sample_problem(texts, ("zh",)) is None


def test_language_texts_catches_prose_in_the_wrong_language() -> None:
    """输入：模型把「用户目标」「备注」两格写成了英文。输出：无。

    预期行为：报上去的两格能被语言尺子抓住——这正是「只报自然语言正文」换来的东西：报整屏
    （含 ASCII 命令行）时，命令会把汉字占比稀释到看不出来。真机那两条写歪的产物
    （用户目标格写成 `DELETE_ACMECORP_PROD_TABLE_ORDERS_2024` / `ls -la "$HOME/..."`）
    就是这么被钉住的。
    """
    state = render.to_state(
        {
            "goal_line": "Delete the production orders table",
            "command": "rm -rf /home/dev/build",
            "note": "This command removes the build directory recursively",
        },
        locales.load("zh"),
        "bash",
    )
    texts = shell_workflow.ShellWorkflow().language_texts({"state": state})
    assert language.sample_problem(texts, ("zh",)) is not None


def test_language_texts_is_empty_when_there_is_no_prose() -> None:
    """输入：只有环境行与命令行的 state、以及非字符串 state。输出：无。

    预期行为：报空表——那把语言尺子对空表不发表意见，正是「没有自然语言正文」该有的形状；
    非字符串（脏产物）也不许抛异常。
    """
    wf = shell_workflow.ShellWorkflow()
    assert wf.language_texts({"state": "环境：bash\nrm -rf /home/dev/build"}) == {}
    assert wf.language_texts({"state": None}) == {}
