"""hirecheck 产物审计：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path


from jc import pipeline
from jc.jsontypes import JsonValue
from tests.workflows._scaffold import read_rows as _rows, run_config, write_rows as _write
from tests.workflows.hirecheck.conftest import FakeBackend
from workflows.hirecheck import features, render
from workflows.hirecheck import workflow as hirecheck_workflow
from workflows.hirecheck.workflow import HireCheckConfig, HireCheckWorkflow

CASE_LIMIT = 8


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        run_config(
            tmp_path,
            run_id,
            backend=FakeBackend(),
            workflow="hirecheck",
            seed=13,
            override=HireCheckConfig(split="train", case_limit=CASE_LIMIT),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return HireCheckWorkflow().audit(run_dir, splits, "zh")


def test_a_missing_flag_question_is_caught(tmp_path: Path) -> None:
    """输入：把逐判据那一族的四道题删掉一道。输出：无。

    预期行为：`flag_bad` 报 1——四道题少一道就与四条判据对不上位，答案再多也没有意义。
    """
    run_dir = _run(tmp_path, "t-flag-count")
    rows = _rows(run_dir)
    rows[0]["questions"].pop("flag_4")
    rows[0]["answers"].pop("flag_4")
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    assert report["train"]["flag_bad"] == 1


def test_a_rewritten_line_is_caught(tmp_path: Path) -> None:
    """输入：把一条**计划是「没有工作许可」**的样本的 `permit:` 那一行换成有许可的状态。输出：无。

    预期行为：`flag_bad` / `grade_bad` / `combo_bad` / `act_bad` 里至少有一个报出来——审计走的是
    「从文本重推标签」这条独立路径，把形状擦掉之后重推出来的命中组合就与题面答案对不上了。
    """
    run_dir = _run(tmp_path, "t-rewrite")
    rows = _rows(run_dir)
    target = next(
        row
        for row in rows
        if "H3" in features.hit_set(render.parse_state(row["state"]).text or "")
    )
    target["state"] = "\n".join(
        "permit: citizen" if line.strip().startswith("permit:") else line
        for line in str(target["state"]).splitlines()
    )
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    counts = report["train"]
    assert counts["flag_bad"] + counts["grade_bad"] + counts["combo_bad"] + counts["act_bad"] > 0


def test_an_unknown_family_is_counted(tmp_path: Path) -> None:
    """输入：往样本里塞一道题族名不认识的题。输出：无。

    预期行为：`unknown_families` 点名它——审计靠这个发现「题族改了名」这类漂移，而不是静默把它当成
    一个不认识的题族放过。
    """
    run_dir = _run(tmp_path, "t-family")
    rows = _rows(run_dir)
    rows[0]["questions"]["extra"] = dict(rows[0]["questions"]["combo"])
    rows[0]["answers"]["extra"] = dict(rows[0]["answers"]["combo"])
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["unknown_families"]["extra"] == 1
    assert missing == []


def test_cases_for_is_the_same_list_the_pipeline_sees(tmp_path: Path) -> None:
    """输入：同一条采样参数。输出：无。

    预期行为：`workflow.cases_for` 与管线挑出来的 case 是同一批——审计之外的那几把尺子（题族齐不齐、
    维度跨度）都靠它，两边不一致就会量错东西。
    """
    config = HireCheckConfig(split="train", case_limit=CASE_LIMIT)
    assert len(hirecheck_workflow.cases_for(config)) == CASE_LIMIT
    assert tmp_path.exists()
