"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import jsonio, pipeline
from jc.jsontypes import JsonValue

from tests.workflows.alertrisk.conftest import FakeBackend
from workflows.alertrisk import features, render
from workflows.alertrisk import workflow as alertrisk_workflow
from workflows.alertrisk.workflow import AlertRiskConfig, AlertRiskWorkflow

CASE_LIMIT = 8

#: 审计报告里那五个「不自洽」计数器。
BAD_KEYS = ("content_bad", "fact_bad", "grade_bad", "highs_bad", "act_bad")

#: 计数器 → 缺失项列表里那句话的抬头（报告给人看，所以点名的是人话不是内部键）。
BAD_LABELS = {
    "content_bad": "因子表或警报记录认不出来",
    "fact_bad": "逐因子答案与警报记录不自洽",
    "grade_bad": "风险等级与警报记录不自洽",
    "highs_bad": "命中组合与警报记录不自洽",
    "act_bad": "处置档位与警报记录不自洽",
}


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="alertrisk",
            split="train",
            locales=("zh",),
            run_seed=13,
            per_case=1,
            out_root=tmp_path,
            run_id=run_id,
            config_override=AlertRiskConfig(split="train", case_limit=CASE_LIMIT),
            backend=FakeBackend(),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _rows(run_dir: Path) -> list[JsonValue]:
    """输入：run 目录。输出：合格样本行。"""
    return [row for _, row in jsonio.iter_jsonl(run_dir / "train.jsonl")]


def _write(run_dir: Path, rows: list[JsonValue]) -> None:
    """输入：run 目录、样本行。输出：无。预期行为：原样覆盖 train.jsonl（模拟产物被改坏）。"""
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return AlertRiskWorkflow().audit(run_dir, splits, "zh")


def _flip_choice(row: JsonValue, qid: str) -> None:
    """输入：样本行、题号。输出：无。预期行为：把 choice 的答案编号改成另一个选项。"""
    question = row["questions"][qid]
    answer = row["answers"][qid]
    answer["choice"] = (answer["choice"] + 1) % len(question["criteria"])


def test_clean_run_audits_without_missing(tmp_path: Path) -> None:
    """输入：一批干净产物。输出：无。预期行为：四族都出了、五个计数器全零、缺失项为空。"""
    report, missing = _audit(_run(tmp_path))
    counts = report["train"]
    assert missing == []
    assert counts["samples"] == CASE_LIMIT
    for family in ("fact", "grade", "highs", "act"):
        assert counts["families"][family] > 0, family
    assert counts["types"]["noul"] == CASE_LIMIT * len(features.TAG_IDS)
    assert counts["types"]["score"] == CASE_LIMIT
    assert counts["types"]["choice"] == CASE_LIMIT * 2
    for key in BAD_KEYS:
        assert counts[key] == 0, key


@pytest.mark.parametrize(
    ("qid", "counter"),
    [("highs", "highs_bad"), ("act", "act_bad")],
)
def test_choice_answer_mismatch_is_caught(tmp_path: Path, qid: str, counter: str) -> None:
    """输入：把一条 choice 的答案改掉。输出：无。预期行为：对应的计数器报 1，并进缺失项列表。"""
    run_dir = _run(tmp_path, f"t-{qid}")
    rows = _rows(run_dir)
    _flip_choice(rows[0], qid)
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"][counter] == 1
    assert any(BAD_LABELS[counter] in item for item in missing)


def test_score_answer_mismatch_is_caught(tmp_path: Path) -> None:
    """输入：把一条 `grade`（score）的等级下标改掉。输出：无。

    预期行为：`grade_bad` 报 1——审计拿答案下标取回等级描述，再与按文本查矩阵算出来的那一档比。
    """
    run_dir = _run(tmp_path, "t-grade")
    rows = _rows(run_dir)
    answer = rows[0]["answers"]["grade"]
    answer["score"] = (answer["score"] + 1) % len(features.LEVELS)
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["grade_bad"] == 1
    assert any(BAD_LABELS["grade_bad"] in item for item in missing)


def test_fact_answer_mismatch_is_caught(tmp_path: Path) -> None:
    """输入：把一条逐因子 noul 的答案取反。输出：无。预期行为：`fact_bad` 报 1。"""
    run_dir = _run(tmp_path, "t-fact")
    rows = _rows(run_dir)
    answer = rows[0]["answers"]["fact"]
    answer["noul"] = not answer["noul"]
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["fact_bad"] == 1
    assert any(BAD_LABELS["fact_bad"] in item for item in missing)


def test_state_without_a_fact_table_is_caught(tmp_path: Path) -> None:
    """输入：一段认不出抬头的 state。输出：无。预期行为：`content_bad` 报 1。"""
    run_dir = _run(tmp_path, "t-content")
    rows = _rows(run_dir)
    rows[0]["state"] = "这里什么抬头都没有"
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["content_bad"] == 1
    assert any(BAD_LABELS["content_bad"] in item for item in missing)


def test_state_missing_a_structure_key_is_caught(tmp_path: Path) -> None:
    """输入：把一条样本的 `exploit:` 那一行抹掉。输出：无。

    预期行为：`content_bad` 报 1——五个结构键是 state 格式的一部分，少一行那几条钉在行上的判据
    就定位不到，整条样本算不出来。
    """
    run_dir = _run(tmp_path, "t-key")
    rows = _rows(run_dir)
    lines = [
        line
        for line in str(rows[0]["state"]).splitlines()
        if not line.strip().startswith("exploit:")
    ]
    rows[0]["state"] = "\n".join(lines)
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    assert report["train"]["content_bad"] == 1


def test_a_rewritten_line_is_caught(tmp_path: Path) -> None:
    """输入：把一条**计划是「公网可达」**的样本的 `exposure:` 那一行换成内网。输出：无。

    预期行为：`fact_bad` / `grade_bad` / `highs_bad` / `act_bad` 里至少有一个报出来——审计走的是
    「从文本重推标签」这条独立路径，把形状擦掉之后重推出来的命中组合就与题面答案对不上了。
    """
    run_dir = _run(tmp_path, "t-rewrite")
    rows = _rows(run_dir)
    target = next(
        row
        for row in rows
        if "R1" in features.hit_set(render.parse_state(row["state"]).text or "")
    )
    lines = [
        "exposure: reachable=internal -- 只有内网能访问"
        if line.strip().startswith("exposure:")
        else line
        for line in str(target["state"]).splitlines()
    ]
    target["state"] = "\n".join(lines)
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    counts = report["train"]
    assert counts["fact_bad"] + counts["grade_bad"] + counts["highs_bad"] + counts["act_bad"] > 0


def test_unknown_locale_is_rejected(tmp_path: Path) -> None:
    """输入：不存在的语言。输出：无。预期行为：当场抛错，别拖到解析阶段才报。"""
    run_dir = _run(tmp_path, "t-locale")
    with pytest.raises(FileNotFoundError):
        AlertRiskWorkflow().audit(run_dir, ("train",), "nope")


def test_cases_for_is_the_same_list_the_pipeline_sees(tmp_path: Path) -> None:
    """输入：同一条采样参数。输出：无。

    预期行为：`workflow.cases_for` 与管线挑出来的 case 是同一批——审计之外的那几把尺子（题族齐不齐、
    维度跨度）都靠它，两边不一致就会量错东西。
    """
    config = AlertRiskConfig(split="train", case_limit=CASE_LIMIT)
    assert len(alertrisk_workflow.cases_for(config)) == CASE_LIMIT
