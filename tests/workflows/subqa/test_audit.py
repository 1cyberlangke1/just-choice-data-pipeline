"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import jsonio, pipeline
from jc.jsontypes import JsonValue

from tests.workflows.subqa.conftest import FakeBackend
from workflows.subqa import features, render
from workflows.subqa import workflow as subqa_workflow
from workflows.subqa.workflow import SubQaConfig

CASE_LIMIT = 8

#: 审计报告里那五个「不自洽」计数器。
BAD_KEYS = ("content_bad", "flag_bad", "count_bad", "combo_bad", "act_bad")

#: 计数器 → 缺失项列表里那句话的抬头（报告给人看，所以点名的是人话不是内部键）。
BAD_LABELS = {
    "content_bad": "判据表或字幕对认不出来",
    "flag_bad": "逐判据答案与字幕对不自洽",
    "count_bad": "命中条数与字幕对不自洽",
    "combo_bad": "命中组合与字幕对不自洽",
    "act_bad": "处置档位与字幕对不自洽",
}


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="subqa",
            split="train",
            locales=("zh",),
            run_seed=11,
            per_case=1,
            out_root=tmp_path,
            run_id=run_id,
            config_override=SubQaConfig(split="train", case_limit=CASE_LIMIT),
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
    return subqa_workflow.SubQaWorkflow().audit(run_dir, splits, "zh")


def _flip_choice(row: JsonValue, qid: str) -> None:
    """输入：样本行、题号。输出：无。预期行为：把 choice 的答案编号改成另一个选项。"""
    question = row["questions"][qid]
    answer = row["answers"][qid]
    answer["choice"] = (answer["choice"] + 1) % len(question["criteria"])


def test_state_missing_a_structure_key_is_caught(tmp_path: Path) -> None:
    """输入：把一条样本的 `Note:` 那一行抹掉。输出：无。

    预期行为：`content_bad` 报 1——四个结构键是 state 格式的一部分，少一行那几条钉在行上的
    判据就定位不到，整条样本算不出来。
    """
    run_dir = _run(tmp_path, "t-key")
    rows = _rows(run_dir)
    lines = [
        line for line in str(rows[0]["state"]).splitlines() if not line.strip().startswith("Note:")
    ]
    rows[0]["state"] = "\n".join(lines)
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    assert report["train"]["content_bad"] == 1


def test_a_rewritten_translation_is_caught(tmp_path: Path) -> None:
    """输入：把一条**译文里留着标记**的样本的 `Target:` 那一行换成干净英文。输出：无。

    预期行为：`flag_bad` / `count_bad` / `combo_bad` / `act_bad` 里至少有一个报出来——审计走的是
    「从文本重推标签」这条独立路径，把标记擦掉之后重推出来的命中组合就与题面答案对不上了。
    """
    run_dir = _run(tmp_path, "t-rewrite")
    rows = _rows(run_dir)
    target_row = next(
        row
        for row in rows
        if "S2" in features.hit_set(render.parse_state(row["state"]).text or "")
    )
    lines = [
        "Target: We should move the meeting to Thursday, if that still works."
        if line.strip().startswith("Target:")
        else line
        for line in str(target_row["state"]).splitlines()
    ]
    target_row["state"] = "\n".join(lines)
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    counts = report["train"]
    assert counts["flag_bad"] + counts["count_bad"] + counts["combo_bad"] + counts["act_bad"] > 0


def test_unknown_locale_is_rejected(tmp_path: Path) -> None:
    """输入：不存在的语言。输出：无。预期行为：当场抛错，别拖到解析阶段才报。"""
    run_dir = _run(tmp_path, "t-locale")
    with pytest.raises(FileNotFoundError):
        subqa_workflow.SubQaWorkflow().audit(run_dir, ("train",), "nope")
