"""行级规则检查的产物审计测试：纯代码能重算的每一处，改坏了都要报出来。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import jsonio, pipeline
from jc.jsontypes import JsonValue
from workflows.lint import features, locales, render, workflow as lint_workflow

from tests.workflows.lint.conftest import FakeBackend

CASE_LIMIT = 8


def build_run(tmp_path: Path, run_id: str = "audit-base") -> Path:
    """输入：临时目录。输出：run 目录。预期行为：用假后端跑一小批，产物落盘。"""
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="lint",
            split="train",
            locales=("zh",),
            run_seed=11,
            out_root=tmp_path,
            run_id=run_id,
            config_override=lint_workflow.LintConfig(split="train", case_limit=CASE_LIMIT),
            backend=FakeBackend(),
        )
    )
    assert result.accepted == CASE_LIMIT
    return result.run_dir


def rewrite(run_dir: Path, mutate: object) -> None:
    """输入：run 目录、一个「拿一行、改一行」的回调。输出：无。预期行为：原地重写 train.jsonl。"""
    rows = [row for _, row in jsonio.iter_jsonl(run_dir / "train.jsonl")]
    assert rows
    for row in rows:
        mutate(row)
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)


def audit(run_dir: Path) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录。输出：(报告, 缺失项)。预期行为：只审 train 一个切分。"""
    return lint_workflow.LintWorkflow().audit(run_dir, ("train",), "zh")


def test_clean_run_has_no_findings(tmp_path: Path) -> None:
    """输入：假后端跑出来的一批。输出：无。预期行为：审计一个缺失项都没有、计数全零。"""
    report, missing = audit(build_run(tmp_path))
    assert missing == []
    counts = report["train"]
    assert counts["samples"] == CASE_LIMIT
    for key in (
        "rule_bad",
        "genre_bad",
        "lines_bad",
        "gate_table_bad",
        "line_bad",
        "count_bad",
        "gate_bad",
        "spot_bad",
    ):
        assert counts[key] == 0, key
    assert counts["families"]["line"] == CASE_LIMIT * features.SIZE
    assert counts["families"]["count"] == CASE_LIMIT
    assert counts["families"]["gate"] == CASE_LIMIT
    assert counts["families"]["spot"] > 0


def test_missing_split_is_reported(tmp_path: Path) -> None:
    """输入：只跑了 train 的目录，却要审 val。输出：无。预期行为：报「产物不存在」。"""
    run_dir = build_run(tmp_path, "audit-missing")
    _report, missing = lint_workflow.LintWorkflow().audit(run_dir, ("train", "val"), "zh")
    assert any("val" in item for item in missing)


def test_tampered_line_text_is_caught(tmp_path: Path) -> None:
    """输入：把一条命中行的原文换成干净行。输出：无。

    预期行为：逐行标记与片段不自洽——这正是「构造侧从计划推」与「审计侧从文本重推」打架。
    """
    run_dir = build_run(tmp_path, "audit-line")
    touched = 0

    def mutate(row: dict[str, JsonValue]) -> None:
        nonlocal touched
        state = str(row["state"])
        parsed = render.parse_state(state)
        assert parsed.rule is not None
        for index, text in enumerate(parsed.lines):
            if features.matches(parsed.rule, text):
                row["state"] = state.replace(f"{index + 1}. {text}", f"{index + 1}. import os")
                touched += 1
                return

    rewrite(run_dir, mutate)
    assert touched > 0, "这一批里一条命中行都没有，改不动"
    report, missing = audit(run_dir)
    assert report["train"]["line_bad"] > 0
    assert any("逐行命中标记" in item for item in missing)


def test_tampered_count_answer_is_caught(tmp_path: Path) -> None:
    """输入：把 `count` 的答案编号改一位。输出：无。预期行为：报「命中行数不自洽」。"""
    run_dir = build_run(tmp_path, "audit-count")

    def mutate(row: dict[str, JsonValue]) -> None:
        answer = row["answers"]["count"]
        assert isinstance(answer, dict)
        answer["choice"] = (int(answer["choice"]) + 1) % (features.SIZE + 1)

    rewrite(run_dir, mutate)
    report, missing = audit(run_dir)
    assert report["train"]["count_bad"] == CASE_LIMIT
    assert any("命中行数" in item for item in missing)


def test_tampered_gate_and_spot_answers_are_caught(tmp_path: Path) -> None:
    """输入：把 `gate` / `spot` 的答案编号改一位。输出：无。预期行为：各自报出来。"""
    run_dir = build_run(tmp_path, "audit-gate")

    def mutate(row: dict[str, JsonValue]) -> None:
        gate = row["answers"]["gate"]
        assert isinstance(gate, dict)
        gate["choice"] = (int(gate["choice"]) + 1) % len(features.GATES)
        spot = row["answers"].get("spot")
        if isinstance(spot, dict):
            spot["choice"] = (int(spot["choice"]) + 1) % features.SIZE

    rewrite(run_dir, mutate)
    report, _missing = audit(run_dir)
    assert report["train"]["gate_bad"] == CASE_LIMIT
    assert report["train"]["spot_bad"] > 0


def test_tampered_handling_table_is_caught(tmp_path: Path) -> None:
    """输入：把处置办法表的一行改掉。输出：无。预期行为：报「处置办法表与规范表对不上」。"""
    run_dir = build_run(tmp_path, "audit-table")
    rows = render.gate_rows(locales.load("zh"))

    def mutate(row: dict[str, JsonValue]) -> None:
        row["state"] = str(row["state"]).replace(rows[0], "- 命中 0~2 行 → 拦下")

    rewrite(run_dir, mutate)
    report, missing = audit(run_dir)
    assert report["train"]["gate_table_bad"] == CASE_LIMIT
    assert any("处置办法表" in item for item in missing)


def test_unreadable_rule_name_is_caught(tmp_path: Path) -> None:
    """输入：把规则名换成一句不在规则表里的话。输出：无。预期行为：报「认不出规则名」。"""
    run_dir = build_run(tmp_path, "audit-rule")
    loc = locales.load("zh")

    def mutate(row: dict[str, JsonValue]) -> None:
        parsed = render.parse_state(row["state"])
        assert parsed.rule is not None
        name = render.rule_name(loc, parsed.rule)
        row["state"] = str(row["state"]).replace(name, "看不出来的规则")

    rewrite(run_dir, mutate)
    report, missing = audit(run_dir)
    assert report["train"]["rule_bad"] == CASE_LIMIT
    assert any("规则名" in item for item in missing)


def test_unknown_family_is_reported(tmp_path: Path) -> None:
    """输入：给样本塞一道不属于本 workflow 的题。输出：无。预期行为：记进 unknown_families。"""
    run_dir = build_run(tmp_path, "audit-family")

    def mutate(row: dict[str, JsonValue]) -> None:
        row["questions"]["bogus"] = {"type": "noul", "instructions": "x", "criteria": {}}
        row["answers"]["bogus"] = {"type": "noul", "noul": True}

    rewrite(run_dir, mutate)
    report, _missing = audit(run_dir)
    assert report["train"]["unknown_families"]["bogus"] == CASE_LIMIT


def test_unknown_locale_is_rejected() -> None:
    """输入：不存在的语言。输出：无。预期行为：审计当场报错，不静默跳过。"""
    with pytest.raises(FileNotFoundError):
        lint_workflow.LintWorkflow().audit(Path("dataset"), ("train",), "nope")
