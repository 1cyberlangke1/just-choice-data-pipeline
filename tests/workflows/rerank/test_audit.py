"""审计测试：产物被改坏时，对应的那一格计数必须真的响。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jc import pipeline
from jc.jsontypes import JsonValue

from workflows.rerank import audit as audit_mod, workflow as rerank_workflow
from tests.workflows.rerank.conftest import FakeBackend

CASE_LIMIT = 8


def make_config(tmp_path: Path, run_id: str, backend: object) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。

    预期行为：**关掉语言闸**——本包的假后端只写英文（夹具局限，理由见
    `tests/workflows/rerank/test_pipeline.make_config`）。
    """
    return pipeline.RunConfig(
        workflow="rerank",
        split="train",
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=rerank_workflow.RerankConfig(split="train", case_limit=CASE_LIMIT),
        backend=backend,
        language_gate=False,
    )


def _rows(run_dir: Path) -> list[dict[str, JsonValue]]:
    """输入：run 目录。输出：train 切分的产物行。"""
    text = (run_dir / "train.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _rewrite(tmp_path: Path, rows: list[dict[str, JsonValue]], name: str = "tampered") -> Path:
    """输入：临时目录、行、目录名。输出：新 run 目录。

    预期行为：只写 `train.jsonl`——审计只读它，其余产物（进度日志等）与它无关。
    """
    target = tmp_path / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )
    return target


def _clean_rows(tmp_path: Path) -> list[dict[str, JsonValue]]:
    """输入：临时目录。输出：一份跑出来的、审计全绿的产物行。"""
    result = pipeline.run(make_config(tmp_path, "clean", FakeBackend()))
    assert result.accepted == CASE_LIMIT
    return _rows(result.run_dir)


def test_clean_products_pass_the_audit(tmp_path: Path) -> None:
    """输入：假后端跑出来的产物。输出：无。预期行为：一条缺失都没有。"""
    rows = _clean_rows(tmp_path)
    report, missing = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    for counter in (
        "terms_bad",
        "candidates_bad",
        "rule_bad",
        "top_bad",
        "count_bad",
        "pair_bad",
        "order_bad",
    ):
        assert report["train"][counter] == 0, counter


def _flip_choice(row: dict[str, JsonValue], qid: str) -> None:
    """输入：一行产物、题号。输出：无。预期行为：把该题的编号换成另一个（原地改）。"""
    answers = row["answers"]
    question = row["questions"][qid]
    assert isinstance(answers, dict) and isinstance(question, dict)
    criteria = question["criteria"]
    assert isinstance(criteria, dict)
    answer = answers[qid]
    assert isinstance(answer, dict)
    answer["choice"] = (int(answer["choice"]) + 1) % len(criteria)


@pytest.mark.parametrize(
    ("qid", "counter"),
    [("count", "count_bad"), ("order", "order_bad")],
)
def test_wrong_choice_answers_are_caught(tmp_path: Path, qid: str, counter: str) -> None:
    """输入：把某道 choice 的编号改掉。输出：无。预期行为：对应的那一格响一次。"""
    rows = _clean_rows(tmp_path)
    target = next(row for row in rows if qid in row["questions"])
    _flip_choice(target, qid)
    report, missing = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"][counter] == 1
    assert any(counter in item or "不自洽" in item for item in missing)


def test_wrong_top_answer_is_caught(tmp_path: Path) -> None:
    """输入：把 `top` 的编号改掉。输出：无。预期行为：`top_bad` 响一次。"""
    rows = _clean_rows(tmp_path)
    target = next(row for row in rows if "top" in row["questions"])
    _flip_choice(target, "top")
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["top_bad"] == 1


def test_missing_top_family_is_caught(tmp_path: Path) -> None:
    """输入：把一条该有 `top` 的样本里的 top 题删掉。输出：无。预期行为：`top_bad` 响一次。"""
    rows = _clean_rows(tmp_path)
    target = next(row for row in rows if "top" in row["questions"])
    assert isinstance(target["questions"], dict) and isinstance(target["answers"], dict)
    del target["questions"]["top"]
    del target["answers"]["top"]
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["top_bad"] == 1


def test_wrong_pair_answer_is_caught(tmp_path: Path) -> None:
    """输入：把某道 `pair` 的布尔翻过来。输出：无。预期行为：`pair_bad` 响一次。"""
    rows = _clean_rows(tmp_path)
    target = rows[0]
    answers = target["answers"]
    assert isinstance(answers, dict)
    answer = answers["pair"]
    assert isinstance(answer, dict)
    answer["noul"] = not answer["noul"]
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["pair_bad"] == 1


def test_tampered_rule_block_is_caught(tmp_path: Path) -> None:
    """输入：把 state 里的排序口径改掉。输出：无。预期行为：`rule_bad` 响一次。"""
    rows = _clean_rows(tmp_path)
    rows[0]["state"] = str(rows[0]["state"]).replace("得分高的排在前面", "得分低的排在前面")
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["rule_bad"] == 1


def test_dropped_candidate_line_is_caught(tmp_path: Path) -> None:
    """输入：从 state 里删掉一条候选行。输出：无。预期行为：`candidates_bad` 响一次。"""
    rows = _clean_rows(tmp_path)
    lines = str(rows[0]["state"]).splitlines()
    rows[0]["state"] = "\n".join(line for line in lines if not line.startswith("5. "))
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["candidates_bad"] == 1


def test_family_and_pair_index_helpers() -> None:
    """输入：题号。输出：无。预期行为：族名去掉数字后缀，`pair` 的下标从 0 起。"""
    assert audit_mod._family_of("pair_17") == "pair"
    assert audit_mod._family_of("top") == "top"
    assert audit_mod.pair_index("pair") == 0
    assert audit_mod.pair_index("pair_20") == 19
