"""testimpact 的审计测试：产物自洽性靠纯代码重算，改坏一处就要被抓住。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import contract, jsonio, pipeline
from jc.jsontypes import JsonValue
from tests.workflows.testimpact.conftest import FakeBackend
from tests.workflows.testimpact.test_pipeline import make_config
from workflows.testimpact import features, judges, render
from workflows.testimpact.workflow import ImpactWorkflow


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run 名。输出：产物目录。预期行为：用假后端跑一小批。"""
    return pipeline.run(make_config(tmp_path, run_id, FakeBackend())).run_dir


def _rows(run_dir: Path) -> list[JsonValue]:
    """输入：产物目录。输出：train 切分的全部行。"""
    return [row for _, row in jsonio.iter_jsonl(run_dir / "train.jsonl")]


def _write(run_dir: Path, rows: list[JsonValue]) -> None:
    """输入：产物目录、行。输出：无。预期行为：原样写回（审计只读产物）。"""
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)


def _audit(
    run_dir: Path, split: str = "train", locale: str = "zh"
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：产物目录、切分、语言。输出：审计报告。"""
    return ImpactWorkflow().audit(run_dir, (split,), locale)


def _first_choice(row: JsonValue) -> tuple[str, dict[str, JsonValue]]:
    """输入：一行产物。输出：(题号, 答案)。预期行为：取第一条 choice 题（没有就报错）。"""
    answers = row["answers"]
    for qid, answer in answers.items():
        if answer["type"] == "choice":
            return qid, answer
    raise AssertionError("这一行没有 choice 题")


def test_clean_run_audits_without_missing(tmp_path: Path) -> None:
    """输入：一小批合格产物。输出：无。预期行为：零缺失，四族都出得来，五个计数器全零。"""
    run_dir = _run(tmp_path)
    report, missing = _audit(run_dir)
    assert missing == []
    for family in ("flag", "count", "which", "route"):
        assert report["train"]["families"][family] > 0, family
    for key in ("table_bad", "flag_bad", "count_bad", "which_bad", "route_bad"):
        assert report["train"][key] == 0, key


@pytest.mark.parametrize(
    ("family", "counter"),
    [("count", "count_bad"), ("which", "which_bad"), ("route", "route_bad")],
)
def test_choice_answer_mismatch_is_caught(
    tmp_path: Path, family: str, counter: str
) -> None:
    """输入：把某族 choice 的答案编号换到另一个选项上。输出：无。

    预期行为：该族的计数器加一——审计比的是**选项描述**，所以换一个编号就必然对不上。
    """
    run_dir = _run(tmp_path, f"audit-{family}")
    rows = _rows(run_dir)
    row = rows[0]
    answers = row["answers"]
    options = contract.option_names(row["questions"][family])
    answers[family] = {
        "type": "choice",
        "choice": (answers[family]["choice"] + 1) % len(options),
    }
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"][counter] == 1
    assert any(counter in item for item in missing) is False


def test_flag_answer_mismatch_is_caught(tmp_path: Path) -> None:
    """输入：把逐套件那一族的某条答案取反。输出：无。预期行为：`flag_bad` 加一。"""
    run_dir = _run(tmp_path, "audit-flag")
    rows = _rows(run_dir)
    row = rows[0]
    qid = next(qid for qid in row["answers"] if qid.startswith("flag"))
    answer = row["answers"][qid]
    row["answers"][qid] = {"type": "noul", "noul": not answer["noul"]}
    _write(run_dir, rows)
    report, _missing = _audit(run_dir)
    assert report["train"]["flag_bad"] == 1


def test_state_without_a_coverage_table_is_caught(tmp_path: Path) -> None:
    """输入：把覆盖表里的一行删掉。输出：无。预期行为：`table_bad` 加一。"""
    run_dir = _run(tmp_path, "audit-table")
    rows = _rows(run_dir)
    row = rows[0]
    rows[0] = {**row, "state": str(row["state"]).replace("- P3", "- 没了")}
    _write(run_dir, rows)
    report, _missing = _audit(run_dir)
    assert report["train"]["table_bad"] == 1


def test_a_diff_touching_a_path_outside_the_map_is_caught(tmp_path: Path) -> None:
    """输入：把改动里的文件换成覆盖表之外的路径。输出：无。预期行为：`table_bad` 加一（落点算不出来）。"""
    run_dir = _run(tmp_path, "audit-outside")
    rows = _rows(run_dir)
    row = rows[0]
    pid = render.parse_state(row["state"]).touched[0]
    path = features.PATH_FILES[pid]
    patched = str(row["state"]).replace(f"b/{path}", "b/src/other/thing.py")
    patched = patched.replace(f"a/{path}", "a/src/other/thing.py")
    rows[0] = {**row, "state": patched}
    _write(run_dir, rows)
    report, _missing = _audit(run_dir)
    assert report["train"]["table_bad"] == 1


def test_unknown_family_is_counted(tmp_path: Path) -> None:
    """输入：把一条题号改成别的题族。输出：无。预期行为：`unknown_families` 记一笔（串版本看得见）。"""
    run_dir = _run(tmp_path, "audit-unknown")
    rows = _rows(run_dir)
    row = rows[0]
    qid = next(iter(row["questions"]))
    questions = {("mystery" if key == qid else key): value for key, value in row["questions"].items()}
    answers = {("mystery" if key == qid else key): value for key, value in row["answers"].items()}
    rows[0] = {**row, "questions": questions, "answers": answers}
    _write(run_dir, rows)
    report, _missing = _audit(run_dir)
    assert report["train"]["unknown_families"]["mystery"] == 1


def test_missing_split_is_reported(tmp_path: Path) -> None:
    """输入：一个没有产物的切分。输出：无。预期行为：缺失项里点名它，而不是静默空报告。"""
    run_dir = _run(tmp_path, "audit-missing")
    _report, missing = _audit(run_dir, split="val")
    assert missing and "val" in missing[0]


def test_unknown_locale_is_rejected(tmp_path: Path) -> None:
    """输入：不存在的语言。输出：无。预期行为：当场报错（别拖到解析阶段才报）。"""
    run_dir = _run(tmp_path, "audit-locale")
    with pytest.raises(FileNotFoundError, match="fr"):
        _audit(run_dir, locale="fr")


def test_flag_count_mismatch_is_caught(tmp_path: Path) -> None:
    """输入：删掉一条逐套件题。输出：无。预期行为：`flag_bad` 加一（条数不对也算不自洽）。"""
    run_dir = _run(tmp_path, "audit-flagcount")
    rows = _rows(run_dir)
    row = rows[0]
    qid = next(qid for qid in row["answers"] if qid.startswith("flag"))
    rows[0] = {
        **row,
        "questions": {k: v for k, v in row["questions"].items() if k != qid},
        "answers": {k: v for k, v in row["answers"].items() if k != qid},
    }
    _write(run_dir, rows)
    report, _missing = _audit(run_dir)
    assert report["train"]["flag_bad"] == 1
    assert features.SUITES
