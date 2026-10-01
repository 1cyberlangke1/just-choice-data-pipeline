"""审计测试：产物被改坏时，对应的那一格计数必须真的响。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jc import pipeline
from jc.jsontypes import JsonValue

from workflows.citation import audit as audit_mod, features, locales, workflow as citation_workflow
from tests.workflows.citation.conftest import FakeBackend

CASE_LIMIT = 8


def make_config(tmp_path: Path, run_id: str, backend: object) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。

    预期行为：**关掉语言闸**——本包的假后端只写英文（夹具局限，理由见
    `tests/workflows/citation/test_pipeline.make_config`）。
    """
    return pipeline.RunConfig(
        workflow="citation",
        split="train",
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=citation_workflow.CitationConfig(split="train", case_limit=CASE_LIMIT),
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


COUNTERS = (
    "doc_bad",
    "quote_bad",
    "claim_bad",
    "verdict_bad",
    "in_doc_bad",
    "section_bad",
    "in_claimed_bad",
)


def test_clean_products_pass_the_audit(tmp_path: Path) -> None:
    """输入：假后端跑出来的产物。输出：无。预期行为：一条缺失都没有。"""
    rows = _clean_rows(tmp_path)
    report, missing = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    for counter in COUNTERS:
        assert report["train"][counter] == 0, counter


def test_missing_product_is_reported(tmp_path: Path) -> None:
    """输入：一个空目录。输出：无。预期行为：报「产物不存在」。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    _, missing = audit_mod.audit_run(empty, ("train",), "zh")
    assert missing and "产物不存在" in missing[0]


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


def _flip_noul(row: dict[str, JsonValue], qid: str) -> None:
    """输入：一行产物、题号。输出：无。预期行为：把该题的布尔翻过来（原地改）。"""
    answers = row["answers"]
    assert isinstance(answers, dict)
    answer = answers[qid]
    assert isinstance(answer, dict)
    answer["noul"] = not answer["noul"]


@pytest.mark.parametrize(
    ("qid", "counter"),
    [("verdict", "verdict_bad"), ("section", "section_bad")],
)
def test_wrong_choice_answers_are_caught(tmp_path: Path, qid: str, counter: str) -> None:
    """输入：把某道 choice 的编号改掉。输出：无。预期行为：对应的那一格响一次。"""
    rows = _clean_rows(tmp_path)
    target = next(row for row in rows if qid in row["questions"])
    _flip_choice(target, qid)
    report, missing = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"][counter] == 1
    assert any(counter in item or "不自洽" in item for item in missing)


@pytest.mark.parametrize(
    ("qid", "counter"),
    [("in_doc", "in_doc_bad"), ("in_claimed", "in_claimed_bad")],
)
def test_wrong_noul_answers_are_caught(tmp_path: Path, qid: str, counter: str) -> None:
    """输入：把某道 noul 的布尔翻过来。输出：无。预期行为：对应的那一格响一次。"""
    rows = _clean_rows(tmp_path)
    _flip_noul(rows[0], qid)
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"][counter] == 1


def test_missing_section_family_is_caught(tmp_path: Path) -> None:
    """输入：把一条该有 `section` 的样本里的那一族删掉。输出：无。

    预期行为：`section_bad` 响一次——查得到的引文必须出这一族，少一条就是构造侧漏了。
    """
    rows = _clean_rows(tmp_path)
    target = next(row for row in rows if "section" in row["questions"])
    questions = target["questions"]
    answers = target["answers"]
    assert isinstance(questions, dict) and isinstance(answers, dict)
    del questions["section"]
    del answers["section"]
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["section_bad"] == 1


def test_section_family_on_a_fabricated_quote_is_caught(tmp_path: Path) -> None:
    """输入：给一条查不到的引文硬加 `section` 那一族。输出：无。

    预期行为：`section_bad` 响一次——「查不到的引文出自第几节」没有唯一解，出了就是坏数据。
    """
    rows = _clean_rows(tmp_path)
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["section_bad"] == 0
    fabricated = [
        row
        for row in rows
        if all(qid != "section" for qid in row["questions"])
    ]
    if not fabricated:
        pytest.skip("这一批里没有查不到的引文，换不到可验的样本")
    target = fabricated[0]
    questions = target["questions"]
    answers = target["answers"]
    assert isinstance(questions, dict) and isinstance(answers, dict)
    questions["section"] = {
        "type": "choice",
        "instructions": "这条引文真正出自第几节？",
        "criteria": {"s0": "第 1 节", "s1": "第 2 节", "s2": "第 3 节"},
    }
    answers["section"] = {"type": "choice", "choice": 0}
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["section_bad"] == 1


def test_tampered_claim_line_is_caught(tmp_path: Path) -> None:
    """输入：把 state 里的声称出处改掉。输出：无。预期行为：重推出来的结论与答案打架。

    预期行为（细）：只挑**引文查得到、且原来的声称出处不是第 3 节**的那一条——把声称出处改到
    第 3 节之后，重推出来的结论一定变（要么 `verified` ↔ `mislocated` 互换，要么
    `in_claimed` 翻面），而答案还是旧的，两边必然打架。查不到的引文改不出这个效果：它改前改后
    都是 `fabricated`。
    """
    rows = _clean_rows(tmp_path)
    loc = locales.load("zh")
    mark = str(loc["state"]["claim_mark"])
    changed = False
    for row in rows:
        if "section" not in row["questions"]:
            continue
        lines = str(row["state"]).splitlines()
        index = next(i for i, line in enumerate(lines) if line.strip().startswith(mark))
        if lines[index].strip().endswith("3 节"):
            continue
        lines[index] = f"{mark}{str(loc['state']['claim_value']).format(n=3)}"
        row["state"] = "\n".join(lines)
        changed = True
        break
    assert changed, "这一批里没有可改的样本（引文查得到且声称出处不是第 3 节）"
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["claim_bad"] == 0
    assert report["train"]["verdict_bad"] + report["train"]["in_claimed_bad"] > 0


def test_out_of_range_claim_line_is_caught(tmp_path: Path) -> None:
    """输入：把声称出处改成第 9 节。输出：无。预期行为：`claim_bad` 响一次。"""
    rows = _clean_rows(tmp_path)
    loc = locales.load("zh")
    mark = str(loc["state"]["claim_mark"])
    lines = str(rows[0]["state"]).splitlines()
    index = next(i for i, line in enumerate(lines) if line.strip().startswith(mark))
    lines[index] = f"{mark}{str(loc['state']['claim_value']).format(n=9)}"
    rows[0]["state"] = "\n".join(lines)
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["claim_bad"] == 1


def test_dropped_section_heading_is_caught(tmp_path: Path) -> None:
    """输入：从 state 里删掉一个节抬头。输出：无。预期行为：`doc_bad` 响一次。"""
    rows = _clean_rows(tmp_path)
    loc = locales.load("zh")
    mark = str(loc["state"]["sec_mark"]).format(n=features.SECTIONS)
    lines = str(rows[0]["state"]).splitlines()
    rows[0]["state"] = "\n".join(line for line in lines if line.strip() != mark)
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["doc_bad"] == 1


def test_missing_quote_line_is_caught(tmp_path: Path) -> None:
    """输入：把引文那一行整行删掉。输出：无。预期行为：`quote_bad` 响一次。"""
    rows = _clean_rows(tmp_path)
    loc = locales.load("zh")
    mark = str(loc["state"]["cite_mark"])
    lines = str(rows[0]["state"]).splitlines()
    rows[0]["state"] = "\n".join(line for line in lines if not line.strip().startswith(mark))
    report, _ = audit_mod.audit_run(_rewrite(tmp_path, rows), ("train",), "zh")
    assert report["train"]["quote_bad"] == 1


def test_unknown_locale_is_rejected_up_front(tmp_path: Path) -> None:
    """输入：不存在的语言。输出：无。预期行为：当场抛（不拖到解析阶段）。"""
    with pytest.raises(FileNotFoundError):
        audit_mod.audit_run(tmp_path, ("train",), "nope")


def test_family_helper_strips_the_suffix() -> None:
    """输入：题号。输出：无。预期行为：族名去掉数字后缀。"""
    assert audit_mod._family_of("section_2") == "section"
    assert audit_mod._family_of("in_claimed") == "in_claimed"
    assert audit_mod._family_of("in_doc") == "in_doc"
