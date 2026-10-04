"""会话上下文压缩的审计测试：交叉校验、篡改检测、缺失产物。"""

from __future__ import annotations

import json
from pathlib import Path


from jc import pipeline

from workflows.compact import audit as audit_mod, features, questions
from workflows.compact.workflow import CompactConfig

from tests.workflows._scaffold import run_config
from tests.workflows.compact.conftest import FakeBackend

CASE_LIMIT = 4


def make_config(tmp_path: Path, run_id: str, backend: object) -> pipeline.RunConfig:
    """构造一个小 case 集合的试跑配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="compact",
        seed=7,
        override=CompactConfig(split="train", case_limit=CASE_LIMIT),
    )


def read_rows(run_dir: Path) -> list[dict[str, object]]:
    """输入：run 目录。输出：train.jsonl 里的行（已解析）。"""
    path = run_dir / "train.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_rows(run_dir: Path, rows: list[dict[str, object]]) -> None:
    """输入：run 目录、行。输出：无。预期行为：按产物同样的格式写回（每行一条 + 末尾换行）。"""
    path = run_dir / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_disposition_from_reads_the_table() -> None:
    """输入：规则名 / 未知名 / 非文本。输出：无。预期行为：认得出来、认不出就 None。"""
    assert audit_mod.disposition_from("budget_tool") == "summarize"
    assert audit_mod.disposition_from("critical_old") == "drop"
    assert audit_mod.disposition_from("tight_old") == "summarize"
    assert audit_mod.disposition_from("none") == "keep"
    assert audit_mod.disposition_from("nope") is None
    assert audit_mod.disposition_from(3) is None


def test_family_of_strips_dedupe_suffixes() -> None:
    """输入：题号。输出：无。预期行为：去掉 `_2` 这类后缀，也认得带下划线的族名。"""
    assert audit_mod._family_of("rule") == "rule"
    assert audit_mod._family_of("rule_2") == "rule"
    assert audit_mod._family_of("pin_overrode") == "pin_overrode"
    assert audit_mod._family_of("pin_overrode_2") == "pin_overrode"


def test_pin_overrode_consistency_rule() -> None:
    """输入：四种组合。输出：无。

    预期行为：真值要求「规则 == pinned_keep」且「位置行写着 pinned」；
    假值不做额外要求（不钉住、或者钉了但本来也是 keep，两种都讲得通）。
    """
    assert audit_mod.pin_overrode_consistent("pinned_keep", "pinned", True)
    assert not audit_mod.pin_overrode_consistent("critical_old", "pinned", True)
    assert not audit_mod.pin_overrode_consistent("pinned_keep", "unpinned", True)
    assert audit_mod.pin_overrode_consistent("critical_old", "unpinned", False)
    assert audit_mod.pin_overrode_consistent("nope", "unpinned", False)


def test_audit_passes_a_clean_run(tmp_path: Path) -> None:
    """输入：干净的一批产物。输出：无。预期行为：零问题、零缺失。"""
    result = pipeline.run(make_config(tmp_path, "a0", FakeBackend()))
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    assert report["train"]["unknown_families"] == {}


def test_audit_flags_a_tampered_disposition(tmp_path: Path) -> None:
    """输入：把一条样本的 `disposition` 答案改掉。输出：无。

    预期行为：审计报出 `disposition_bad`——这条交叉校验是纯代码可重算的，改一个字节就该暴露。
    """
    result = pipeline.run(make_config(tmp_path, "a1", FakeBackend()))
    rows = read_rows(result.run_dir)
    row = rows[0]
    questions_of = row["questions"]
    answers_of = row["answers"]
    assert isinstance(questions_of, dict) and isinstance(answers_of, dict)
    options = questions_of["disposition"]["criteria"]
    index = answers_of["disposition"]["choice"]
    answers_of["disposition"]["choice"] = (index + 1) % len(options)
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["disposition_bad"] == 1
    assert any("不自洽" in item for item in missing)


def test_audit_flags_a_tampered_pin_flag(tmp_path: Path) -> None:
    """输入：把「钉住有没有起作用」一律改成真。输出：无。

    预期行为：审计报出 `pin_bad`——没钉住的那几条不可能「钉住起了作用」。
    """
    result = pipeline.run(make_config(tmp_path, "a2", FakeBackend()))
    rows = read_rows(result.run_dir)
    for row in rows:
        answers_of = row["answers"]
        assert isinstance(answers_of, dict)
        answers_of["pin_overrode"]["noul"] = True
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["pin_bad"] >= 1
    assert any("钉住" in item for item in missing)


def test_audit_flags_a_tampered_rule_table(tmp_path: Path) -> None:
    """输入：把 state 里处置表某一行的处置改掉。输出：无。

    预期行为：审计报出 `rules_bad`——处置表是代码渲染的规范表，产物里必须逐行相同。
    """
    result = pipeline.run(make_config(tmp_path, "a3", FakeBackend()))
    rows = read_rows(result.run_dir)
    state = rows[0]["state"]
    assert isinstance(state, str)
    rows[0]["state"] = state.replace("| keep", "| drop", 1)
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["rules_bad"] == 1
    assert any("处置表" in item for item in missing)


def test_audit_flags_a_broken_where_row(tmp_path: Path) -> None:
    """输入：把 state 里的位置行整行删掉。输出：无。

    预期行为：审计报出 `frame_bad`——位置行解析不出来时，后面的交叉校验也就无从谈起。
    """
    result = pipeline.run(make_config(tmp_path, "a4", FakeBackend()))
    rows = read_rows(result.run_dir)
    state = rows[0]["state"]
    assert isinstance(state, str)
    row = next(line for line in state.splitlines() if line.startswith("- ") and "|" in line)
    rows[0]["state"] = state.replace(row, "- 位置未知")
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["frame_bad"] == 1
    assert any("位置行" in item for item in missing)


def test_known_families_are_exactly_the_built_ones() -> None:
    """输入：无。输出：无。预期行为：审计认得的题族就是题族表本身。"""
    assert audit_mod.KNOWN_FAMILIES == frozenset(questions.FAMILIES)
    assert features.RULES
