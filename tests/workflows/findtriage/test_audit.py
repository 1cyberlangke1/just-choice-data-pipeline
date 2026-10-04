"""改动引入性判定的审计测试：交叉校验、篡改检测、缺失产物。"""

from __future__ import annotations

import json
from pathlib import Path

from jc import pipeline
from jc.jsontypes import JsonValue
from workflows.findtriage import audit as audit_mod, features, render, workflow as ft_workflow

from tests.workflows._scaffold import run_config
from tests.workflows.findtriage.conftest import FakeBackend


def make_config(tmp_path: Path, run_id: str, backend: object) -> pipeline.RunConfig:
    """构造一个小 case 集合的试跑配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="findtriage",
        seed=7,
        override=ft_workflow.FindTriageConfig(split="train", case_limit=4),
    )


def read_rows(path: Path) -> list[dict[str, JsonValue]]:
    """输入：产物路径。输出：逐行解析出来的样本。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict[str, JsonValue]]) -> None:
    """输入：产物路径、样本。输出：无。预期行为：按管线同样的写法落盘。"""
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n", encoding="utf-8"
    )


def test_evidence_symbol_prefers_a_symbol_that_is_in_the_table() -> None:
    """输入：记录表与两种发现原文。输出：无。

    预期行为：表内出现过的记号优先（那才是「引用记录」）；表外只有记号时取第一个。
    """
    rows = (
        features.RecordRow(line=10, touch="added", symbol="alpha_fn"),
        features.RecordRow(line=20, touch="untouched", symbol="beta_fn"),
    )
    assert audit_mod.evidence_symbol_of(rows, "第 10 行：`beta_fn` 有问题") == "beta_fn"
    assert audit_mod.evidence_symbol_of(rows, "第 10 行：`gamma_fn` 有问题") == "gamma_fn"
    assert audit_mod.evidence_symbol_of(rows, "第 10 行：没有记号") == ""


def test_family_of_strips_dedupe_suffixes() -> None:
    """输入：题号。输出：无。预期行为：去掉 `_2` 这类后缀。"""
    assert audit_mod._family_of("triage") == "triage"
    assert audit_mod._family_of("triage_2") == "triage"


def test_audit_passes_a_clean_run(tmp_path: Path) -> None:
    """输入：假后端跑出来的一份产物。输出：无。预期行为：零缺失。"""
    result = pipeline.run(make_config(tmp_path, "a0", FakeBackend()))
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == 4


def test_audit_flags_a_tampered_triage(tmp_path: Path) -> None:
    """输入：把一条样本的 `triage` 答案改掉。输出：无。

    预期行为：审计报出 `triage_bad`——这条交叉校验是纯代码可重算的，改一个字节就该暴露。
    """
    result = pipeline.run(make_config(tmp_path, "a1", FakeBackend()))
    path = result.run_dir / "train.jsonl"
    rows = read_rows(path)
    question = rows[0]["questions"]["triage"]
    index = rows[0]["answers"]["triage"]["choice"]
    rows[0]["answers"]["triage"]["choice"] = (index + 1) % len(question["criteria"])
    write_rows(path, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["triage_bad"] == 1
    assert any("处置档" in item for item in missing)


def test_audit_flags_a_tampered_actionable(tmp_path: Path) -> None:
    """输入：把一条样本的 `actionable` 答案改掉。输出：无。预期行为：审计报出 `actionable_bad`。"""
    result = pipeline.run(make_config(tmp_path, "a2", FakeBackend()))
    path = result.run_dir / "train.jsonl"
    rows = read_rows(path)
    rows[0]["answers"]["actionable"]["noul"] = not rows[0]["answers"]["actionable"]["noul"]
    write_rows(path, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["actionable_bad"] == 1
    assert any("落不落地" in item for item in missing)


def test_audit_flags_a_tampered_owner(tmp_path: Path) -> None:
    """输入：把一条样本的 `owner` 答案改掉。输出：无。

    预期行为：审计报出 `owner_bad`——归属是拿产物里的记录表 + 发现原文重推的，改一个字节就该暴露。
    """
    result = pipeline.run(make_config(tmp_path, "a4", FakeBackend()))
    path = result.run_dir / "train.jsonl"
    rows = read_rows(path)
    question = rows[0]["questions"]["owner"]
    index = rows[0]["answers"]["owner"]["choice"]
    rows[0]["answers"]["owner"]["choice"] = (index + 1) % len(question["criteria"])
    write_rows(path, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["owner_bad"] == 1
    assert any("归属" in item for item in missing)


def test_audit_flags_a_tampered_finding(tmp_path: Path) -> None:
    """输入：把发现里的行号改掉。输出：无。

    预期行为：审计按**发现原文**重推「指向第几条」，改掉行号就该报 `cite_bad`。
    """
    result = pipeline.run(make_config(tmp_path, "a3", FakeBackend()))
    path = result.run_dir / "train.jsonl"
    rows = read_rows(path)
    frame = render.parse_state(rows[0]["state"])
    cited = render.cite_line_any(frame.finding)
    assert cited is not None, "假后端写出来的发现必须带行号"
    rows[0]["state"] = rows[0]["state"].replace(
        f"第 {cited} 行", f"第 {cited + 500} 行", 1
    )
    write_rows(path, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["cite_bad"] == 1
    assert any("指向哪一行" in item for item in missing)


def test_known_families_are_exactly_the_built_ones() -> None:
    """输入：无。输出：无。预期行为：审计认得的题族就是题族表本身。"""
    from workflows.findtriage import questions

    assert audit_mod.KNOWN_FAMILIES == frozenset(questions.FAMILIES)
    assert audit_mod.CHECKED_FAMILIES == frozenset(questions.FAMILIES)
