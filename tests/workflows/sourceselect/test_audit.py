"""来源选择的审计测试：交叉校验、篡改检测、缺失产物。"""

from __future__ import annotations

import json
from pathlib import Path

from jc import pipeline
from workflows.sourceselect import audit as audit_mod, features, workflow as ss_workflow

from tests.workflows.sourceselect.conftest import FakeBackend


def make_config(tmp_path: Path, run_id: str, backend: object) -> pipeline.RunConfig:
    """构造一个小 case 集合的试跑配置。"""
    return pipeline.RunConfig(
        workflow="sourceselect",
        split="train",
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=ss_workflow.SourceSelectConfig(split="train", case_limit=4),
        backend=backend,
    )


def test_expected_pick_reads_the_table() -> None:
    """输入：一张候选表与三个首要要求。输出：无。

    预期行为：审计用的是特征层那一份口径（不重写第二套）；三条要求的答案都落在表里。
    """
    rows = (
        features.SourceRow(type="unknown", freshness="fresh", coverage="full"),
        features.SourceRow(type="official", freshness="stale", coverage="partial"),
    )
    assert audit_mod.expected_pick(rows, "authority") == "s2"
    assert audit_mod.expected_pick(rows, "freshness") == "s1"
    assert audit_mod.expected_pick(rows, "coverage") == "s1"
    empty: tuple[features.SourceRow, ...] = (
        features.SourceRow(type="unknown", freshness="stale", coverage="none"),
    )
    assert audit_mod.expected_pick(empty, "authority") == features.NO_PICK


def test_family_of_strips_dedupe_suffixes() -> None:
    """输入：题号。输出：无。预期行为：去掉 `_2` 这类后缀。"""
    assert audit_mod._family_of("pick") == "pick"
    assert audit_mod._family_of("pick_2") == "pick"


def test_audit_flags_a_tampered_pick(tmp_path: Path) -> None:
    """输入：把一条样本的 `pick` 答案改掉。输出：无。

    预期行为：审计报出 `pick_bad`——这条交叉校验是纯代码可重算的，改一个字节就该暴露。
    """
    result = pipeline.run(make_config(tmp_path, "a1", FakeBackend()))
    path = result.run_dir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    row = rows[0]
    question = row["questions"]["pick"]
    index = row["answers"]["pick"]["choice"]
    row["answers"]["pick"]["choice"] = (index + 1) % len(question["criteria"])
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n", encoding="utf-8"
    )
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["pick_bad"] == 1
    assert any("选哪一行" in item for item in missing)


def test_audit_flags_a_tampered_near_miss(tmp_path: Path) -> None:
    """输入：把一条样本的 `near_miss` 答案改掉。输出：无。预期行为：审计报出 `near_miss_bad`。"""
    result = pipeline.run(make_config(tmp_path, "a2", FakeBackend()))
    path = result.run_dir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows[0]["answers"]["near_miss"]["noul"] = not rows[0]["answers"]["near_miss"]["noul"]
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n", encoding="utf-8"
    )
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["near_miss_bad"] == 1
    assert any("诱饵" in item for item in missing)


def test_known_families_are_exactly_the_built_ones() -> None:
    """输入：无。输出：无。预期行为：审计认得的题族就是题族表本身。"""
    from workflows.sourceselect import questions

    assert audit_mod.KNOWN_FAMILIES == frozenset(questions.FAMILIES)
    assert audit_mod.CHECKED_FAMILIES == frozenset(questions.FAMILIES)
