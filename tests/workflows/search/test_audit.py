"""检索结果判读的审计测试：交叉校验、篡改检测、缺失产物。"""

from __future__ import annotations

import json
from pathlib import Path

from jc import pipeline
from workflows.search import audit as audit_mod, features

from tests.workflows._scaffold import run_config
from tests.workflows.search.conftest import FakeBackend


def make_config(tmp_path: Path, run_id: str, backend: object) -> pipeline.RunConfig:
    """构造一个小 case 集合的试跑配置。"""
    from workflows.search import workflow as search_workflow

    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="search",
        seed=7,
        override=search_workflow.SearchConfig(split="train", case_limit=4),
    )


def test_verdict_from_reads_the_table() -> None:
    """输入：规则名 / 未知名 / 非文本。输出：无。预期行为：认得出来、认不出就 None。"""
    assert audit_mod.verdict_from("spam_result") == "drop"
    assert audit_mod.verdict_from("none") == "cite"
    assert audit_mod.verdict_from("nope") is None
    assert audit_mod.verdict_from(3) is None


def test_audit_flags_a_tampered_verdict(tmp_path: Path) -> None:
    """输入：把一条样本的 `verdict` 答案改掉。输出：无。

    预期行为：审计报出 `verdict_bad`——这条交叉校验是纯代码可重算的，改一个字节就该暴露。
    """
    result = pipeline.run(make_config(tmp_path, "a1", FakeBackend()))
    path = result.run_dir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    row = rows[0]
    question = row["questions"]["verdict"]
    index = row["answers"]["verdict"]["choice"]
    row["answers"]["verdict"]["choice"] = (index + 1) % len(question["criteria"])
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n", encoding="utf-8"
    )
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["verdict_bad"] == 1
    assert any("不自洽" in item for item in missing)


def test_known_families_are_exactly_the_built_ones() -> None:
    """输入：无。输出：无。预期行为：审计认得的题族就是题族表本身。"""
    from workflows.search import questions

    assert audit_mod.KNOWN_FAMILIES == frozenset(questions.FAMILIES)
    assert features.RULES
