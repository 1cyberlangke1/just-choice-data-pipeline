"""改动审查的产物审计测试：能重算的那几条要真查得住，查不住的要说清楚。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jc import jsonio, pipeline, workflow as workflow_mod
from jc.jsontypes import JsonValue

from workflows.review import audit, features, render, workflow as review_workflow

from tests.workflows.review.conftest import FakeBackend


def make_config(tmp_path: Path, run_id: str, *, split: str = "train") -> pipeline.RunConfig:
    """构造一个小 case 集合的运行配置（指向 tmp 目录）。"""
    return pipeline.RunConfig(
        workflow="review",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=review_workflow.ReviewConfig(split=split, case_limit=6),
        backend=FakeBackend(),
    )


def rewrite(rows: list[JsonValue], target: Path) -> Path:
    """输入：样本行、目标文件。输出：写出后的路径。预期行为：整批重写一遍，供审计复读。"""
    jsonio.write_jsonl(target, rows)
    return target


def test_clean_run_audits_clean(tmp_path: Path) -> None:
    """输入：干净产物。输出：无。预期行为：零缺失、全部计数为零。"""
    result = pipeline.run(make_config(tmp_path, "a1"))
    report, missing = workflow_mod.load("review").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == 6
    for family in ("secret", "debug", "rule", "severity", "gate"):
        assert report["train"]["families"][family] > 0


def test_missing_product_is_reported(tmp_path: Path) -> None:
    """输入：没有产物的目录。输出：无。预期行为：缺失项里点名那个文件，不抛异常。"""
    empty = tmp_path / "nothing"
    empty.mkdir()
    report, missing = audit.audit_run(empty, ("train", "test_ood"), "zh")
    assert report["train"]["samples"] == 0
    assert len(missing) == 2
    assert all("产物不存在" in item for item in missing)


def test_unknown_locale_reports_before_reading(tmp_path: Path) -> None:
    """输入：不存在的语言。输出：无。预期行为：当场报错，不拖到解析阶段。"""
    with pytest.raises(FileNotFoundError):
        audit.audit_run(tmp_path, ("train",), "nope")


def test_a_tampered_manifest_is_caught(tmp_path: Path) -> None:
    """输入：把清单行里的规模改掉。输出：无。

    预期行为：`body_shape_bad` 记一笔（行数与新规模对不上）——正文形状是真查得住的，
    不是「看着像校验」。
    """
    result = pipeline.run(make_config(tmp_path, "a2"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    for row in rows:
        state = row["state"]
        assert isinstance(state, str)
        parsed = render.parse_state(state)
        # 改成与真实行数**一定对不上**的那一档：行数超过 3 就谎报「小」，否则谎报「大」
        wrong = "small" if len(parsed.body_lines) > 3 else "large"
        row["state"] = re.sub(r"\| (small|medium|large)$", f"| {wrong}", state, flags=re.M)
    run_dir = tmp_path / "tampered"
    rewrite(rows, run_dir / "train.jsonl")
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"]["body_shape_bad"] > 0
    assert any("改动正文形状" in item for item in missing)


def test_a_tampered_gate_answer_is_caught(tmp_path: Path) -> None:
    """输入：把门控答案换掉。输出：无。预期行为：`gate_bad` 记一笔。"""
    result = pipeline.run(make_config(tmp_path, "a3"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    for row in rows:
        answer = row["answers"]["gate"]
        answer["choice"] = (int(answer["choice"]) + 1) % 3
    run_dir = tmp_path / "tampered-gate"
    rewrite(rows, run_dir / "train.jsonl")
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"]["gate_bad"] > 0
    assert any("门控" in item for item in missing)


def test_a_tampered_severity_answer_is_caught(tmp_path: Path) -> None:
    """输入：把严重度答案换掉。输出：无。预期行为：`severity_bad` 记一笔。"""
    result = pipeline.run(make_config(tmp_path, "a4"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    for row in rows:
        answer = row["answers"]["severity"]
        answer["score"] = (int(answer["score"]) + 1) % features.SEVERITY_LEVELS
    run_dir = tmp_path / "tampered-severity"
    rewrite(rows, run_dir / "train.jsonl")
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"]["severity_bad"] > 0
    assert any("严重度" in item for item in missing)


def test_a_tampered_rule_answer_is_not_caught_by_the_audit(tmp_path: Path) -> None:
    """输入：把「命中哪条规则」的答案换掉。输出：无。

    预期行为：审计**查不出来**——`rule` 依赖正文承载的两维，纯代码读不出来，只能靠合成期的
    反解判别器把关。这条测试把这个已知边界钉住，免得有人误以为审计查过了。
    """
    result = pipeline.run(make_config(tmp_path, "a5"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    for row in rows:
        answer = row["answers"]["rule"]
        answer["choice"] = (int(answer["choice"]) + 1) % len(features.REVIEW_RULES)
    run_dir = tmp_path / "tampered-rule"
    rewrite(rows, run_dir / "train.jsonl")
    report, _ = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"]["rules_bad"] == 0  # 表本身没动
    assert report["train"]["manifest_bad"] == 0


def test_severity_from_maps_rule_and_file_kind() -> None:
    """输入：规则名与文件类型。输出：无。预期行为：兜底一律 0 档，其余查表，推不出给 None。"""
    assert audit.severity_from("none", "source") == 0
    assert audit.severity_from("credential", "source") == 3
    assert audit.severity_from("credential", "test") == 2
    assert audit.severity_from("size", "source") == 1
    assert audit.severity_from("credential", "docs") is None  # 不适用
    assert audit.severity_from("nope", "source") is None
    assert audit.severity_from(None, "source") is None


def test_gate_from_maps_severity_and_change_kind() -> None:
    """输入：严重度与变更类型。输出：无。预期行为：查表；越界或类型不对给 None。"""
    assert audit.gate_from(3, "added") == "block"
    assert audit.gate_from(1, "modified") == "request_changes"
    assert audit.gate_from(0, "deleted") == "allow"
    assert audit.gate_from(4, "added") is None
    assert audit.gate_from(-1, "added") is None
    assert audit.gate_from(True, "added") is None
    assert audit.gate_from("3", "added") is None


def test_body_shape_problem_reports_the_first_issue() -> None:
    """输入：各种与清单对不上的正文。输出：无。预期行为：各报各的那一条。"""
    small = render.ManifestState(file_kind="source", change_kind="modified", size="small")
    large = render.ManifestState(file_kind="source", change_kind="modified", size="large")
    added = render.ManifestState(file_kind="source", change_kind="added", size="small")
    deleted = render.ManifestState(file_kind="source", change_kind="deleted", size="small")
    assert audit.body_shape_problem(("+ a = 1", "- b = 2"), small) == ""
    assert "行数" in audit.body_shape_problem(("+ a = 1",), large)
    assert "一行都没变" in audit.body_shape_problem(("a = 1",), small)
    assert "一行 + 都没有" in audit.body_shape_problem(("a = 1",), added)
    assert "一行 - 都没有" in audit.body_shape_problem(("a = 1",), deleted)
    assert "新增" in audit.body_shape_problem(("+ a = 1", "- b = 2"), added)
    assert "删除" in audit.body_shape_problem(("+ a = 1", "- b = 2"), deleted)
    # 「修改」只加行在真实 diff 里很常见：不算问题（真机第一批为此误拒过三条）
    assert audit.body_shape_problem(("+ a = 1",), small) == ""
    # 整段缩进两格、上下文行没写空格前缀：都是同一件事，不算问题（范式 §9.19）
    assert audit.body_shape_problem(("  + a = 1", "  - b = 2"), small) == ""
    assert audit.body_shape_problem(("def f():", "-     return 1"), small) == ""


def test_a_tampered_manifest_token_is_caught(tmp_path: Path) -> None:
    """输入：把清单行的词元换成不认识的。输出：无。预期行为：`manifest_bad` 记一笔。"""
    result = pipeline.run(make_config(tmp_path, "a6"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    for row in rows:
        state = row["state"]
        assert isinstance(state, str)
        row["state"] = re.sub(r"\| (small|medium|large)$", "| huge", state, flags=re.M)
    run_dir = tmp_path / "tampered-token"
    rewrite(rows, run_dir / "train.jsonl")
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"]["manifest_bad"] == len(rows)
    assert any("清单行解析不出来" in item for item in missing)


@pytest.mark.parametrize(
    ("mutate", "counter", "hint"),
    [
        (
            lambda s: s.replace("- credential | 改动的新增行里有", "- credential | 改过的条件"),
            "rules_bad",
            "审查规则表",
        ),
        (
            lambda s: s.replace("- credential | 3 | 2 | 3 | 0", "- credential | 1 | 2 | 3 | 0"),
            "severity_table_bad",
            "严重度表",
        ),
        (lambda s: s.replace("- 3 | block | block | block", "- 3 | allow | block | block"), "gate_table_bad", "门控表"),
    ],
)
def test_a_tampered_table_is_caught(
    tmp_path: Path, mutate: object, counter: str, hint: str
) -> None:
    """输入：把三张表之一改掉。输出：无。预期行为：对应计数记一笔，缺失项点名那张表。"""
    result = pipeline.run(make_config(tmp_path, f"a7-{counter}"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    for row in rows:
        state = row["state"]
        assert isinstance(state, str)
        row["state"] = mutate(state)
    run_dir = tmp_path / f"tampered-{counter}"
    rewrite(rows, run_dir / "train.jsonl")
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"][counter] == len(rows)
    assert any(hint in item for item in missing)


def test_an_unknown_family_is_counted(tmp_path: Path) -> None:
    """输入：产物里出现本 workflow 不认识的题号。输出：无。

    预期行为：记进 `unknown_families`，而且原来那一族因为「一条都没出」被点出来——
    产物串版本时这两条一起报，比只报一条好定位。
    """
    result = pipeline.run(make_config(tmp_path, "a8"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    for row in rows:
        questions = row["questions"]
        answers = row["answers"]
        assert isinstance(questions, dict) and isinstance(answers, dict)
        questions["ghost_family"] = questions.pop("gate")
        answers["ghost_family"] = answers.pop("gate")
    run_dir = tmp_path / "unknown-family"
    rewrite(rows, run_dir / "train.jsonl")
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"]["unknown_families"]["ghost_family"] == len(rows)
    assert any("题族 gate 一条都没出" in item for item in missing)


def test_recorded_reads_every_answer_shape() -> None:
    """输入：三种题型的题面与答案。输出：无。预期行为：choice 换回选项名，其余原样。"""
    question = {"type": "choice", "criteria": {"allow": "放行", "block": "拦下"}}
    assert audit._recorded(question, {"type": "choice", "choice": 1}) == "block"
    assert audit._recorded(question, {"type": "choice", "choice": 9}) is None
    assert audit._recorded({"type": "score"}, {"type": "score", "score": 2}) == 2
    assert audit._recorded({"type": "noul"}, {"type": "noul", "noul": True}) is True
