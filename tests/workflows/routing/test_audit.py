"""工单分诊的审计测试：交叉校验、篡改检测、缺失产物。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jc import pipeline

from workflows.routing import audit as audit_mod, features, locales, questions, render
from workflows.routing.workflow import RoutingConfig

from tests.workflows.routing.conftest import FakeBackend

CASE_LIMIT = 4


def make_config(tmp_path: Path, run_id: str, backend: object) -> pipeline.RunConfig:
    """构造一个小 case 集合的试跑配置。"""
    return pipeline.RunConfig(
        workflow="routing",
        split="train",
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=RoutingConfig(split="train", case_limit=CASE_LIMIT),
        backend=backend,
    )


def read_rows(run_dir: Path) -> list[dict[str, object]]:
    """输入：run 目录。输出：train.jsonl 里的行（已解析）。"""
    path = run_dir / "train.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(run_dir: Path, rows: list[dict[str, object]]) -> None:
    """输入：run 目录、行。输出：无。预期行为：按产物同样的格式写回（每行一条 + 末尾换行）。"""
    path = run_dir / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_queue_from_reads_the_table() -> None:
    """输入：规则名 / 未知名 / 非文本。输出：无。预期行为：认得出来、认不出就 None。"""
    assert audit_mod.queue_from("outage_response") == "incident_response"
    assert audit_mod.queue_from("loss_response") == "incident_response"
    assert audit_mod.queue_from("defect") == "engineering"
    assert audit_mod.queue_from("nope") is None
    assert audit_mod.queue_from(3) is None


def test_priority_from_reads_the_table() -> None:
    """输入：紧急度 + 客户等级。输出：无。预期行为：查表；不认识就 None。"""
    assert audit_mod.priority_from("blocked", "enterprise") == 3
    assert audit_mod.priority_from("can_wait", "free") == 0
    assert audit_mod.priority_from("nope", "free") is None
    assert audit_mod.priority_from("blocked", "gold") is None
    assert audit_mod.priority_from(None, "free") is None


def test_family_of_strips_dedupe_suffixes() -> None:
    """输入：题号。输出：无。预期行为：去掉 `_2` 这类后缀，也认得带下划线的族名。"""
    assert audit_mod._family_of("queue") == "queue"
    assert audit_mod._family_of("queue_2") == "queue"
    assert audit_mod._family_of("needs_human") == "needs_human"
    assert audit_mod._family_of("needs_human_2") == "needs_human"


def test_recorded_handles_every_answer_shape() -> None:
    """输入：三种题型的答案。输出：无。预期行为：choice 换回选项名，score/noul 原样。"""
    question = {"criteria": {"a": "A", "b": "B"}}
    assert audit_mod._recorded(question, {"type": "choice", "choice": 1}) == "b"
    assert audit_mod._recorded(question, {"type": "choice", "choice": 9}) is None
    assert audit_mod._recorded({}, {"type": "score", "score": 2}) == 2
    assert audit_mod._recorded({}, {"type": "noul", "noul": True}) is True


def test_needs_human_consistency_rule() -> None:
    """输入：几组组合。输出：无。

    预期行为：事故队列一律要人工；非事故队列只有「关键域 + 企业版」才要；
    记下的值不是布尔、或者队列词元不认识，就不做要求（返回真，交给别的计数）。
    """
    assert audit_mod.needs_human_consistent("incident_response", "storage", "free", True)
    assert not audit_mod.needs_human_consistent("incident_response", "storage", "free", False)
    assert audit_mod.needs_human_consistent("engineering", "payments", "enterprise", True)
    assert not audit_mod.needs_human_consistent("engineering", "payments", "pro", True)
    assert audit_mod.needs_human_consistent("engineering", "storage", "free", False)
    assert audit_mod.needs_human_consistent("nowhere", "storage", "free", False)
    assert audit_mod.needs_human_consistent("engineering", "storage", "free", "yes")


def test_audit_passes_a_clean_run(tmp_path: Path) -> None:
    """输入：干净的一批产物。输出：无。预期行为：零问题、零缺失。"""
    result = pipeline.run(make_config(tmp_path, "a0", FakeBackend()))
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    assert report["train"]["unknown_families"] == {}


def test_audit_flags_a_tampered_queue(tmp_path: Path) -> None:
    """输入：把一条样本的 `queue` 答案改掉。输出：无。

    预期行为：审计报出 `queue_bad`——这条交叉校验是纯代码可重算的，改一个字节就该暴露。
    """
    result = pipeline.run(make_config(tmp_path, "a1", FakeBackend()))
    rows = read_rows(result.run_dir)
    questions_of = rows[0]["questions"]
    answers_of = rows[0]["answers"]
    assert isinstance(questions_of, dict) and isinstance(answers_of, dict)
    options = questions_of["queue"]["criteria"]
    index = answers_of["queue"]["choice"]
    answers_of["queue"]["choice"] = (index + 1) % len(options)
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["queue_bad"] == 1
    assert any("队列" in item for item in missing)


def test_audit_flags_a_tampered_priority(tmp_path: Path) -> None:
    """输入：把每条样本的 `priority` 档位改掉。输出：无。

    预期行为：审计报出 `priority_bad`——档位必须与「紧急度 × 客户等级」自洽。
    """
    result = pipeline.run(make_config(tmp_path, "a2", FakeBackend()))
    rows = read_rows(result.run_dir)
    for row in rows:
        answers_of = row["answers"]
        assert isinstance(answers_of, dict)
        answers_of["priority"]["score"] = (int(answers_of["priority"]["score"]) + 1) % 4
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["priority_bad"] >= 1
    assert any("优先级" in item for item in missing)


def test_audit_flags_a_tampered_needs_human(tmp_path: Path) -> None:
    """输入：把「要不要人工」一律取反。输出：无。

    预期行为：审计报出 `human_bad`——它必须与队列 / 属性行自洽。
    """
    result = pipeline.run(make_config(tmp_path, "a3", FakeBackend()))
    rows = read_rows(result.run_dir)
    for row in rows:
        answers_of = row["answers"]
        assert isinstance(answers_of, dict)
        answers_of["needs_human"]["noul"] = not answers_of["needs_human"]["noul"]
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["human_bad"] >= 1
    assert any("人工" in item for item in missing)


def test_audit_flags_a_tampered_queue_table(tmp_path: Path) -> None:
    """输入：把 state 里派单表某一行的队列改掉。输出：无。

    预期行为：审计报出 `queue_table_bad`——派单表是代码渲染的规范表，产物里必须逐行相同。
    """
    result = pipeline.run(make_config(tmp_path, "a4", FakeBackend()))
    rows = read_rows(result.run_dir)
    state = rows[0]["state"]
    assert isinstance(state, str)
    rows[0]["state"] = state.replace("| incident_response", "| nowhere", 1)
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["queue_table_bad"] == 1
    assert any("派单表" in item for item in missing)


def test_audit_flags_a_tampered_priority_table(tmp_path: Path) -> None:
    """输入：把 state 里优先级表某一行的档位改掉。输出：无。

    预期行为：审计报出 `priority_table_bad`。
    """
    result = pipeline.run(make_config(tmp_path, "a5", FakeBackend()))
    rows = read_rows(result.run_dir)
    state = rows[0]["state"]
    assert isinstance(state, str)
    loc = locales.load("zh")
    first = render.priority_lines(loc)[0]
    rows[0]["state"] = state.replace(first, first.replace("| 0 | 0 | 1", "| 3 | 3 | 3"))
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["priority_table_bad"] == 1
    assert any("优先级表" in item for item in missing)


def test_audit_flags_a_broken_attr_row(tmp_path: Path) -> None:
    """输入：把 state 里的属性行整行删掉。输出：无。

    预期行为：审计报出 `attrs_bad`——属性行解析不出来时，后面的交叉校验也就无从谈起。
    """
    result = pipeline.run(make_config(tmp_path, "a6", FakeBackend()))
    rows = read_rows(result.run_dir)
    state = rows[0]["state"]
    assert isinstance(state, str)
    loc = locales.load("zh")
    head = locales.header(loc, loc["meta"]["attrs_label"])
    lines = state.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(head))
    del lines[index + 1]  # 属性行紧跟段落头
    rows[0]["state"] = "\n".join(lines)
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["attrs_bad"] == 1
    assert any("属性行" in item for item in missing)


def test_audit_flags_a_tampered_flag_table(tmp_path: Path) -> None:
    """输入：把 state 里附加规则某一行的说明改掉。输出：无。

    预期行为：审计报出 `flags_bad`——附加规则也是代码渲染的规范表。
    """
    result = pipeline.run(make_config(tmp_path, "a8", FakeBackend()))
    rows = read_rows(result.run_dir)
    state = rows[0]["state"]
    assert isinstance(state, str)
    loc = locales.load("zh")
    first = render.flag_lines(loc)[0]
    rows[0]["state"] = state.replace(first, "- critical_areas | 关键产品域只有三个")
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["flags_bad"] == 1
    assert any("附加规则" in item for item in missing)


def test_audit_reports_a_missing_product(tmp_path: Path) -> None:
    """输入：没有产物的切分。输出：无。预期行为：缺失项里点名它，计数全为零。"""
    report, missing = audit_mod.audit_run(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)
    assert report["train"]["families"] == {}


def test_audit_rejects_an_unknown_locale(tmp_path: Path) -> None:
    """输入：不存在的语言。输出：无。预期行为：当场报错，不拖到解析阶段。"""
    with pytest.raises(FileNotFoundError):
        audit_mod.audit_run(tmp_path, ("train",), "nope")


def test_known_families_are_exactly_the_built_ones() -> None:
    """输入：无。输出：无。预期行为：审计认得的题族就是题族表本身。"""
    assert audit_mod.KNOWN_FAMILIES == frozenset(questions.FAMILIES)
    assert features.QUEUE_RULES


def test_audit_flags_an_unknown_family_and_a_missing_one(tmp_path: Path) -> None:
    """输入：把一个题族改名成不认识的。输出：无。

    预期行为：报出 `unknown_families` 一笔，同时点名原来那一族一条都没出——
    产物串版本时这两条会一起出现。
    """
    result = pipeline.run(make_config(tmp_path, "a7", FakeBackend()))
    rows = read_rows(result.run_dir)
    for row in rows:
        questions_of = row["questions"]
        answers_of = row["answers"]
        assert isinstance(questions_of, dict) and isinstance(answers_of, dict)
        questions_of["bogus"] = questions_of.pop("issue")
        answers_of["bogus"] = answers_of.pop("issue")
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["unknown_families"]["bogus"] == CASE_LIMIT
    assert any("题族 issue" in item for item in missing)
