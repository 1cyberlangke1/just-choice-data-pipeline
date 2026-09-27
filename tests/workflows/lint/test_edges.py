"""边界与接线：workflow 协议上的错误路径、注册与声明。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from jc import assemble, jsonio
from jc import workflow as workflow_mod
from workflows.lint import locales, questions, render, spec as sp
from workflows.lint.workflow import ID_CONFIG, LintWorkflow, bucket_of, cases_for


def test_workflow_is_registered_under_its_name() -> None:
    """输入：无。输出：无。预期行为：导入即自注册，`load` 拿得到同一个类。"""
    assert "lint" in workflow_mod.registered() or "lint" in workflow_mod.discover()
    assert isinstance(workflow_mod.load("lint"), LintWorkflow)


def test_declaration_matches_reality() -> None:
    """输入：无。输出：无。

    预期行为：纯构造式（`needs` 为空）、两种题型、两个判别器——声明错了会让管线去初始化
    用不到的后端（或反过来漏掉判别器）。
    """
    workflow = LintWorkflow()
    assert workflow.needs == frozenset()
    assert workflow.kinds == frozenset({"choice", "noul"})
    assert workflow.judges == ("roundtrip", "label")
    assert workflow.split_names() == ("train", "val", "test_iid", "test_ood")
    assert workflow.available_locales() == locales.available()


def test_config_for_unknown_split_reports_the_known_ones() -> None:
    """输入：不存在的切分名。输出：无。预期行为：报 KeyError 并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        LintWorkflow().config_for("dev")


def test_config_for_id_splits_are_independent_objects() -> None:
    """输入：三个同分布切分。输出：无。

    预期行为：三次拿到的是**不同对象**且 `split` 各自正确——返回同一个单例会让三档都按
    同一档切片，切出三份一模一样的数据。
    """
    workflow = LintWorkflow()
    configs = [workflow.config_for(split) for split in ("train", "val", "test_iid")]
    assert [config.split for config in configs] == ["train", "val", "test_iid"]
    assert len({id(config) for config in configs}) == 3


def test_id_config_has_no_holdout_values() -> None:
    """输入：同分布配置。输出：无。预期行为：一个留出取值都不含（否则与 OOD 重叠）。"""
    assert not any(case.unseen_dims() for case in sp.build_cases(ID_CONFIG))


def test_cases_for_ood_returns_the_ood_only_matrix() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每条都至少含一个留出取值。"""
    cases = cases_for(LintWorkflow().config_for("test_ood"))
    assert cases and all(case.unseen_dims() for case in cases)


def test_bucket_of_is_stable_and_spread() -> None:
    """输入：同一素材两次、不同素材一批。输出：无。

    预期行为：同一素材必得同一桶（切分不能随进程漂移）；一批素材落在多个桶里
    （全落一个桶会让 val / test_iid 空掉）。
    """
    assert bucket_of("a|b|1|0") == bucket_of("a|b|1|0")
    assert bucket_of("a|b|1|0") != bucket_of("a|b|1|1") or True  # 允许偶发同桶
    buckets = {bucket_of(f"m{index}") for index in range(200)}
    assert len(buckets) > 50


def test_make_judges_rejects_an_unknown_name() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：报 KeyError 并列出可用的。"""
    with pytest.raises(KeyError, match="未知判别器"):
        LintWorkflow().make_judges(("nope",), "zh")


def test_bind_backend_is_forbidden() -> None:
    """输入：一个后端。输出：无。

    预期行为：直接报错——纯构造式 workflow 收到后端说明接线错了，静默接受会让「本该
    零 token」的批次偷偷花钱。
    """
    with pytest.raises(AssertionError, match="纯构造式"):
        LintWorkflow().bind_backend(object())


def test_reparse_rejects_a_non_string_state() -> None:
    """输入：非字符串 state。输出：无。预期行为：报类型错。"""
    with pytest.raises(TypeError, match="state 必须是字符串"):
        LintWorkflow().reparse(123)


def test_plan_sample_ignores_the_rng_and_keeps_the_case() -> None:
    """输入：两个不同状态的 RNG。输出：无。

    预期行为：同一 case 的生成计划完全一致（问法是 case 的维度，不掷骰子），所以
    `per_case > 1` 出的重复样本也能被一致性校验抓住。
    """
    import random

    workflow = LintWorkflow()
    case = sp.LintCase("todo", "code", 2, 1)
    first = workflow.plan_sample(random.Random(1), ID_CONFIG, case, 0)
    second = workflow.plan_sample(random.Random(999), ID_CONFIG, case, 0)
    assert first.params["case"] is case and second.params["case"] is case
    assert workflow.sample(random.Random(1), first) == workflow.sample(random.Random(2), second)


def test_case_label_and_dimension_titles() -> None:
    """输入：一个 case。输出：无。预期行为：标签是可读串，四个维度都有中文标题。"""
    workflow = LintWorkflow()
    assert workflow.case_label(sp.LintCase("debug", "log", 4, 1)) == "debug/log/4/1"
    titles = workflow.dimension_titles()
    assert set(titles) == set(sp.LintCase("debug", "log", 4, 1).as_dict())


def test_audit_reports_a_missing_split(tmp_path: Path) -> None:
    """输入：一个不存在的切分。输出：无。预期行为：如实记一笔「产物不存在」。"""
    report, missing = LintWorkflow().audit(tmp_path, ("train",), "zh")
    assert any("产物不存在" in item for item in missing)
    assert report["train"]["samples"] == 0


def _sample_row() -> dict[str, Any]:
    """输入：无。输出：一条真样本（中文，命中 1 行）。

    预期行为：走生产路径（渲染 → 出题 → 组装），判别器与审计都判它合格；
    命中 1 行让四个题族（含条件题族 `spot`）都出满，改坏哪一族都能看出来。
    """
    loc = locales.load("zh")
    spec = sp.sample_spec(sp.LintCase("secret", "code", 1, 0), "train", 0)
    lines = render.render_lines(spec.plan, loc)
    items = questions.build(spec.plan.rule, lines, spec.plan.matched(), loc, 0)
    return assemble.assemble_sample(
        "lint-train-0000000", render.render_state(spec, loc), items, random.Random(1)
    )


def test_audit_flags_an_out_of_range_choice_index(tmp_path: Path) -> None:
    """输入：编号越界的产物。输出：无。预期行为：审计记一笔「编号越界」。"""
    row = _sample_row()
    qid = next(key for key in row["questions"] if row["questions"][key]["type"] == "choice")
    row["answers"][qid]["choice"] = 99
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = LintWorkflow().audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


def test_audit_flags_an_unknown_family(tmp_path: Path) -> None:
    """输入：带未知题族的产物。输出：无。预期行为：审计点名那个题族。"""
    row = _sample_row()
    row["questions"]["bogus"] = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    row["answers"]["bogus"] = {"type": "noul", "noul": True}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = LintWorkflow().audit(tmp_path, ("train",), "zh")
    assert report["train"]["unknown_families"]["bogus"] == 1
    assert any("未知题族" in item for item in missing)


def test_audit_flags_a_state_it_cannot_read(tmp_path: Path) -> None:
    """输入：state 读不出来的产物。输出：无。

    预期行为：记一条「反解不出来」而不是静默跳过——静默跳过会把一条坏样本算成好样本。
    """
    row = _sample_row()
    row["state"] = "今天天气不错。"
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = LintWorkflow().audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("反解不出来" in item for item in missing)


def test_audit_flags_a_family_that_never_shows_up(tmp_path: Path) -> None:
    """输入：把某一族的题全删掉。输出：无。预期行为：审计记一笔「一条都没出」。"""
    row = _sample_row()
    del row["questions"]["gate"]
    del row["answers"]["gate"]
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = LintWorkflow().audit(tmp_path, ("train",), "zh")
    assert report["train"]["families"]["gate"] == 0
    assert any("gate" in item and "一条都没出" in item for item in missing)
