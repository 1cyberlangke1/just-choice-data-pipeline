"""边界与接线：workflow 协议上的错误路径、注册与声明、审计的错误分支。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from jc import assemble, jsonio
from jc import workflow as workflow_mod
from workflows.modelroute import locales, questions, render, spec as sp
from workflows.modelroute.workflow import ID_CONFIG, ModelRouteWorkflow, bucket_of, cases_for

WF = ModelRouteWorkflow()


def test_workflow_is_registered_under_its_name() -> None:
    """输入：无。输出：无。预期行为：导入即自注册，`load` 拿得到同一个类。"""
    assert "modelroute" in workflow_mod.registered() or "modelroute" in workflow_mod.discover()
    assert isinstance(workflow_mod.load("modelroute"), ModelRouteWorkflow)


def test_declaration_matches_reality() -> None:
    """输入：无。输出：无。

    预期行为：纯构造式（`needs` 为空）、两种题型、两个判别器——声明错了会让管线去初始化
    用不到的后端（或反过来漏掉判别器）。
    """
    assert WF.needs == frozenset()
    assert WF.kinds == frozenset({"choice", "noul"})
    assert WF.judges == ("roundtrip", "label")
    assert WF.split_names() == ("train", "val", "test_iid", "test_ood")
    assert WF.available_locales() == locales.available()


def test_config_for_unknown_split_reports_the_known_ones() -> None:
    """输入：不存在的切分名。输出：无。预期行为：报 KeyError 并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        WF.config_for("dev")


def test_config_for_id_splits_are_independent_objects() -> None:
    """输入：三个同分布切分。输出：无。

    预期行为：三次拿到的是**不同对象**且 `split` 各自正确——返回同一个单例会让三档都按
    同一档切片，切出三份一模一样的数据。
    """
    configs = [WF.config_for(split) for split in ("train", "val", "test_iid")]
    assert [config.split for config in configs] == ["train", "val", "test_iid"]
    assert len({id(config) for config in configs}) == 3


def test_id_config_has_no_holdout_values() -> None:
    """输入：同分布配置。输出：无。预期行为：一个留出取值都不含（否则与 OOD 重叠）。"""
    assert not any(case.unseen_dims() for case in sp.build_cases(ID_CONFIG))


def test_cases_for_ood_returns_the_ood_only_matrix() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每条都至少含一个留出取值。"""
    cases = cases_for(WF.config_for("test_ood"))
    assert cases and all(case.unseen_dims() for case in cases)


def test_bucket_of_is_stable_and_spread() -> None:
    """输入：同一素材两次、不同素材一批。输出：无。

    预期行为：同一素材必得同一桶（切分不能随进程漂移）；一批素材落在多个桶里
    （全落一个桶会让 val / test_iid 空掉）。
    """
    assert bucket_of("a|b|c|d|e|f|0") == bucket_of("a|b|c|d|e|f|0")
    buckets = {bucket_of(f"m{index}") for index in range(200)}
    assert len(buckets) > 50


def test_make_judges_rejects_an_unknown_name() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：报 KeyError 并列出可用的。"""
    with pytest.raises(KeyError, match="未知判别器"):
        WF.make_judges(("nope",), "zh")


def test_bind_backend_is_forbidden() -> None:
    """输入：一个后端。输出：无。

    预期行为：直接报错——纯构造式 workflow 收到后端说明接线错了，静默接受会让「本该
    零 token」的批次偷偷花钱。
    """
    with pytest.raises(AssertionError, match="纯构造式"):
        WF.bind_backend(object())


def test_reparse_rejects_a_non_string_state() -> None:
    """输入：非字符串 state。输出：无。预期行为：报类型错。"""
    with pytest.raises(TypeError, match="state 必须是字符串"):
        WF.reparse(123)


def test_reparse_ignores_the_case_argument() -> None:
    """输入：渲染出来的 state 与一个 case。输出：无。

    预期行为：`reparse` 走合并标签表、忽略 `case` 参数（它是协议里的位置，读法用不上）。
    """
    spec = sp.sample_spec(sp.RouteCase("extract", "low", "relaxed", "normal", "first", "public", 0), "train", 0, 0)
    request = WF.reparse(render.build(spec, locales.load("zh")), case=spec.case)
    assert request.body == locales.load("zh")["body"]["extract"]


def test_plan_sample_ignores_the_rng_and_keeps_the_case() -> None:
    """输入：两个不同状态的 RNG。输出：无。

    预期行为：同一 case 的生成计划一致（真值不掷骰子），所以 `per_case > 1` 出的重复样本
    也能被一致性校验抓住。
    """
    case = sp.RouteCase("codefix", "high", "strict", "tight", "retry", "private", 1)
    first = WF.plan_sample(random.Random(1), ID_CONFIG, case, 0)
    second = WF.plan_sample(random.Random(999), ID_CONFIG, case, 0)
    assert first.params["case"] is case and second.params["case"] is case
    assert WF.sample(random.Random(1), first) == WF.sample(random.Random(2), second)


def test_case_label_and_dimension_titles() -> None:
    """输入：一个 case。输出：无。预期行为：标签是可读串，七个维度都有中文标题。"""
    case = sp.RouteCase("reason", "high", "strict", "tight", "retry", "private", 1)
    assert WF.case_label(case) == "reason/high/strict/tight/retry/private/1"
    assert set(WF.dimension_titles()) == set(case.as_dict())
    assert set(WF.stats_dimensions(case)) == set(case.as_dict())


# ---------- 审计 ----------


def _sample_row() -> dict[str, Any]:
    """输入：无。输出：一条真样本（中文，`extract` 任务，六个题族都出满）。

    预期行为：走生产路径（渲染 → 出题 → 组装），判别器与审计都判它合格；
    `extract` 有升级链，所以 `next` 那一族也在。
    """
    loc = locales.load("zh")
    spec = sp.sample_spec(sp.RouteCase("extract", "mid", "normal", "normal", "first", "public", 0), "train", 0, 0)
    items = questions.build(spec, loc, 0, random.Random(1))
    return assemble.assemble_sample("modelroute-train-0000000", render.build(spec, loc), items, random.Random(1))


def test_audit_reports_a_missing_split(tmp_path: Path) -> None:
    """输入：一个不存在的切分。输出：无。预期行为：如实记一笔「产物不存在」。"""
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert any("产物不存在" in item for item in missing)
    assert report["train"]["samples"] == 0


def test_audit_flags_an_out_of_range_choice_index(tmp_path: Path) -> None:
    """输入：编号越界的产物。输出：无。预期行为：审计记一笔「编号越界」。"""
    row = _sample_row()
    qid = next(key for key in row["questions"] if row["questions"][key]["type"] == "choice")
    row["answers"][qid]["choice"] = 99
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


def test_audit_flags_an_unknown_family(tmp_path: Path) -> None:
    """输入：带未知题族的产物。输出：无。预期行为：审计点名那个题族。"""
    row = _sample_row()
    row["questions"]["bogus"] = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    row["answers"]["bogus"] = {"type": "noul", "noul": True}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["unknown_families"]["bogus"] == 1
    assert any("未知题族" in item for item in missing)


def test_audit_flags_a_state_it_cannot_read(tmp_path: Path) -> None:
    """输入：state 读不出来的产物。输出：无。

    预期行为：记一条「反解不出来」而不是静默跳过——静默跳过会把一条坏样本算成好样本。
    """
    row = _sample_row()
    row["state"] = "今天天气不错。"
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("反解不出来" in item for item in missing)


def test_audit_flags_a_family_that_never_shows_up(tmp_path: Path) -> None:
    """输入：把某一族的题全删掉。输出：无。预期行为：审计记一笔「一条都没出」。"""
    row = _sample_row()
    del row["questions"]["next"]
    del row["answers"]["next"]
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["families"]["next"] == 0
    assert any("next" in item and "一条都没出" in item for item in missing)


def test_audit_flags_an_answer_that_contradicts_the_text(tmp_path: Path) -> None:
    """输入：把某个 noul 答案翻过来的产物。输出：无。

    预期行为：记一笔「与文本重算不一致」——审计不依赖真值规格，只靠文本重算。
    """
    row = _sample_row()
    row["answers"]["cache"]["noul"] = not row["answers"]["cache"]["noul"]
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("重算不一致" in item for item in missing)


def test_audit_flags_a_question_type_this_workflow_never_emits(tmp_path: Path) -> None:
    """输入：把 `cache` 那道题换成 `score` 题型。输出：无。

    预期行为：记一笔答案对不上——本题型只出 choice / noul，混进来的题不静默放过。
    """
    row = _sample_row()
    row["questions"]["cache"] = {"type": "score", "criteria": ["a", "b"]}
    row["answers"]["cache"] = {"type": "score", "score": 3}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("重算不一致" in item for item in missing)


def test_audit_counts_tiers_and_efforts(tmp_path: Path) -> None:
    """输入：一条合格产物。输出：无。预期行为：审计顺带记下档位与推理深度的分布。"""
    row = _sample_row()
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert dict(report["train"]["tiers"]) == {"mini": 1}
    assert dict(report["train"]["efforts"]) == {"mid": 1}
    assert report["train"]["samples"] == 1
