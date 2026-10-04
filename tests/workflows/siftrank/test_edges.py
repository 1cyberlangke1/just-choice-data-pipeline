"""边角与接线：审计的四种报错、协议路径、坏配置。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from pathlib import Path

from jc import assemble, jsonio
from tests.workflows.siftrank.conftest import tiny_config
from workflows.siftrank import locales, parse, questions
from workflows.siftrank import spec as sp
from workflows.siftrank import workflow as siftrank_workflow

WORKFLOW = siftrank_workflow.SiftRankWorkflow()
LOC = locales.load("zh")


def build_row(config: sp.RankConfig, case: sp.RankCase, index: int, locale: str = "zh") -> dict[str, JsonValue]:
    """输入：采样参数、case、序号、语言。输出：一条契约样本。

    预期行为：走真管线那三步（计划 → 渲染 → 出题 → 组装），不手写样本。
    """
    plan = WORKFLOW.plan_sample(random.Random(index), config, case, index)
    spec = WORKFLOW.sample(random.Random(index), plan)
    labeled = WORKFLOW.ask(spec, random.Random(index), locale)
    return assemble.assemble_sample(f"s-{index:07d}", WORKFLOW.render(spec, random.Random(index), locale), labeled, random.Random(index))


def test_audit_flags_tampered_answers_and_unknown_families(tmp_path: Path) -> None:
    """输入：被改坏的产物。输出：无。

    预期行为：审计逐条报出来——答案与 state 重算不一致、编号越界、未知题族、题族一条没出。
    """
    config = tiny_config("train")
    rows = [build_row(config, case, index) for index, case in enumerate(WORKFLOW.build_cases(config))]
    rows[0]["answers"]["cut_size"]["score"] = (
        rows[0]["answers"]["cut_size"]["score"] + 1
    ) % len(sp.CUTS)
    rows[1]["answers"]["agg_top"]["choice"] = 99
    rows[2]["questions"]["mystery"] = dict(rows[2]["questions"]["agg_top"])
    rows[2]["answers"]["mystery"] = dict(rows[2]["answers"]["agg_top"])
    for row in rows:
        row["questions"].pop("stable")
        row["answers"].pop("stable")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)
    report, missing = WORKFLOW.audit(run_dir, ("train",), "zh")
    assert report["train"]["samples"] == len(rows)
    assert report["train"]["label_mismatch"] >= 1
    assert report["train"]["index_bad"] >= 1
    assert report["train"]["unknown_families"]["mystery"] >= 1
    assert any("不一致" in item for item in missing)
    assert any("编号异常" in item for item in missing)
    assert any("未知题族" in item for item in missing)
    assert any("stable 一条都没出" in item for item in missing)


def test_audit_counts_the_answer_shapes(tmp_path: Path) -> None:
    """输入：一份干净产物。输出：无。

    预期行为：审计顺手统计相关条目数 / 一致性 / 谁更相关 / 头名四张分布表（报告要用），
    且题族齐全、零缺失。
    """
    config = tiny_config("train")
    rows = [build_row(config, case, index) for index, case in enumerate(WORKFLOW.build_cases(config))]
    run_dir = tmp_path / "run3"
    run_dir.mkdir()
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)
    report, missing = WORKFLOW.audit(run_dir, ("train",), "zh")
    assert missing == []
    counts = report["train"]
    for key in ("cut", "stable", "relation", "top"):
        assert sum(counts[key].values()) == len(rows), key
    assert set(counts["types"]) == {"choice", "noul", "score"}
    assert counts["index_bad"] == 0


def test_reparse_through_the_protocol_is_language_agnostic() -> None:
    """输入：两种语言渲染的 state。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，靠合并标签表也能反解出四个批次——它是
    「产物审计不依赖语言」的那条路。
    """
    config = tiny_config("train")
    case = WORKFLOW.build_cases(config)[0]
    plan = WORKFLOW.plan_sample(random.Random(0), config, case, 0)
    spec = WORKFLOW.sample(random.Random(0), plan)
    for lang in ("zh", "en"):
        parsed = WORKFLOW.reparse(WORKFLOW.render(spec, random.Random(0), lang))
        assert parsed.request_id == spec.request_id
        assert len(parsed.batches) == sp.TRIALS


def test_reparse_tolerates_blank_lines_and_indentation() -> None:
    """输入：行首行尾加了空白、中间插了空行的 state。输出：无。

    预期行为：反解按 `strip()` 后的内容认抬头与批次——空行是给人看的，不该影响结构。
    """
    config = tiny_config("train")
    case = WORKFLOW.build_cases(config)[0]
    plan = WORKFLOW.plan_sample(random.Random(0), config, case, 0)
    spec = WORKFLOW.sample(random.Random(0), plan)
    lines = WORKFLOW.render(spec, random.Random(0), "zh").splitlines()
    noisy = "".join(f"   {line}   \n\n" for line in lines)
    parsed = parse.parse_state(noisy, parse.tables_of(LOC))
    assert parsed.request_id == spec.request_id
    assert len(parsed.batches) == sp.TRIALS


def test_language_texts_is_empty() -> None:
    """输入：一条样本。输出：无。

    预期行为：纯构造族报**空表**——state 由代码按语言模板拼，按语言判只会把好数据当坏数据。
    """
    config = tiny_config("train")
    rows = [build_row(config, case, index) for index, case in enumerate(WORKFLOW.build_cases(config))]
    assert WORKFLOW.language_texts(rows[0]) == {}


def test_config_for_returns_a_fresh_split_for_the_id_side() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：拿到的配置带着调用方要的那一档 `split`——直接返回 `ID_CONFIG` 单例会让
    train/val/test_iid 三次都按同一档切片（切出来三份一模一样的数据）。
    """
    for split in ("train", "val", "test_iid"):
        assert WORKFLOW.config_for(split).split == split
    assert WORKFLOW.config_for("test_ood").split == "test_ood"
    assert WORKFLOW.config_for("test_ood").ood_only is True


def test_dimension_titles_cover_every_dimension() -> None:
    """输入：一个 case。输出：无。预期行为：`stats_dimensions` 的每个维都有中文标题。"""
    case = WORKFLOW.build_cases(tiny_config("train"))[0]
    assert set(WORKFLOW.stats_dimensions(case)) == set(WORKFLOW.dimension_titles())


def test_plan_sample_carries_the_case_and_a_phrasing() -> None:
    """输入：一次计划。输出：无。

    预期行为：`params` 里带 case / 问法 / 序号，而且问法取自配置里的池子——`sample` 阶段
    不再掷骰子，随机决策全在计划这一步。
    """
    config = tiny_config("train")
    case = WORKFLOW.build_cases(config)[0]
    plan = WORKFLOW.plan_sample(random.Random(3), config, case, 7)
    assert plan.name == "train"
    assert plan.params["case"] == case
    assert plan.params["seq"] == 7
    assert plan.params["phrasing"] in config.phrasings


def test_questions_are_deterministic_across_the_workflow() -> None:
    """输入：同一条计划、两次 `ask`。输出：无。预期行为：题目逐字段一致。"""
    config = tiny_config("train")
    case = WORKFLOW.build_cases(config)[1]
    plan = WORKFLOW.plan_sample(random.Random(1), config, case, 1)
    spec = WORKFLOW.sample(random.Random(1), plan)
    first = WORKFLOW.ask(spec, random.Random(1), "en")
    second = WORKFLOW.ask(spec, random.Random(1), "en")
    assert first == second
    assert [item.family for item in first] == list(questions.FAMILIES)
