"""边界与错误路径：坏切分、坏判别器名、坏产物、越界取值、构造式约束。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from pathlib import Path


import pytest

from jc import assemble, jsonio
from tests.workflows.match.conftest import tiny_config
from workflows.match import locales, parse, render
from workflows.match import spec as sp
from workflows.match import workflow as match_workflow

LOC = locales.load("zh")
WORKFLOW = match_workflow.MatchWorkflow()


def build_row(config: sp.MatchConfig, case: sp.MatchCase, seq: int) -> dict[str, JsonValue]:
    """输入：配置、case、序号。输出：契约样本。

    预期行为：走与管线同一条路（计划 → 规格 → 渲染 → 出题 → 组装），不手写样本。
    """
    plan = WORKFLOW.plan_sample(random.Random(seq), config, case, seq)
    spec = WORKFLOW.sample(random.Random(seq), plan)
    state = WORKFLOW.render(spec, random.Random(seq), "zh")
    labeled = WORKFLOW.ask(spec, random.Random(seq), "zh")
    return assemble.assemble_sample(f"m-{seq:07d}", state, labeled, random.Random(seq))


def test_workflow_protocol_surface() -> None:
    """输入：无。输出：无。预期行为：协议方法齐备、声明正确。"""
    assert WORKFLOW.name == "match"
    assert WORKFLOW.version
    assert WORKFLOW.needs == frozenset()
    assert WORKFLOW.judges == ("roundtrip", "label")
    assert WORKFLOW.kinds == frozenset({"choice", "noul", "score"})
    assert WORKFLOW.split_names() == ("train", "val", "test_iid", "test_ood")
    assert WORKFLOW.available_locales() == ["en", "zh"]
    assert WORKFLOW.dimension_titles() == dict(match_workflow.DIMENSION_TITLES)


def test_case_helpers_expose_the_dimensions() -> None:
    """输入：一个 case。输出：无。预期行为：描述与维度表都摊平了五个维度。"""
    case = sp.MatchCase("lager", sp.PATTERNS[0], sp.SPELLINGS[0], "plain", "none")
    assert WORKFLOW.case_label(case) == case.describe()
    assert WORKFLOW.stats_dimensions(case) == case.as_dict()


def test_seen_keeps_the_original_order() -> None:
    """输入：完整取值表与见过的集合。输出：无。

    预期行为：按原表顺序过滤，不排序——排序会把写法对 0-1 与 1-2 对调，报告会对不上。
    """
    assert sp.seen_in_order(("a", "b", "c", "d"), frozenset({"d", "b"})) == ("b", "d")


def test_bucket_of_is_deterministic_and_bounded() -> None:
    """输入：若干素材指纹。输出：无。预期行为：同一素材同桶、桶号落在 0~99、换种子换桶。"""
    material = sp.material_of(
        sp.MatchCase("lager", sp.PATTERNS[0], sp.SPELLINGS[0], "plain", "none")
    )
    assert match_workflow.bucket_of(material) == match_workflow.bucket_of(material)
    assert 0 <= match_workflow.bucket_of(material) < 100
    assert match_workflow.bucket_of(material, seed=1) != match_workflow.bucket_of(material, seed=2)


def test_slice_keeps_a_material_in_one_split() -> None:
    """输入：全枚举 case 的三档切片。输出：无。

    预期行为：同一素材（同一条请求）整组落在同一档——素材留出的地基。
    """
    cases = sp.build_cases(tiny_config("train"))
    where: dict[str, set[str]] = {}
    for split in ("train", "val", "test_iid"):
        for case in match_workflow._slice(cases, split):
            where.setdefault(sp.material_of(case), set()).add(split)
    assert where and all(len(splits) == 1 for splits in where.values())


def test_render_rejects_unknown_cell_key() -> None:
    """输入：表里没有的取值。输出：无。预期行为：KeyError（语言文件漏了某个取值）。"""
    with pytest.raises(KeyError):
        render.cell("size", 42, 0, LOC)
    with pytest.raises(KeyError):
        render.cell("name", "sake", 0, LOC)


def test_render_rejects_unknown_layout() -> None:
    """输入：不存在的显示顺序。输出：无。预期行为：KeyError。"""
    case = sp.MatchCase("lager", sp.PATTERNS[0], sp.SPELLINGS[0], "zigzag", "none")
    with pytest.raises(KeyError):
        render.record_lines(case, 0, LOC, sp.ID_PRODUCTS)


def test_parse_rejects_an_empty_state() -> None:
    """输入：空 state。输出：无。预期行为：报「没有抬头」。"""
    with pytest.raises(ValueError, match="没有抬头"):
        parse.parse_state("", parse.tables_of(LOC))


def test_parse_tolerates_blank_lines() -> None:
    """输入：掺了空行与行尾空格的 state。输出：无。

    预期行为：空行跳过、行首行尾空格忽略——state 是人读的文本，排版噪声不该影响反解。
    """
    config = tiny_config("train")
    case = WORKFLOW.build_cases(config)[0]
    plan = WORKFLOW.plan_sample(random.Random(0), config, case, 0)
    spec = WORKFLOW.sample(random.Random(0), plan)
    state = WORKFLOW.render(spec, random.Random(0), "zh")
    padded = "\n\n".join(f"  {line}  " for line in state.splitlines()) + "\n"
    parsed = parse.parse_state(padded, parse.tables_of(LOC))
    assert parsed.request_id == spec.query_id
    assert set(parsed.left) == set(sp.FIELDS)


def test_reparse_through_the_protocol_is_language_agnostic() -> None:
    """输入：两种语言渲染的 state。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，靠合并标签表也能反解出五栏——它是
    「产物审计不依赖语言」的那条路。
    """
    config = tiny_config("train")
    case = WORKFLOW.build_cases(config)[0]
    plan = WORKFLOW.plan_sample(random.Random(0), config, case, 0)
    spec = WORKFLOW.sample(random.Random(0), plan)
    for lang in ("zh", "en"):
        parsed = WORKFLOW.reparse(WORKFLOW.render(spec, random.Random(0), lang))
        assert parsed.request_id == spec.query_id
        assert set(parsed.left) == set(sp.FIELDS) == set(parsed.right)


def test_audit_flags_tampered_answers_and_unknown_families(tmp_path: Path) -> None:
    """输入：被改坏的产物。输出：无。

    预期行为：审计逐条报出来——答案与 state 重算不一致、编号越界、未知题族、题族一条没出。
    """
    config = tiny_config("train")
    rows = [
        build_row(config, case, index)
        for index, case in enumerate(WORKFLOW.build_cases(config))
    ]
    rows[0]["answers"]["band"]["score"] = (rows[0]["answers"]["band"]["score"] + 1) % 3
    rows[1]["answers"]["verdict"]["choice"] = 99
    rows[2]["questions"]["mystery"] = dict(rows[2]["questions"]["band"])
    rows[2]["answers"]["mystery"] = dict(rows[2]["answers"]["band"])
    for row in rows:
        row["questions"].pop("same_pack")
        row["answers"].pop("same_pack")
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
    assert any("same_pack 一条都没出" in item for item in missing)


def test_audit_flags_an_unparsable_state(tmp_path: Path) -> None:
    """输入：state 缺了抬头的那条产物。输出：无。预期行为：报「反解不出来」并计入不一致。"""
    config = tiny_config("train")
    rows = [
        build_row(config, case, index)
        for index, case in enumerate(WORKFLOW.build_cases(config))
    ]
    rows[0]["state"] = "乱七八糟\n"
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)
    report, missing = WORKFLOW.audit(run_dir, ("train",), "zh")
    assert any("反解不出来" in item for item in missing)
    assert report["train"]["label_mismatch"] >= 1


def test_audit_counts_the_three_answer_shapes(tmp_path: Path) -> None:
    """输入：一份干净产物。输出：无。

    预期行为：审计顺手统计处置 / 档位 / 写法栏数三张分布表（报告要用），且题族齐全。
    """
    config = tiny_config("train")
    rows = [
        build_row(config, case, index)
        for index, case in enumerate(WORKFLOW.build_cases(config))
    ]
    run_dir = tmp_path / "run3"
    run_dir.mkdir()
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)
    report, missing = WORKFLOW.audit(run_dir, ("train",), "zh")
    assert missing == []
    counts = report["train"]
    assert sum(counts["verdict"].values()) == len(rows)
    assert sum(counts["band"].values()) == len(rows)
    assert sum(counts["variant_count"].values()) == len(rows)
    assert set(counts["types"]) == {"choice", "noul", "score"}
