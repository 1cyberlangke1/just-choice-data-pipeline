"""边界与错误路径：坏切分、坏判别器名、坏产物、构造式约束、读法的极端输入。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from jc import assemble, jsonio
from jc import workflow as workflow_mod
from tests.workflows.autoformat.conftest import _make_case, tiny_config
from workflows.autoformat import judges, locales, parse, render
from workflows.autoformat import spec as sp
from workflows.autoformat import workflow as af_workflow

LOC = locales.load("zh")
WORKFLOW = af_workflow.AutoformatWorkflow()


def _row(case: sp.DocCase, seq: int = 0) -> tuple[dict[str, Any], sp.DocSpec]:
    """输入：case、序号。输出：(契约样本, 真值规格)。

    预期行为：走与管线同一条路（计划 → 规格 → 渲染 → 出题 → 组装），不手写样本。
    """
    config = tiny_config("train")
    plan = WORKFLOW.plan_sample(random.Random(seq), config, case, seq)
    spec = WORKFLOW.sample(random.Random(seq), plan)
    state = WORKFLOW.render(spec, random.Random(seq), "zh")
    labeled = WORKFLOW.ask(spec, random.Random(seq), "zh")
    row = assemble.assemble_sample(
        f"autoformat-train-{seq:07d}", state, labeled, random.Random(seq)
    )
    return row, spec


def test_workflow_protocol_surface() -> None:
    """输入：无。输出：无。预期行为：协议方法齐备、声明正确。"""
    assert WORKFLOW.name == "autoformat"
    assert WORKFLOW.version
    assert WORKFLOW.needs == frozenset()
    assert WORKFLOW.judges == ("roundtrip", "label")
    assert WORKFLOW.kinds == frozenset({"choice", "noul"})
    assert WORKFLOW.split_names() == ("train", "val", "test_iid", "test_ood")
    assert WORKFLOW.available_locales() == ["en", "zh"]
    assert WORKFLOW.dimension_titles() == dict(af_workflow.DIMENSION_TITLES)


def test_config_for_rejects_unknown_split() -> None:
    """输入：不存在的切分。输出：无。预期行为：报错并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        WORKFLOW.config_for("nope")


def test_config_for_swaps_the_split_name() -> None:
    """输入：三个同分布切分。输出：无。

    预期行为：每次返回的配置里 `split` 都是调用方要的那一档——`ID_CONFIG` 是单例，
    直接返回它会让三档切出一模一样的数据。
    """
    for split in ("train", "val", "test_iid"):
        assert WORKFLOW.config_for(split).split == split
    assert WORKFLOW.config_for("test_ood").ood_only is True


def test_make_judges_rejects_unknown_name() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：报错并列出可用名字。"""
    with pytest.raises(KeyError, match="未知判别器"):
        WORKFLOW.make_judges(("nope",), "zh")
    assert [judge.name for judge in WORKFLOW.make_judges(("roundtrip", "label"), "zh")] == [
        "roundtrip",
        "label",
    ]


def test_bind_backend_is_refused() -> None:
    """输入：一个假后端。输出：无。

    预期行为：纯构造式 workflow 收到后端说明接线错了，直接报错——静默忽略会让
    「以为接上了」的错误拖到跑完才发现。
    """
    with pytest.raises(AssertionError, match="纯构造式"):
        WORKFLOW.bind_backend(object())


def test_callout_index_rejects_a_word_in_two_kinds() -> None:
    """输入：同一个关键词挂在两个种类下的判据表。输出：无。

    预期行为：报 ValueError 并点名那个词——反推种类有歧义时宁可炸，也不要静默取一个。
    """
    bad = {"callout_word": {"note": ["备注"], "tip": ["备注", "提示"]}}
    with pytest.raises(ValueError, match="备注"):
        parse.callout_index(bad)


def test_build_cases_skips_a_shape_without_an_allowed_family() -> None:
    """输入：只要 `code` 形状、但命令族只开 `na` 的配置。输出：无。

    预期行为：这种组合一个合法 case 都没有，直接**跳过**这个形状，而不是改写成别的形状
    ——改写成别的形状会让「形状」这一维的覆盖报告撒谎。
    """
    assert sp.build_cases(sp.DocConfig(shapes=("code",), code_families=("na",))) == []


def test_missing_slot_table_reports_plainly(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：语言文件里没有 `[slot]` 表。输出：无。

    预期行为：报 RuntimeError，不静默留下 `{xxx}` 占位符（那会一路渲染进产物）。
    """
    monkeypatch.setattr(sp, "_SLOT_TABLE", {})
    monkeypatch.setattr(locales, "load", lambda lang: {})
    with pytest.raises(RuntimeError, match="slot"):
        sp._slot_table()


def test_render_reports_a_leftover_placeholder() -> None:
    """输入：模板里带没人填的占位符。输出：无。

    预期行为：报 ValueError 并点名占位符——渲染完还剩 `{xxx}` 说明语言文件与计划对不上。
    """
    loc = {"mark": {"sentence_join": ""}, "x": {"y": ["正文里有 {bogus} 这种没人填的槽位。"]}}
    plan = (sp.BlockPlan(kind=sp.KIND_PARAGRAPH, parts=(sp.Part("x", "y", 0),)),)
    spec = sp.DocSpec(_make_case(), "t", 0, 0, plan)
    with pytest.raises(ValueError, match="占位符"):
        render.render_state(spec, loc)


@pytest.mark.parametrize("text", ("abc" + " " * 99 + "def", "abc" + " " * 95 + "d"))
def test_wrap_text_never_emits_a_short_non_final_line(text: str) -> None:
    """输入：中间有一长串空格的文本（尾巴长到能独立成行 / 短到被整个吞掉两种）。输出：无。

    预期行为：断点落进空档时，宁可让这一行**超宽**也要超过短行上限——非末行一旦短过
    `SHORT_LIMIT`，反解会把它读成标题，整块的切分就错了。拼回去仍等于原文。
    """
    lines = render.wrap_text(text, 64)
    for line in lines[:-1]:
        assert len(line) > parse.SHORT_LIMIT
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")


def test_coordinates_report_unknown_lines_as_none() -> None:
    """输入：一份渲染结果与几个不在文档里的行标识。输出：无。

    预期行为：查不到的行标识一律给 None / False，不抛异常、也不猜一个块出来。
    """
    rendered = render.layout(sp.sample_spec(_make_case(shape="mixed", variant="warning", code_family="build"), "t", 0, 0), LOC)
    blank = next(ident for ident, text in zip(rendered.ids, rendered.texts, strict=True) if not text)
    assert rendered.line_number("L999") is None
    assert rendered.block_index_of("L999") is None
    assert rendered.block_index_of(blank) is None
    assert rendered.same_block(blank, rendered.ids[0]) is False
    parsed = parse.parse_state(rendered.state(), LOC)
    assert parsed.block_index("L999") is None
    assert parsed.same_block("L999", "L001") is False


def test_label_judge_rejects_a_question_id_mismatch() -> None:
    """输入：题号与规格不符的样本。输出：无。

    预期行为：报「题号与规格不符」——按题号逐题比对的判别器不许拿别的题顶上。
    """
    row, spec = _row(_make_case(shape="list", variant="ordered"))
    first = next(iter(row["questions"]))
    row["questions"] = {
        ("zzz" if qid == first else qid): value for qid, value in row["questions"].items()
    }
    row["answers"]["zzz"] = row["answers"].pop(first)
    verdict = judges.LabelJudge("zh").judge(row, spec)
    assert not verdict.ok
    assert "题号与规格不符" in verdict.reason


def test_audit_reports_a_missing_artifact(tmp_path: Path) -> None:
    """输入：空目录。输出：缺失项。预期行为：说「产物不存在」，绝不静默通过。"""
    report, missing = workflow_mod.load("autoformat").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


def test_audit_counts_a_row_that_cannot_be_parsed(tmp_path: Path) -> None:
    """输入：一段不是这份契约的 state。输出：缺失项。预期行为：说「反解不出来」。"""
    jsonio.write_jsonl(
        tmp_path / "train.jsonl",
        [{"id": "autoformat-train-0000000", "state": "随便一段话", "questions": {}, "answers": {}}],
    )
    report, missing = workflow_mod.load("autoformat").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 1
    assert report["train"]["label_mismatch"] == 1
    assert any("反解不出来" in item for item in missing)


def test_audit_catches_an_answer_that_contradicts_the_text(tmp_path: Path) -> None:
    """输入：把一条 noul 答案翻转过的产物。输出：缺失项。

    预期行为：产物级重算与产物里的答案对不上就记一笔——审计是「事后也能查」的那道闸。
    """
    row, _spec = _row(_make_case())
    qid = next(qid for qid, q in row["questions"].items() if q["type"] == "noul")
    row["answers"][qid]["noul"] = not row["answers"][qid]["noul"]
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = workflow_mod.load("autoformat").audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("重算" in item for item in missing)


def test_label_judge_rejects_an_answer_that_disagrees_with_the_spec() -> None:
    """输入：答案被改过的样本。输出：无。

    预期行为：报「按规格应为 …」——判别器拿真值规格重新出题，组装器写错答案时当场红。
    """
    row, spec = _row(_make_case(shape="list", variant="ordered"))
    qid = next(qid for qid, q in row["questions"].items() if q["type"] == "noul")
    row["answers"][qid]["noul"] = not row["answers"][qid]["noul"]
    verdict = judges.LabelJudge("zh").judge(row, spec)
    assert not verdict.ok
    assert "按规格应为" in verdict.reason


def test_audit_counts_an_out_of_range_choice_index(tmp_path: Path) -> None:
    """输入：choice 答案编号越界的产物。输出：缺失项。

    预期行为：编号越界单独记一笔并点名——编号越界等于答案指不到任何选项。
    """
    row, _spec = _row(_make_case(shape="list", variant="ordered"))
    qid = next(qid for qid, q in row["questions"].items() if q["type"] == "choice")
    row["answers"][qid]["choice"] = 999
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = workflow_mod.load("autoformat").audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


def test_audit_counts_an_unknown_family(tmp_path: Path) -> None:
    """输入：一条多出未知题族的行。输出：报告。

    预期行为：未知题族被点名——题号写错时不许静默出一批错答案。
    """
    row, _spec = _row(_make_case())
    row["questions"]["made_up"] = {
        "type": "noul",
        "instructions": "x",
        "criteria": {"true": "1", "false": "2"},
    }
    row["answers"]["made_up"] = {"type": "noul", "noul": True}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = workflow_mod.load("autoformat").audit(tmp_path, ("train",), "zh")
    assert report["train"]["unknown_families"] == {"made_up": 1}
    assert any("未知题族" in item for item in missing)
