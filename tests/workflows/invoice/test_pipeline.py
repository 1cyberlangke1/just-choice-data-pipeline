"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from jc import assemble, jsonio
from jc import pipeline, workflow as workflow_mod
from jc import cli
from workflows.invoice import judges as judges_mod
from workflows.invoice import locales, questions, render
from workflows.invoice import spec as sp
from workflows.invoice import workflow as inv_workflow


def tiny_config(split: str = "train") -> sp.InvoiceConfig:
    """输入：切分名。输出：极小采样参数。

    预期行为：1 版式 × 1 标签风格 × 1 货币 × 1 日期格式 × 1 行数 × 2 错处 × 2 小计 × 2 税额
    = 8 个组合（同分布切分再按哈希桶切一刀，所以 train 只留下其中一部分）。
    """
    return sp.InvoiceConfig(
        split=split,
        layouts=("labeled_lines",),
        label_styles=("canonical",),
        currencies=("cny",),
        date_formats=("dashed",),
        line_counts=(2,),
        defect_counts=(0, 1),
        subtotal_flags=(True, False),
        tax_states=("none", "ok"),
        phrasings=(0, 1),
    )


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="invoice",
        split=split,
        locales=("zh",),
        run_seed=11,
        per_case=1,
        concurrency=concurrency,
        out_root=tmp_path,
        run_id=run_id,
        config_override=tiny_config(split),
    )


def _case_count(config: pipeline.RunConfig) -> int:
    """输入：运行配置。输出：该切分的 case 数。预期行为：与产物条数对得上。"""
    return len(inv_workflow.InvoiceWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：极小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "i1")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("invoice").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["tax_mismatch"] == 0


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """输入：同种子、不同并发度跑两次。输出：两份产物。

    预期行为：逐字节一致——RNG 由 (seed, workflow, split, seq) 派生，并发不改变内容。
    """
    serial = pipeline.run(make_config(tmp_path, "i2a", concurrency=1))
    parallel = pipeline.run(make_config(tmp_path, "i2b", concurrency=4))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        parallel.run_dir / "train.jsonl"
    ).read_bytes()


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分 / case 列表 / 「不需要后端」；
    stats 报告里出现 8 个维度的中文标题与两个判别器的通过率。
    """
    result = pipeline.run(make_config(tmp_path, "i3"))
    manifest = json.loads((result.run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "invoice" and manifest["needs"] == []
    assert manifest["split"] == "train" and manifest["cases"]
    stats = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    for title in inv_workflow.DIMENSION_TITLES.values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_needs_no_backend_and_rejects_being_bound(tmp_path: Path) -> None:
    """输入：无后端 / 硬塞一个后端。输出：跑通 / AssertionError。

    预期行为：纯构造式的 workflow 不初始化任何后端；真被绑定说明接线错了，直接报错。
    """
    workflow = inv_workflow.InvoiceWorkflow()
    assert workflow.needs == frozenset()
    with pytest.raises(AssertionError, match="纯构造式"):
        workflow.bind_backend(object())
    assert pipeline.run(make_config(tmp_path, "i4")).accepted > 0


def test_config_for_rejects_an_unknown_split() -> None:
    """输入：不存在的切分名。输出：KeyError。预期行为：报错并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        inv_workflow.InvoiceWorkflow().config_for("nope")


def test_make_judges_rejects_an_unknown_name() -> None:
    """输入：不存在的判别器名。输出：KeyError。预期行为：绝不静默少跑一个判别器。"""
    with pytest.raises(KeyError, match="未知判别器"):
        inv_workflow.InvoiceWorkflow().make_judges(("nope",), "zh")
    assert [judge.name for judge in inv_workflow.InvoiceWorkflow().make_judges(("roundtrip", "label"), "en")] == [
        "roundtrip",
        "label",
    ]


def test_splits_hold_out_combinations_and_values() -> None:
    """输入：四个切分。输出：各自的 case 列表。

    预期行为：同分布三档一个留出取值都不含、三档之间互不重叠（切分单位是**组合**）；
    OOD 档每条至少含一个留出取值，且与三档没有交集。
    """
    workflow = inv_workflow.InvoiceWorkflow()
    id_cases: list[sp.InvoiceCase] = []
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases
        assert all(not case.unseen_dims() for case in cases)
        id_cases.extend(cases)
    assert len(set(id_cases)) == len(id_cases)
    ood = workflow.build_cases(workflow.config_for("test_ood"))
    assert ood and all(case.unseen_dims() for case in ood)
    assert not (set(ood) & set(id_cases))


def test_case_label_and_dimensions() -> None:
    """输入：一个 case。输出：可读描述与统计维度。

    预期行为：`case_label` 一行说清 8 个维度（小计对/错写成中文，其余照抄取值）；
    `stats_dimensions` 与 `case.as_dict()` 一致，摊平成 8 栏。
    """
    workflow = inv_workflow.InvoiceWorkflow()
    case = workflow.build_cases(workflow.config_for("train"))[0]
    label = workflow.case_label(case)
    dims = workflow.stats_dimensions(case)
    assert set(dims) == set(inv_workflow.DIMENSION_TITLES)
    assert dims == case.as_dict()
    for token in (case.layout, case.labels, case.currency, case.date_format, case.tax_state):
        assert token in label
    assert f"{case.lines}行" in label and f"{case.defects}处错" in label
    assert ("小计对" if case.subtotal_ok else "小计错") in label


def test_plan_sample_and_sample_agree() -> None:
    """输入：配置与 case。输出：生成计划与规格。

    预期行为：问法从配置的问法池里抽、case 与序号进 params，`sample` 不再掷骰子
    （同一种子必得同一份规格）。
    """
    workflow = inv_workflow.InvoiceWorkflow()
    config = tiny_config()
    case = workflow.build_cases(config)[0]
    for seed in range(5):
        plan = workflow.plan_sample(random.Random(seed), config, case, 3)
        assert plan.name == "train"
        assert plan.params["case"] is case and plan.params["seq"] == 3
        assert plan.params["phrasing"] in config.phrasings
        spec = workflow.sample(random.Random(seed), plan)
        assert spec.phrasing == plan.params["phrasing"] and spec.seq == 3


def test_render_and_ask_never_touch_a_backend() -> None:
    """输入：规格、RNG、语言。输出：文本与题目。

    预期行为：纯构造式 workflow 的 `render` / `ask` 完全由 spec 推出——
    `render` 连 RNG 都不用（换一个 RNG 结果一样），`ask` 同种子必得同结果。
    """
    workflow = inv_workflow.InvoiceWorkflow()
    spec = sp.sample_spec(
        random.Random(5),
        sp.InvoiceCase("labeled_lines", "canonical", "cny", "dashed", 2, 1, True, "ok"),
        "train",
        0,
        1,
    )
    assert workflow.render(spec, random.Random(0), "zh") == workflow.render(
        spec, random.Random(9), "zh"
    )
    first = workflow.ask(spec, random.Random(1), "zh")
    assert first == workflow.ask(spec, random.Random(1), "zh")
    assert [item.family for item in first] == list(questions.FAMILIES)


def test_reparse_reads_a_rendered_document() -> None:
    """输入：渲染出来的 state。输出：反解结果。

    预期行为：`Workflow.reparse` 没有语言参数，靠合并标签表把字段与明细读出来。
    """
    spec = sp.sample_spec(
        random.Random(3),
        sp.InvoiceCase("pipe_table", "aliased", "usd", "dotted", 3, 1, True, "ok"),
        "train",
        0,
        0,
    )
    parsed = inv_workflow.InvoiceWorkflow().reparse(render.build(spec, locales.load("zh")))
    assert set(parsed.fields) == set(render.visible_fields(spec))
    assert len(parsed.items) == 3


def test_available_locales() -> None:
    """输入：无。输出：语言列表。预期行为：与语言文件目录一致。"""
    assert inv_workflow.InvoiceWorkflow().available_locales() == locales.available()


def _row(state: str, questions: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    """造一条最小样本行（审计只读 state / questions / answers）。"""
    return {
        "id": "invoice-train-0000000",
        "state": state,
        "questions": questions,
        "answers": answers,
    }


def test_audit_reports_a_missing_artifact(tmp_path: Path) -> None:
    """输入：空目录。输出：(报告, 缺失项)。

    预期行为：缺失的切分要明说「产物不存在」，而不是当成「0 条样本、审计通过」。
    """
    report, missing = workflow_mod.load("invoice").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


def test_audit_catches_an_impossible_tax_question_and_a_bad_index(tmp_path: Path) -> None:
    """输入：一条「没有税额行却问了税额一致性」+「答案编号越界」+「未知题族」的产物行。
    输出：(报告, 缺失项)。预期行为：三件事都要报出来，缺掉的题族也要点名。
    """
    state = "费用结算单据\n小计：¥100\n总计：¥100\n"
    questions_map = {
        "tax_consistent": {
            "type": "noul",
            "instructions": "x",
            "criteria": {"true": "a", "false": "b"},
        },
        "made_up": {"type": "choice", "instructions": "y", "criteria": {"a": "1"}},
    }
    answers = {
        "tax_consistent": {"type": "noul", "noul": True},
        "made_up": {"type": "choice", "choice": 5},
    }
    jsonio.write_jsonl(tmp_path / "train.jsonl", [_row(state, questions_map, answers)])
    _, missing = workflow_mod.load("invoice").audit(tmp_path, ("train",), "zh")
    assert any("tax_consistent 出现在没有税额行的单据上" in item for item in missing)
    assert any("答案编号越界" in item for item in missing)
    assert any("未知题族" in item for item in missing)
    assert any("题族 date_field 一条都没出" in item for item in missing)


def test_audit_passes_a_self_consistent_artifact(tmp_path: Path, make_spec: Any) -> None:
    """输入：一条自己组装出来的、自洽的产物行（含税额行）。输出：(报告, 缺失项)。

    预期行为：零缺失——题族齐全、答案编号都在范围内、有税额行的单据才问税额一致性。
    """
    loc = locales.load("zh")
    spec = make_spec(13, tax_state="ok", lines=2, defects=1)
    rng = random.Random(13)
    labeled = questions.build(spec, loc, spec.phrasing, rng)
    row = assemble.assemble_sample(
        "invoice-train-0000000", render.build(spec, loc), labeled, rng
    )
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = workflow_mod.load("invoice").audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == 1
    assert report["train"]["families"]["tax_consistent"] == 1
    assert report["train"]["tax_mismatch"] == 0


def test_judges_are_the_declared_defaults() -> None:
    """输入：无。输出：workflow 声明。预期行为：题型三种都产、判别器是两个规则判别器。"""
    workflow = inv_workflow.InvoiceWorkflow()
    assert workflow.kinds == frozenset({"choice", "noul", "score"})
    assert workflow.judges == ("roundtrip", "label")
    assert workflow.name == "invoice" and workflow.version == inv_workflow.WORKFLOW_VERSION
    assert isinstance(judges_mod.RoundtripJudge(), object)


def test_cli_audit_prints_the_invoice_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """输入：一次真跑 + `jc audit`。输出：退出码与 stdout。

    预期行为：退出码 0，报告里出现 invoice 自己的计数键——通用层的审计打印必须按
    产物报的键走；写死某个 workflow 的键名时，别的 workflow 一跑审计就 KeyError（实测踩过）。
    """
    result = pipeline.run(make_config(tmp_path, "i6"))
    code = cli.main(
        ["audit", "--workflow", "invoice", "--dir", str(result.run_dir), "--split", "train"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "审计通过" in out
    assert "index_bad" in out and "tax_mismatch" in out
