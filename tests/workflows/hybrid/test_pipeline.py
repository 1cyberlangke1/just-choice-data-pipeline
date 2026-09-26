"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

from jc import assemble, cli, jsonio, pipeline
from jc import workflow as workflow_mod
from workflows.hybrid import judges as judges_mod
from workflows.hybrid import locales, questions, render
from workflows.hybrid import spec as sp
from workflows.hybrid import workflow as hy_workflow


def tiny_config(split: str = "train") -> sp.HybridConfig:
    """输入：切分名。输出：极小采样参数。

    预期行为：2 场数 × 2 起始时段 × 2 冲突档 × 1 版式 × 2 星期 × 1 时长档
    = 16 个组合（其中 `same_room` + 2 场会被 `compatible` 剔掉），
    同分布切分再按哈希桶切一刀，所以 train 只留下其中一部分。
    """
    return sp.HybridConfig(
        split=split,
        meeting_counts=(2, 3),
        openings=("early", "noon"),
        clashes=("none", "same_room"),
        layouts=("bullet",),
        weekdays=(1, 3),
        duration_mixes=("short",),
        phrasings=(0, 1),
    )


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1, backend: Any = None
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度、后端。输出：指向 tmp 的运行配置。"""
    return pipeline.RunConfig(
        workflow="hybrid",
        split=split,
        locales=("zh",),
        run_seed=11,
        per_case=1,
        concurrency=concurrency,
        out_root=tmp_path,
        run_id=run_id,
        config_override=tiny_config(split),
        backend=backend,
    )


def _case_count(config: pipeline.RunConfig) -> int:
    """输入：运行配置。输出：该切分的 case 数。预期行为：与产物条数对得上。"""
    return len(hy_workflow.HybridWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: Any) -> None:
    """输入：极小配置 + 假后端。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "h1", backend=fake_backend())
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"skeleton", "answers", "text"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("hybrid").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0 and report["train"]["options_bad"] == 0


def test_two_runs_are_byte_identical(tmp_path: Path, fake_backend: Any) -> None:
    """输入：同种子、不同并发度跑两次。输出：两份产物。

    预期行为：逐字节一致——RNG 由 (seed, workflow, split, seq) 派生，并发不改变内容。
    这条对 hybrid 尤其重要：文本写在 `spec.written` 上，一旦挂错地方就会串味。
    """
    serial = pipeline.run(make_config(tmp_path, "h2a", concurrency=1, backend=fake_backend()))
    parallel = pipeline.run(make_config(tmp_path, "h2b", concurrency=4, backend=fake_backend()))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        parallel.run_dir / "train.jsonl"
    ).read_bytes()


def test_manifest_and_stats_are_written(tmp_path: Path, fake_backend: Any) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分 / case 列表 / 「需要 text 能力」；
    stats 报告里出现 6 个维度的中文标题与三个判别器的通过率。
    """
    result = pipeline.run(make_config(tmp_path, "h3", backend=fake_backend()))
    manifest = json.loads((result.run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "hybrid" and manifest["needs"] == ["text"]
    assert manifest["split"] == "train" and manifest["cases"]
    stats = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    for title in hy_workflow.DIMENSION_TITLES.values():
        assert title in stats
    for name in ("skeleton", "answers", "text"):
        assert name in stats


def test_a_missing_backend_is_refused(tmp_path: Path) -> None:
    """输入：不给后端。输出：ValueError / RuntimeError。

    预期行为：`needs` 非空的 workflow 缺后端要立刻报错——绝不静默降级成
    「没有文本外壳的样本」（那会产出一整个假数据集）。
    """
    with pytest.raises(ValueError, match="声明需要后端"):
        pipeline.run(make_config(tmp_path, "h4"))
    workflow = hy_workflow.HybridWorkflow()
    spec = sp.sample_spec(
        random.Random(1),
        sp.HybridCase(2, "early", "none", "bullet", 1, "short"),
        "train",
        0,
        0,
        0,
    )
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(1), "zh")


def test_bind_backend_stores_the_backend(fake_backend: Any) -> None:
    """输入：一个后端。输出：无。预期行为：存下来给 `render` 用。"""
    workflow = hy_workflow.HybridWorkflow()
    backend = fake_backend()
    workflow.bind_backend(backend)
    assert workflow.backend is backend


def test_config_for_rejects_an_unknown_split() -> None:
    """输入：不存在的切分名。输出：KeyError。预期行为：不猜、不回落。"""
    with pytest.raises(KeyError, match="未知切分"):
        hy_workflow.HybridWorkflow().config_for("test_whatever")


def test_samples_draw_theme_directions_from_the_config() -> None:
    """输入：只给两个主题方向的采样参数。输出：各样本的规格。

    预期行为：`plan_sample` 从配置里抽方向、`sample` 原样放进 spec——两个方向都要抽得到，
    且 spec 里存的就是抽中的那个。接线断了的话提示词会永远用同一个方向，
    标题就会塌成几句套话（那正是加这个旋钮要治的病）。
    """
    workflow = hy_workflow.HybridWorkflow()
    config = dataclasses.replace(tiny_config("train"), themes=(3, 7))
    cases = workflow.build_cases(config)
    seen: set[int] = set()
    for seq in range(12):
        plan = workflow.plan_sample(random.Random(seq), config, cases[seq % len(cases)], seq)
        assert plan.params["theme"] in (3, 7)
        spec = workflow.sample(random.Random(seq), plan)
        assert spec.theme == plan.params["theme"]
        seen.add(spec.theme)
    assert seen == {3, 7}


def test_a_config_with_no_cases_is_refused(tmp_path: Path, fake_backend: Any) -> None:
    """输入：一个「所有组合都写不出来」的采样参数。输出：ValueError（人话）。

    预期行为：`compatible` 剔完一个 case 都不剩时要**当场报错**，不能让管线拿着空表
    往下跑——实测那样会抛 `ZeroDivisionError: integer modulo by zero`，看不出哪一步坏了。
    """
    config = pipeline.RunConfig(
        workflow="hybrid",
        split="train",
        locales=("zh",),
        out_root=tmp_path,
        run_id="h5",
        config_override=sp.HybridConfig(split="train", meeting_counts=(1,)),
        backend=fake_backend(),
    )
    with pytest.raises(ValueError, match="一个 case 都没有"):
        pipeline.run(config)


def test_splits_are_disjoint_and_ood_holds_out_values() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：同分布三档两两不相交（组合留出）；`test_ood` 每个 case 都含没见过的取值。
    """
    workflow = hy_workflow.HybridWorkflow()
    buckets = {
        split: {case.describe() for case in workflow.build_cases(workflow.config_for(split))}
        for split in workflow.split_names()
    }
    for left in ("train", "val", "test_iid"):
        for right in ("train", "val", "test_iid"):
            if left != right:
                assert not (buckets[left] & buckets[right])
    assert buckets["test_ood"]
    assert all(
        case.unseen_dims()
        for case in workflow.build_cases(workflow.config_for("test_ood"))
    )


def test_bucket_of_is_stable_and_case_label_roundtrips() -> None:
    """输入：一个 case。输出：桶号 / 标签 / 维度。

    预期行为：同一组合永远同桶（不用内置 `hash()`）；维度标题覆盖全部维度。
    """
    workflow = hy_workflow.HybridWorkflow()
    case = sp.HybridCase(3, "noon", "overlap", "bullet", 5, "long")
    assert hy_workflow.bucket_of(case) == hy_workflow.bucket_of(case)
    assert 0 <= hy_workflow.bucket_of(case) < 100
    assert workflow.case_label(case) == case.describe()
    assert set(workflow.stats_dimensions(case)) == set(workflow.dimension_titles())


def test_reparse_reads_a_rendered_schedule(make_spec: Any) -> None:
    """输入：一份渲染好的日程。输出：反解结果。

    预期行为：`Workflow.reparse` 没有语言参数，靠结构与合并标签表读出来。
    """
    spec = make_spec(3, meetings=3)
    state = render.build(spec, locales.load("zh"))
    parsed = hy_workflow.HybridWorkflow().reparse(state)
    assert parsed.date == spec.date
    assert len(parsed.meetings) == 3
    assert parsed.meetings[0].title == spec.text_of("m1").title


def test_available_locales() -> None:
    """输入：无。输出：语言列表。预期行为：与语言文件目录一致。"""
    assert hy_workflow.HybridWorkflow().available_locales() == locales.available()


def test_judges_are_the_declared_defaults() -> None:
    """输入：无。输出：workflow 声明。

    预期行为：三种题型都产、需要 `text` 能力、默认三个规则判别器。
    """
    workflow = hy_workflow.HybridWorkflow()
    assert workflow.kinds == frozenset({"choice", "noul", "score"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("skeleton", "answers", "text")
    assert workflow.name == "hybrid" and workflow.version == hy_workflow.WORKFLOW_VERSION
    assert isinstance(judges_mod.TextJudge(), object)


def _row(state: str, questions: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    """造一条最小样本行（审计只读 state / questions / answers）。"""
    return {
        "id": "hybrid-train-0000000",
        "state": state,
        "questions": questions,
        "answers": answers,
    }


STATE = (
    "会议日程 · 2026-09-16\n\n"
    "1. 09:00–10:00 · 海棠厅 · 王敏\n   标题：甲会\n   议题：甲的议题\n"
    "2. 11:00–12:00 · 青松厅 · 李强\n   标题：乙会\n   议题：乙的议题\n"
)


def test_audit_reports_a_missing_artifact(tmp_path: Path) -> None:
    """输入：空目录。输出：(报告, 缺失项)。

    预期行为：缺失的切分要明说「产物不存在」，而不是当成「0 条样本、审计通过」。
    """
    report, missing = workflow_mod.load("hybrid").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


def test_audit_catches_a_bad_index_and_a_wrong_option_count(tmp_path: Path) -> None:
    """输入：一条「答案编号越界」+「选项个数与场数对不上」+「未知题族」的产物行。
    输出：(报告, 缺失项)。预期行为：三件事都要报出来，缺掉的题族也要点名。
    """
    questions_map = {
        "duration_of": {
            "type": "choice",
            "instructions": "第 1 场会议开了几个小时？",
            "criteria": {"h1": "1 小时", "h2": "2 小时", "h3": "3 小时"},
        },
        "made_up": {"type": "choice", "instructions": "y", "criteria": {"a": "1"}},
    }
    answers = {
        "duration_of": {"type": "choice", "choice": 5},
        "made_up": {"type": "choice", "choice": 0},
    }
    jsonio.write_jsonl(tmp_path / "train.jsonl", [_row(STATE, questions_map, answers)])
    report, missing = workflow_mod.load("hybrid").audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert report["train"]["options_bad"] == 2
    assert any("答案编号越界" in item for item in missing)
    assert any("选项个数与日程场数对不上" in item for item in missing)
    assert any("未知题族" in item for item in missing)
    assert any("题族 earliest_start 一条都没出" in item for item in missing)


def test_audit_passes_a_self_consistent_artifact(tmp_path: Path, make_spec: Any) -> None:
    """输入：一条自己组装出来的、自洽的产物行。输出：(报告, 缺失项)。

    预期行为：零缺失——题族齐全、答案编号在范围内、选项个数与日程场数自洽。
    """
    spec = make_spec(13, meetings=3)
    rng = random.Random(13)
    labeled = questions.build(spec, locales.load("zh"), spec.phrasing, rng)
    row = assemble.assemble_sample(
        "hybrid-train-0000000", render.build(spec, locales.load("zh")), labeled, rng
    )
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = workflow_mod.load("hybrid").audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == 1
    assert report["train"]["options_bad"] == 0
    assert report["train"]["families"]["title_of"] == 1


def test_cli_lists_cases_and_audits(
    tmp_path: Path, fake_backend: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """输入：`jc cases` 与 `jc audit`。输出：退出码与 stdout。

    预期行为：退出码 0，审计打印里出现 hybrid 自己的计数键——通用层的审计打印必须
    按产物报的键走；写死某个 workflow 的键名时，别的 workflow 一跑审计就 KeyError（实测踩过）。
    """
    assert cli.main(["cases", "--workflow", "hybrid", "--split", "train"]) == 0
    assert "个 case" in capsys.readouterr().out
    result = pipeline.run(make_config(tmp_path, "h5", backend=fake_backend()))
    code = cli.main(
        ["audit", "--workflow", "hybrid", "--dir", str(result.run_dir), "--split", "train"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "审计通过" in out
    assert "options_bad" in out and "index_bad" in out
