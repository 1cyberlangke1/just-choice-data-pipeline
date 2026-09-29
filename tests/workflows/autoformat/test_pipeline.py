"""结构恢复的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod
from jc.jsontypes import JsonValue
from tests.workflows.autoformat.conftest import FakeBackend
from workflows.autoformat import features, locales, workflow as af_workflow
from workflows.autoformat.features import DocCase
from workflows.autoformat.spec import DocSpec, spec_to_dict
from workflows.autoformat.workflow import AutoformatConfig, AutoformatWorkflow, bucket_of

#: 全空间 345 组。
ALL_CASES = features.build_cases(features.DocConfig())

#: 小批量试跑的条数上限。train 切片一共 54 组，所以实际跑满的是 10 条——断言一律拿
#: `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")

#: 审计报告里那五个「不自洽」计数器。
BAD_KEYS = ("state_bad", "join_bad", "block_type_bad", "callout_bad", "ordered_bad")


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train", limit: int = CASE_LIMIT
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="autoformat",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=AutoformatConfig(split=split, case_limit=limit),
        backend=backend,
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[DocCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return af_workflow.cases_for(AutoformatConfig(split=split, case_limit=limit))


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，四族题都真的
    出得来。
    """
    result = pipeline.run(make_config(tmp_path, "a1", fake_backend()))
    expected = len(cases_of("train"))
    assert result.accepted == expected
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "roundtrip", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    assert result.stats["conflicts"] == 0
    report, missing = workflow_mod.load("autoformat").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == expected
    for family in ("join", "block_type", "callout_kind", "list_ordered"):
        assert report["train"]["families"][family] > 0, family
    for key in BAD_KEYS:
        assert report["train"][key] == 0, key


def test_two_runs_are_byte_identical(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：同种子跑两次。输出：两份产物。预期行为：逐字节一致。"""
    first = pipeline.run(make_config(tmp_path, "a2a", fake_backend()))
    second = pipeline.run(make_config(tmp_path, "a2b", fake_backend()))
    assert (first.run_dir / "train.jsonl").read_bytes() == (
        second.run_dir / "train.jsonl"
    ).read_bytes()


@pytest.mark.parametrize(
    "switch", ["heading_punct", "short_paragraph", "drop_text"]
)
def test_drafts_that_break_every_case_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str
) -> None:
    """输入：每一组 case 都会踩到的「写歪」。输出：RunResult。

    预期行为：一条都留不下，拒绝码是渲染期的 `E_ASSEMBLE`（重试耗尽后仍被拒，不静默放行）。
    """
    result = pipeline.run(make_config(tmp_path, f"a3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_a_lying_reader_is_rejected_by_trace(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解时少报一块。输出：RunResult。

    预期行为：一条都留不下，拒绝码是 `E_JUDGE_TRACE`——机械口径全对，只有读者模型看得见这种错。
    """
    result = pipeline.run(make_config(tmp_path, "a4", fake_backend(trace_lies=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


@pytest.mark.parametrize(
    ("switch", "predicate"),
    [
        ("code_wrong_tool", lambda case: case.shape not in ("code", "mixed")),
        ("callout_wrong_word", lambda case: case.shape not in ("callout", "mixed")),
    ],
)
def test_a_switch_that_only_touches_some_shapes_spares_the_rest(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, predicate: object
) -> None:
    """输入：只影响带代码块 / 带提示框的那几种形状的「写歪」。输出：RunResult。

    预期行为：本来就没有那一类块的 case 照样合格，其余全被 `E_ASSEMBLE` 拦下——一个都不放过。
    """
    unaffected = sum(1 for case in cases_of("train") if predicate(case))
    result = pipeline.run(make_config(tmp_path, f"a5-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow。输出：无。预期行为：抛 RuntimeError，不静默降级成模板。"""
    workflow = AutoformatWorkflow()
    case = cases_of("train")[0]
    config = workflow.config_for("train")
    spec = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    assert isinstance(spec, DocSpec)
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_trace_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却要 trace 判别器。输出：无。预期行为：抛 RuntimeError（接线错了）。"""
    with pytest.raises(RuntimeError, match="trace"):
        AutoformatWorkflow().make_judges(("trace",), "zh")


def test_unknown_split_and_judge_and_locale_report_available() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：都报人话。"""
    workflow = AutoformatWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


def test_config_knobs_are_validated() -> None:
    """输入：几组坏旋钮。输出：无。预期行为：构造期就报人话，点名是哪个字段。"""
    with pytest.raises(ValueError, match="cover_limit"):
        AutoformatConfig(cover_limit=0)
    with pytest.raises(ValueError, match="case_limit"):
        AutoformatConfig(case_limit=-1)
    with pytest.raises(ValueError, match="topics"):
        AutoformatConfig(topics=())


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：块计划与切分、进程无关。"""
    case = cases_of("train")[0]
    config = AutoformatConfig(split="train", phrasing_pool=(1,))
    workflow = AutoformatWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, DocSpec) and isinstance(second, DocSpec)
    assert first.plan == second.plan
    assert first.plan == features.build_plan(case)
    assert spec_to_dict(first)["kinds"] == list(features.build_plan(case).kinds())


def test_reparse_and_case_label_and_dimensions_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = AutoformatWorkflow()
    case = DocCase("build", 64, "code", "na", "vcs")
    plan = features.build_plan(case)
    from tests.workflows.autoformat.conftest import texts_for
    from workflows.autoformat import render

    loc = locales.load("zh")
    state = render.render_state(plan, texts_for(plan, loc), loc)
    assert workflow.reparse(state).kinds() == plan.kinds()
    assert workflow.case_label(case) == case.describe() == "build/64/code/na/vcs"
    assert workflow.stats_dimensions(case) == case.as_dict()
    assert workflow.dimension_titles()["shape"]


def test_locales_are_reported() -> None:
    """输入：无。输出：无。预期行为：支持的语言与语言文件目录一致。"""
    workflow = AutoformatWorkflow()
    assert workflow.available_locales() == locales.available() == ["en", "zh"]


def test_fake_drafts_fit_every_language() -> None:
    """输入：全空间 345 组 × 两种语言。输出：无。

    预期行为：假后端那几段骨架在**每一种语言**里都过 `check_draft`——字符预算是按语言给的
    （`[limit]`），英文的「短标题」比中文长一截，一份骨架能喂中文不代表能喂英文。
    """
    from tests.workflows.autoformat.conftest import texts_for
    from workflows.autoformat import render

    for lang in locales.available():
        loc = locales.load(lang)
        for case in ALL_CASES:
            plan = features.build_plan(case)
            render.check_draft(plan, texts_for(plan, loc), loc)


def test_declared_contract_fields() -> None:
    """输入：workflow 的声明项。输出：无。预期行为：题型、后端需求与判别器名单都对得上。"""
    workflow = AutoformatWorkflow()
    assert workflow.name == "autoformat"
    assert workflow.kinds == frozenset({"choice", "noul"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("hygiene", "roundtrip", "trace")
    assert workflow.split_names() == ALL_SPLITS


def test_bucket_of_is_stable_and_the_id_sides_exclude_the_held_out_values() -> None:
    """输入：同分布三个切分与 `test_ood`。输出：无。

    预期行为：桶号只看组合（同组合永远同切分）；同分布侧**一个留出取值都没有**，OOD 侧每条至少
    含一个——两边按构造不重叠。
    """
    cases = features.build_cases(features.DocConfig())
    assert all(bucket_of(case) == bucket_of(case) for case in cases[:5])
    for split in ID_SPLITS:
        for case in cases_of(split, limit=0):
            assert case.unseen_dims() == (), f"{split} {case.describe()}"
    ood = cases_of("test_ood", limit=0)
    assert ood
    assert all(case.unseen_dims() for case in ood)
    id_cases = {case for split in ID_SPLITS for case in cases_of(split, limit=0)}
    assert not (set(ood) & id_cases)


def test_cover_pick_covers_every_value_of_every_dimension() -> None:
    """输入：同分布三个切分的小批量挑选。输出：无。

    预期行为：10 条就把每一维的**每一个取值**都覆盖到——小批量里最容易漏的正是「块数」与
    「形状」：纯截断的话 10 条会全落在 `mixed` 上，伴随题族当场「一条都没出」。
    """
    for split in ID_SPLITS:
        full = _dim_values(cases_of(split, limit=0))
        picked = _dim_values(cases_of(split))
        for dim, values in full.items():
            assert picked[dim] == values, f"{split} 的 {dim} 没覆盖全"


def _dim_values(cases: list[DocCase]) -> dict[str, set[str]]:
    """输入：case 表。输出：{维度名: 出现过的取值}（比 `as_dict` 多一个「块数」）。"""
    out: dict[str, set[str]] = {}
    for case in cases:
        for key, value in AutoformatWorkflow().stats_dimensions(case).items():
            out.setdefault(str(key), set()).add(str(value))
        out.setdefault("blocks", set()).add(str(len(features.build_plan(case).slots)))
    return out


def test_stats_are_written_and_readable(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：一小批跑完。输出：无。

    预期行为：统计报告与 manifest 都落盘、内容对得上——报告是给人看产物质量的唯一入口。
    """
    result = pipeline.run(make_config(tmp_path, "a6", fake_backend()))
    stats_md = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    assert "答案均衡" in stats_md
    assert "判别器通过率" in stats_md
    manifest: JsonValue = json.loads(
        (result.run_dir / "manifest-train.json").read_text(encoding="utf-8")
    )
    assert manifest["workflow"] == "autoformat"
    assert manifest["case_count"] == len(cases_of("train"))
    assert len(manifest["cases"]) == manifest["case_count"]
