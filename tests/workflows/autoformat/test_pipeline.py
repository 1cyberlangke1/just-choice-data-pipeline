"""结构恢复的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from jc import pipeline
from jc.jsontypes import JsonValue
from tests.workflows._scaffold import run_config
from tests.workflows.autoformat.conftest import FakeBackend
from workflows.autoformat import features, locales, workflow as af_workflow
from workflows.autoformat.features import DocCase
from workflows.autoformat.spec import DocSpec
from workflows.autoformat.workflow import AutoformatConfig, AutoformatWorkflow, bucket_of

#: 全空间 345 组。
ALL_CASES = features.build_cases(features.DocConfig())

#: 小批量试跑的条数上限。train 切片一共 54 组，所以实际跑满的是 10 条——断言一律拿
#: `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train", limit: int = CASE_LIMIT
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。

    预期行为：**关掉语言闸**（`language_gate=False`）。本包的假后端写的是英文骨架——它们是
    挑出来「长度同时落在两种语言预算里」的那几句（见 `conftest` 的注释），所以 zh 侧的正文
    不是汉字。真机 zh 产物不是这样（模型写中文），这是**夹具的局限**、不是渲染的问题；
    语言闸本身由 `tests/core` 的定点测试与 6 家认 `locale` 的假后端在语言扫描里验。
    """
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="autoformat",
        split=split,
        seed=7,
        override=AutoformatConfig(split=split, case_limit=limit),
        language_gate=False,
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[DocCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return af_workflow.cases_for(AutoformatConfig(split=split, case_limit=limit))


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


def test_a_contract_failure_is_rewritten_with_the_error(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：第一次把段落写短、第二次写对的假后端。输出：RunResult。

    预期行为：渲染期自己带着失败理由重写一次就够——全部 case 都收下、零拒绝，管线那层
    `render_attempts` 一次都不用动（`render_retries` 为 0），而且重写那次的提示词里带着
    `check_draft` 的原话。真机 `af-cnt-b-zh` 那 5 条 `E_ASSEMBLE` 全是字数（58 / 55 / 73），
    对模型是**计数任务**，只换骰子不告诉它差多少是修不掉的。
    """
    backend = fake_backend(short_paragraph_first=True)
    result = pipeline.run(make_config(tmp_path, "a4-fix", backend))
    assert result.rejected == []
    assert result.accepted == len(cases_of("train"))
    assert result.stats["render_retries"] == 0
    header = str(locales.load("zh")["doc"]["fix_header"])
    retold = [prompt for prompt in backend.prompts if header in prompt]
    assert retold, "重写那次的提示词里必须带着失败理由"
    assert all("长度" in prompt for prompt in retold)


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
