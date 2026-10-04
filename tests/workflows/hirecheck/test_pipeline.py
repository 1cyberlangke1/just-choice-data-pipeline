"""hirecheck 的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline
from jc.backend.types import ProviderUnavailable
from tests.workflows._scaffold import run_config
from tests.workflows.hirecheck.conftest import FakeBackend, packet_of
from workflows.hirecheck import features, locales, render
from workflows.hirecheck import workflow as hirecheck_workflow
from workflows.hirecheck.features import HireCheckCase
from workflows.hirecheck.spec import HireCheckSpec
from workflows.hirecheck.workflow import HireCheckConfig, HireCheckWorkflow

#: 小批量试跑的条数上限。train 切片一共 19 组，`_cover_pick` 会从里面挑 8 组。
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 8
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")


def make_config(
    tmp_path: Path,
    run_id: str,
    backend: object,
    *,
    split: str = "train",
    limit: int = CASE_LIMIT,
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="hirecheck",
        split=split,
        seed=7,
        override=HireCheckConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[HireCheckCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return hirecheck_workflow.cases_for(HireCheckConfig(split=split, case_limit=limit))


def test_ood_split_is_the_held_out_channel() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：留出侧**只**有 `campus`、同分布侧**一次都不出现** `campus`；四个切分两两不重叠，
    合起来正好是全空间——留出维「只影响渲染、不影响标签」这条不变量靠它守着。
    """
    per_split = {
        split: {
            (case.flags, case.channel)
            for case in hirecheck_workflow.cases_for(HireCheckConfig(split=split))
        }
        for split in ALL_SPLITS
    }
    assert {case[1] for case in per_split["test_ood"]} == {features.CHANNEL_CAMPUS}
    for split in ID_SPLITS:
        assert features.CHANNEL_CAMPUS not in {case[1] for case in per_split[split]}, split
    total = set().union(*per_split.values())
    assert sum(len(cases) for cases in per_split.values()) == len(total)
    assert total == {(value, channel) for value in features.HIT_SETS for channel in features.CHANNELS}


def test_train_covers_every_level() -> None:
    """输入：train 切分的 case 列表。输出：无。

    预期行为：训练侧五档等级齐全——某一档在训练侧取不到时，`grade` 那一族的那个答案就永远学不到。
    这条把 `SPLIT_SEED` 钉住：换种子时它会当场报出来。
    """
    levels = {
        features.level_of(case.tags())
        for case in hirecheck_workflow.cases_for(HireCheckConfig(split="train"))
    }
    assert levels == set(features.LEVELS)


def test_cover_pick_spreads_the_values() -> None:
    """输入：小批量（8 条）。输出：无。

    预期行为：小批量的挑选**优先补齐没见过的取值**——命中组合、渠道与等级各铺开一遍，而不是从列表
    头上截一段（那会让某些档位在小批量里一条都出不来）。
    """
    picked = cases_of("train")
    assert {features.level_of(case.tags()) for case in picked} == set(features.LEVELS)
    assert {case.channel for case in picked} == {features.CHANNEL_REFERRAL, features.CHANNEL_BOARD}


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = HireCheckCase("H2+H3", features.CHANNEL_BOARD)
    config = HireCheckConfig(split="train", phrasing_pool=(1,))
    workflow = HireCheckWorkflow()
    first = workflow.sample(random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0))
    second = workflow.sample(random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0))
    assert isinstance(first, HireCheckSpec) and isinstance(second, HireCheckSpec)
    assert first.plan == second.plan
    assert first.render_task_id == "hirecheck-train-0000000-render"
    assert first.trace_task_id == "hirecheck-train-0000000-trace"


def test_role_knob_reaches_the_prompt() -> None:
    """输入：只留一个岗位方向的采样参数。输出：无。

    预期行为：`plan_sample` 抽出来的方向真的传到了提示词里（范式 §9.57 / §9.58 的样本级变量：它
    不进 case 空间、不进 `stats_dimensions`，只改 `skills:` 怎么列与 `note:` 怎么写）。
    """
    case = HireCheckCase("H1", features.CHANNEL_BOARD)
    workflow = HireCheckWorkflow()
    config = HireCheckConfig(split="train", roles=(3,))
    spec = workflow.sample(random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0))
    assert isinstance(spec, HireCheckSpec)
    assert spec.role == 3
    prompt = render.build_prompt(locales.load("zh"), spec.case, spec.role, spec.years, spec.name)
    assert locales.pick(locales.load("zh")["prompt"]["roles"], 3) in prompt
    assert workflow.stats_dimensions(case) == {"flags": "H1", "channel": "board"}


def test_years_knob_reaches_the_prompt() -> None:
    """输入：只留一个年限的采样参数。输出：无。

    预期行为：`plan_sample` 抽出来的年限真的传到了提示词里（范式 §9.60 的第二个样本级变量：它不进
    case 空间、不进 `stats_dimensions`，只改 `years:` 那一行怎么写）。池子两侧都要有货，所以这里
    给 `(0, 7)`，而这条 case 不命中 `H1`，抽到的必然是 7。
    """
    case = HireCheckCase("H2", features.CHANNEL_BOARD)
    workflow = HireCheckWorkflow()
    config = HireCheckConfig(split="train", years=(0, 7))
    spec = workflow.sample(random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0))
    assert isinstance(spec, HireCheckSpec)
    assert spec.years == 7
    prompt = render.build_prompt(locales.load("zh"), spec.case, spec.role, spec.years, spec.name)
    assert "**7**" in prompt
    assert workflow.stats_dimensions(case) == {"flags": "H2", "channel": "board"}


def test_years_knob_must_cover_both_halves() -> None:
    """输入：只留「三年以上」的年限池。输出：无。

    预期行为：构造期当场报人话——`H1` 命中的样本只能从「不足三年」那一半取，池子里一个都没有时，
    那些样本在采样期必然报错。这种错在开工前就看得出来，不用拖到真跑（范式 §9.61）。

    采样期那一道（`features.pick_years` 里）是第二条防线，它自己的单元测试在 `test_features` 里。
    """
    with pytest.raises(ValueError, match="不足三年"):
        HireCheckConfig(split="train", years=features.YEAR_HINTS_LONG)


def test_name_knob_reaches_the_prompt() -> None:
    """输入：只留一个姓名的采样参数。输出：无。

    预期行为：`plan_sample` 抽出来的姓名下标真的传到了提示词里（范式 §9.60 / §9.61 的第三个样本级
    变量：它不进 case 空间、不进 `stats_dimensions`，只改 `candidate:` 那一行怎么写）。
    """
    case = HireCheckCase("H1", features.CHANNEL_BOARD)
    workflow = HireCheckWorkflow()
    config = HireCheckConfig(split="train", names=(5,))
    spec = workflow.sample(random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0))
    assert isinstance(spec, HireCheckSpec)
    assert spec.name == 5
    prompt = render.build_prompt(locales.load("zh"), spec.case, spec.role, spec.years, spec.name)
    assert locales.pick(locales.load("zh")["prompt"]["names"], 5) in prompt
    assert workflow.stats_dimensions(case) == {"flags": "H1", "channel": "board"}


def test_trace_judge_needs_a_backend_too() -> None:
    """输入：没注入后端就要 trace 判别器。输出：无。预期行为：报接线错误。"""
    with pytest.raises(RuntimeError, match="bind_backend"):
        HireCheckWorkflow().make_judges(("trace",), "zh")


def test_language_texts_only_reports_the_note() -> None:
    """输入：一份产物样本。输出：无。

    预期行为：只报 `note:` 那一行的备注（键名、姓名、年限、技能、许可与学历都是格式，报上去只会把
    汉字占比稀释掉，把合格的中文产物误报成「语言对不上」）；认不出 state 时报空表。
    """
    workflow = HireCheckWorkflow()
    state = render.to_state(
        render.ContentDraft(packet=packet_of(())), locales.load("zh")
    )
    texts = workflow.language_texts({"state": state})
    assert list(texts) == ["候选人备注"]
    assert texts["候选人备注"].startswith("现在在一家支付公司")
    assert workflow.language_texts({"state": "没有抬头"}) == {}


def test_reparse_delegates_to_the_renderer() -> None:
    """输入：一段 state。输出：无。预期行为：`reparse` 与 `render.parse_state` 是同一件事。"""
    state = render.to_state(render.ContentDraft(packet=packet_of(("H4",))), locales.load("zh"))
    parsed = HireCheckWorkflow().reparse(state)
    assert parsed.facts == features.TAG_IDS
    assert parsed.found


def test_case_helpers_and_locales_are_wired() -> None:
    """输入：无。输出：无。预期行为：`case_label` / `dimension_titles` / `available_locales` 都接好了。"""
    workflow = HireCheckWorkflow()
    case = HireCheckCase("H1+H4", features.CHANNEL_REFERRAL)
    assert workflow.case_label(case) == "H1+H4-referral"
    assert set(workflow.dimension_titles()) == {"flags", "channel", "level"}
    assert workflow.available_locales() == ["en", "zh"]
