"""安全警报风险定级的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

from jc import language, pipeline
from jc.backend.types import ProviderUnavailable

from tests.workflows._scaffold import run_config
from tests.workflows.alertrisk.conftest import (
    BASE_LINES,
    FakeBackend,
    packet_of,
    render_lines,
)
from workflows.alertrisk import features, locales, render, workflow as alertrisk_workflow
from workflows.alertrisk.features import AlertRiskCase
from workflows.alertrisk.spec import AlertRiskSpec
from workflows.alertrisk.workflow import AlertRiskConfig, AlertRiskWorkflow

#: 小批量试跑的条数上限。train 切片一共 9 组，`_cover_pick` 会从里面挑 8 组。
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
        workflow="alertrisk",
        split=split,
        seed=7,
        override=AlertRiskConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[AlertRiskCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return alertrisk_workflow.cases_for(AlertRiskConfig(split=split, case_limit=limit))


def test_a_misread_matrix_is_caught_by_the_trace(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解把等级说成另一档（因子都对）。输出：RunResult。

    预期行为：全被 `E_JUDGE_TRACE` 拦下——`grade` 是这一族的中心题型，读者读不出等级，那一道题
    就等于没有答案。
    """
    result = pipeline.run(make_config(tmp_path, "p3-level", fake_backend(trace_level_lies=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_ood_split_is_the_held_out_channel() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：留出侧**只**有 `cloud`、同分布侧**一次都不出现** `cloud`；四个切分两两不重叠，
    合起来正好是全空间——留出维「只影响渲染、不影响标签」这条不变量靠它守着。
    """
    per_split = {
        split: {
            (case.highs, case.source)
            for case in alertrisk_workflow.cases_for(AlertRiskConfig(split=split))
        }
        for split in ALL_SPLITS
    }
    assert {case[1] for case in per_split["test_ood"]} == {"cloud"}
    for split in ID_SPLITS:
        assert "cloud" not in {case[1] for case in per_split[split]}, split
    total = set().union(*per_split.values())
    assert sum(len(cases) for cases in per_split.values()) == len(total)
    assert total == {
        (value, source) for value in features.HIT_SETS for source in features.SOURCES
    }


def test_train_covers_every_level() -> None:
    """输入：train 切分的 case 列表。输出：无。

    预期行为：训练侧五档等级齐全——某一档在训练侧取不到时，`grade` 那一族的那个答案就永远学不到。
    这条把 `SPLIT_SEED` 钉住：换种子时它会当场报出来。

    **为什么只要求训练侧**：最低那一档只有一个命中组合（三个因子全为低），而它只有两个来源，
    所以最多两条 case；三个同分布切分里必然有一个见不到它。这是**结构上做不到**，不是漏了；
    留出侧的要求只有「值留出」那一条（见 `test_ood_split_is_the_held_out_channel`）。
    """
    levels = {
        features.level_of(case.tags())
        for case in alertrisk_workflow.cases_for(AlertRiskConfig(split="train"))
    }
    assert levels == set(features.LEVELS)


def test_level_one_has_a_single_combination() -> None:
    """输入：全空间 8 组命中组合。输出：无。

    预期行为：五档里有两档各只对应一个组合——最低那一档（三个因子全为低）与最高那一档（三个因子
    全为高）。这是上面那条「只要求训练侧齐全」的理由：一档只有两个来源的两条 case，三个同分布
    切分里必然有一个见不到它。哪天矩阵改成每档都至少两个组合，这条会红，那时可以顺手把要求提到
    三个切分。
    """
    counts = dict.fromkeys(features.LEVELS, 0)
    for value in features.HIT_SETS:
        counts[features.level_of(features.parse_hits(value))] += 1
    assert [level for level, count in counts.items() if count == 1] == [1, 5]


def test_cover_pick_spreads_the_values() -> None:
    """输入：小批量（8 条）。输出：无。

    预期行为：小批量的挑选**优先补齐没见过的取值**——命中组合、报警通道与等级各铺开一遍，
    而不是从列表头上截一段（那会让某些档位在小批量里一条都出不来）。
    """
    picked = cases_of("train")
    assert {features.level_of(case.tags()) for case in picked} == set(features.LEVELS)
    assert {case.source for case in picked} == {"edr", "waf"}


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = AlertRiskCase("R2+R3", features.SOURCE_WAF)
    config = AlertRiskConfig(split="train", phrasing_pool=(1,))
    workflow = AlertRiskWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, AlertRiskSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = AlertRiskWorkflow()
    case = AlertRiskCase("R3", features.SOURCE_WAF)
    state = render.to_state(
        render.ContentDraft(packet=packet_of(case.tags())), locales.load("zh")
    )
    parsed = workflow.reparse(state)
    assert parsed.text == packet_of(case.tags())
    assert parsed.facts == features.TAG_IDS
    assert workflow.case_label(case) == "R3-waf"
    assert workflow.stats_dimensions(case) == {"highs": "R3", "source": "waf"}
    assert workflow.dimension_titles()["highs"]


def test_language_texts_reports_only_the_prose_field() -> None:
    """输入：一条描述是中文的记录（键名、取值与规则名全是 ASCII）。输出：无。

    预期行为：`language_texts` 只报**一段自然语言正文**——告警描述。键名、规则名、主机数、可达性
    与凭据要求都是格式，把它们一起报上去只会把汉字占比稀释掉（范式 §9.50）。
    """
    packet = packet_of(("R1", "R3"))
    row = {"state": render.to_state(render.ContentDraft(packet=packet), locales.load("zh"))}
    texts = AlertRiskWorkflow().language_texts(row)
    assert list(texts) == ["告警描述"]
    assert texts["告警描述"] == BASE_LINES["detail"].removeprefix("detail: ")
    assert language.sample_problem(texts, ("zh",)) is None


def test_language_texts_catches_a_description_in_the_wrong_language() -> None:
    """输入：模型把 `detail:` 那一行写成了英文。输出：无。

    预期行为：报上去的那一段能被语言尺子抓住——这正是「只报自然语言正文」换来的东西：报整屏
    记录时，规则名与取值会把汉字占比稀释到看不出来。
    """
    packet = render_lines(
        {
            **BASE_LINES,
            "detail": "detail: a suspicious powershell was spawned by the build script on a laptop",
        }
    )
    row = {"state": render.to_state(render.ContentDraft(packet=packet), locales.load("zh"))}
    texts = AlertRiskWorkflow().language_texts(row)
    assert language.sample_problem(texts, ("zh",)) is not None


def test_language_texts_is_empty_when_there_is_no_description() -> None:
    """输入：一条没有 `detail:` 那一行的记录。输出：无。

    预期行为：报空表——那把语言尺子对空表不发表意见，正是「没有自然语言正文」该有的形状。
    """
    packet = render_lines({key: value for key, value in BASE_LINES.items() if key != "detail"})
    row = {"state": render.to_state(render.ContentDraft(packet=packet), locales.load("zh"))}
    assert AlertRiskWorkflow().language_texts(row) == {}


def test_direction_knob_reaches_the_prompt() -> None:
    """输入：只留一个手法方向的采样参数。输出：无。

    预期行为：`plan_sample` 抽出来的方向真的传到了提示词里（范式 §9.7 的样本级变量：它不进 case
    空间、不进 `stats_dimensions`，只改 `detail:` 那一行的写法）。
    """
    case = AlertRiskCase("R1", features.SOURCE_WAF)
    workflow = AlertRiskWorkflow()
    config = AlertRiskConfig(split="train", directions=(5,))
    spec = workflow.sample(random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0))
    assert isinstance(spec, AlertRiskSpec)
    assert spec.direction == 5
    prompt = render.build_prompt(locales.load("zh"), spec.case, spec.direction)
    assert locales.pick(locales.load("zh")["prompt"]["directions"], 5) in prompt
    assert workflow.stats_dimensions(case) == {"highs": "R1", "source": "waf"}

