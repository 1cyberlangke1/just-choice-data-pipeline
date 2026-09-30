"""依赖与许可证风险门控的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import language, pipeline, workflow as workflow_mod

from tests.workflows.depcheck.conftest import (
    BASE_LINES,
    FakeBackend,
    packet_of,
    render_lines,
)
from workflows.depcheck import features, locales, render, workflow as depcheck_workflow
from workflows.depcheck.features import DepCheckCase
from workflows.depcheck.spec import DepCheckSpec, spec_to_dict
from workflows.depcheck.workflow import DepCheckConfig, DepCheckWorkflow

#: 小批量试跑的条数上限。train 切片一共 19 组，`_cover_pick` 会从里面挑 8 组。
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 8
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")

#: 审计报告里那五个「不自洽」计数器。
BAD_KEYS = ("content_bad", "flag_bad", "grade_bad", "combo_bad", "act_bad")


def make_config(
    tmp_path: Path,
    run_id: str,
    backend: object,
    *,
    split: str = "train",
    limit: int = CASE_LIMIT,
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="depcheck",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=DepCheckConfig(split=split, case_limit=limit),
        backend=backend,
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[DepCheckCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return depcheck_workflow.cases_for(DepCheckConfig(split=split, case_limit=limit))


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，四族题都真的
    出得来（含 `score` 那一族）。
    """
    result = pipeline.run(make_config(tmp_path, "p1", fake_backend()))
    expected = len(cases_of("train"))
    assert result.accepted == expected
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("depcheck").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == expected
    for family in ("flag", "grade", "combo", "act"):
        assert report["train"]["families"][family] > 0, family
    for key in BAD_KEYS:
        assert report["train"][key] == 0, key


@pytest.mark.parametrize(
    ("switch", "code"),
    [
        ("too_short", "E_ASSEMBLE"),
        ("fence", "E_ASSEMBLE"),
        ("missing_key", "E_ASSEMBLE"),
        ("empty_note", "E_ASSEMBLE"),
    ],
)
def test_always_broken_drafts_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：每一组 case 都会踩到的「写歪」。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是该有的那一个——写歪的记录是渲染期的 `E_ASSEMBLE`
    （重试耗尽后仍被拒，不静默放行）。
    """
    result = pipeline.run(make_config(tmp_path, f"p3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


def test_a_lying_trace_only_spares_the_plain_record(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解漏报最后一条命中。输出：RunResult。

    预期行为：命中组合为空的那一条照样合格（没有可漏报的），其余全被 `E_JUDGE_TRACE` 拦下——
    「反解说反话」与「记录写歪」是两条独立的闸，各自拦各自那一类。
    """
    unaffected = sum(1 for case in cases_of("train") if not case.tags())
    result = pipeline.run(make_config(tmp_path, "p3-trace", fake_backend(trace_lies=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_a_misread_level_is_caught_by_the_trace(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解把等级说成另一档（命中都对）。输出：RunResult。

    预期行为：全被 `E_JUDGE_TRACE` 拦下——`grade` 是这一族的中心题型，读者读不出等级，那一道题
    就等于没有答案。
    """
    result = pipeline.run(make_config(tmp_path, "p3-level", fake_backend(trace_level_lies=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_ood_split_is_the_held_out_ecosystem() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：留出侧**只**有 `cargo`、同分布侧**一次都不出现** `cargo`；四个切分两两不重叠，
    合起来正好是全空间——留出维「只影响渲染、不影响标签」这条不变量靠它守着。
    """
    per_split = {
        split: {
            (case.flags, case.ecosystem)
            for case in depcheck_workflow.cases_for(DepCheckConfig(split=split))
        }
        for split in ALL_SPLITS
    }
    assert {case[1] for case in per_split["test_ood"]} == {"cargo"}
    for split in ID_SPLITS:
        assert "cargo" not in {case[1] for case in per_split[split]}, split
    total = set().union(*per_split.values())
    assert sum(len(cases) for cases in per_split.values()) == len(total)
    assert total == {
        (value, eco) for value in features.HIT_SETS for eco in features.ECOSYSTEMS
    }


def test_train_covers_every_level() -> None:
    """输入：train 切分的 case 列表。输出：无。

    预期行为：训练侧五档等级齐全——某一档在训练侧取不到时，`grade` 那一族的那个答案就永远学不到。
    这条把 `SPLIT_SEED` 钉住：换种子时它会当场报出来。
    """
    levels = {
        features.level_of(case.tags())
        for case in depcheck_workflow.cases_for(DepCheckConfig(split="train"))
    }
    assert levels == set(features.LEVELS)


def test_level_one_has_a_single_combination() -> None:
    """输入：全空间 16 组命中组合。输出：无。

    预期行为：最低那一档只有一个组合（四个标记全不命中）——这是上面那条的理由：一档只有三个来源
    （三个生态）的两条 case，三个同分布切分里必然有一个见不到它。
    """
    counts = dict.fromkeys(features.LEVELS, 0)
    for value in features.HIT_SETS:
        counts[features.level_of(features.parse_hits(value))] += 1
    assert [level for level, count in counts.items() if count == 1] == [1]


def test_cover_pick_spreads_the_values() -> None:
    """输入：小批量（8 条）。输出：无。

    预期行为：小批量的挑选**优先补齐没见过的取值**——命中组合、生态与等级各铺开一遍，而不是从
    列表头上截一段（那会让某些档位在小批量里一条都出不来）。
    """
    picked = cases_of("train")
    assert {features.level_of(case.tags()) for case in picked} == set(features.LEVELS)
    assert {case.ecosystem for case in picked} == {"npm", "pypi"}


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = DepCheckCase("D2+D3", features.ECO_PYPI)
    config = DepCheckConfig(split="train", phrasing_pool=(1,))
    workflow = DepCheckWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, DepCheckSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)
    assert spec_to_dict(first)["hits"] == ["D2", "D3"]
    assert spec_to_dict(first)["level"] == features.level_of(("D2", "D3"))


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = DepCheckWorkflow()
    case = DepCheckCase("D3", features.ECO_CARGO)
    state = render.to_state(
        render.ContentDraft(packet=packet_of(case.tags())), locales.load("zh")
    )
    parsed = workflow.reparse(state)
    assert parsed.text == packet_of(case.tags())
    assert parsed.facts == features.TAG_IDS
    assert workflow.case_label(case) == "D3-cargo"
    assert workflow.stats_dimensions(case) == {"flags": "D3", "ecosystem": "cargo"}


def test_language_texts_reports_only_the_note() -> None:
    """输入：一条兑现计划的样本。输出：无。

    预期行为：只报**一段自然语言正文**（`note:` 那一行）——键名、包名、版本号、许可证标识都是格式，
    报上去只会把汉字占比稀释掉（范式 §9.50）。
    """
    packet = packet_of(("D1", "D3"))
    row = {"state": render.to_state(render.ContentDraft(packet=packet), locales.load("zh"))}
    texts = DepCheckWorkflow().language_texts(row)
    assert list(texts) == ["依赖备注"]
    assert texts["依赖备注"] == BASE_LINES["note"].removeprefix("note: ")
    assert language.sample_problem(texts, ("zh",)) is None


def test_language_texts_catches_a_note_in_the_wrong_language() -> None:
    """输入：模型把 `note:` 那一行写成了英文。输出：无。

    预期行为：报上去的那一段能被语言尺子抓住——这正是「只报自然语言正文」换来的东西：报整屏
    记录时，包名与取值会把汉字占比稀释到看不出来。
    """
    packet = render_lines(
        {
            **BASE_LINES,
            "note": "note: a small helper that ships logs to the collector and is still maintained",
        }
    )
    row = {"state": render.to_state(render.ContentDraft(packet=packet), locales.load("zh"))}
    texts = DepCheckWorkflow().language_texts(row)
    assert language.sample_problem(texts, ("zh",)) is not None


def test_language_texts_is_empty_when_there_is_no_note() -> None:
    """输入：一条没有 `note:` 那一行的记录。输出：无。

    预期行为：报空表——那把语言尺子对空表不发表意见，正是「没有自然语言正文」该有的形状。
    """
    packet = render_lines({key: value for key, value in BASE_LINES.items() if key != "note"})
    row = {"state": render.to_state(render.ContentDraft(packet=packet), locales.load("zh"))}
    assert DepCheckWorkflow().language_texts(row) == {}


def test_declared_contract_fields() -> None:
    """输入：workflow 的声明项。输出：无。

    预期行为：题型（**含 score**）、后端需求与判别器名单都对得上。
    """
    workflow = DepCheckWorkflow()
    assert workflow.name == "depcheck"
    assert workflow.kinds == frozenset({"choice", "noul", "score"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("hygiene", "frame", "trace")


def test_config_knobs_are_validated() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话，不拖到采样阶段。"""
    with pytest.raises(ValueError, match="phrasing_pool"):
        DepCheckConfig(split="train", phrasing_pool=())
    with pytest.raises(ValueError, match="categories"):
        DepCheckConfig(split="train", categories=())
    with pytest.raises(ValueError, match="versions"):
        DepCheckConfig(split="train", versions=(3,))
    with pytest.raises(ValueError, match="licenses"):
        DepCheckConfig(split="train", licenses=())
    # 只给一侧的许可证池也要在构造期拦下：另一半的 case 一采样就没得抽（范式 §9.61）。
    with pytest.raises(ValueError, match="licenses"):
        DepCheckConfig(split="train", licenses=("MIT",))
    with pytest.raises(ValueError, match="case_limit"):
        DepCheckConfig(split="train", case_limit=-1)


def test_category_knob_reaches_the_prompt() -> None:
    """输入：只留一个依赖类别的采样参数。输出：无。

    预期行为：`plan_sample` 抽出来的类别真的传到了提示词里（范式 §9.57 / §9.58 的样本级变量：它
    不进 case 空间、不进 `stats_dimensions`，只改包名与 `note:` 那一行的写法）。
    """
    case = DepCheckCase("D1", features.ECO_PYPI)
    workflow = DepCheckWorkflow()
    config = DepCheckConfig(split="train", categories=(5,))
    spec = workflow.sample(random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0))
    assert isinstance(spec, DepCheckSpec)
    assert spec.category == 5
    assert spec_to_dict(spec)["category"] == 5
    prompt = render.build_prompt(
        locales.load("zh"), spec.case, spec.category, spec.version, spec.license
    )
    assert locales.pick(locales.load("zh")["prompt"]["categories"], 5) in prompt
    assert workflow.stats_dimensions(case) == {"flags": "D1", "ecosystem": "pypi"}


def test_version_knob_reaches_the_prompt() -> None:
    """输入：只留一个版本号尾段的采样参数。输出：无。

    预期行为：`plan_sample` 抽出来的尾段真的传到了提示词里（范式 §9.60 的第二个样本级变量：它
    不进 case 空间、不进 `stats_dimensions`，只改 `version:` 那一行怎么写）。
    """
    case = DepCheckCase("D1", features.ECO_NPM)
    workflow = DepCheckWorkflow()
    config = DepCheckConfig(split="train", versions=(47,))
    spec = workflow.sample(random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0))
    assert isinstance(spec, DepCheckSpec)
    assert spec.version == 47
    assert spec_to_dict(spec)["version"] == 47
    prompt = render.build_prompt(
        locales.load("zh"), spec.case, spec.category, spec.version, spec.license
    )
    assert render.version_shape(47) in prompt
    assert workflow.stats_dimensions(case) == {"flags": "D1", "ecosystem": "npm"}


def test_license_knob_reaches_the_prompt() -> None:
    """输入：只留两个许可证标识符的采样参数。输出：无。

    预期行为：`plan_sample` 抽出来的标识符真的传到了提示词里（范式 §9.61 的第三个样本级变量：它
    不进 case 空间、不进 `stats_dimensions`，只改 `license:` 那一行写成哪个标识符）。**池子按判据
    分半**：`D1` 命中的 case 抽到的值必须落在传染性那一半里、不命中的必须落在宽松那一半里——抽错
    了渲染期的 `check_draft` 会把整条样本打回，所以这条测试同时钉住「抽得到」与「抽得对」。
    """
    workflow = DepCheckWorkflow()
    config = DepCheckConfig(split="train", licenses=("GPL-3.0-or-later", "MIT"))
    hit = workflow.sample(
        random.Random(0),
        workflow.plan_sample(random.Random(1), config, DepCheckCase("D1", features.ECO_NPM), 0),
    )
    miss = workflow.sample(
        random.Random(0),
        workflow.plan_sample(random.Random(1), config, DepCheckCase("D2", features.ECO_NPM), 0),
    )
    assert isinstance(hit, DepCheckSpec) and isinstance(miss, DepCheckSpec)
    assert hit.license in features.LICENSE_COPYLEFT
    assert miss.license in features.LICENSE_PERMISSIVE
    assert spec_to_dict(hit)["license"] == hit.license
    prompt = render.build_prompt(
        locales.load("zh"), hit.case, hit.category, hit.version, hit.license
    )
    assert hit.license in prompt
    assert workflow.stats_dimensions(hit.case) == {"flags": "D1", "ecosystem": "npm"}


def test_unknown_split_and_judge_are_rejected() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：三种都当场报人话。"""
    workflow = DepCheckWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    workflow.bind_backend(FakeBackend())
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


def test_render_without_a_backend_reports_wiring_error() -> None:
    """输入：没注入后端就渲染。输出：无。

    预期行为：报错而不是悄悄降级成模板——`needs` 非空时管线一定会注入，收到「没有后端」说明接线
    错了（范式 §9.23 那条「不许静默降级」）。
    """
    workflow = DepCheckWorkflow()
    case = depcheck_workflow.cases_for(DepCheckConfig(split="train", case_limit=1))[0]
    spec = DepCheckSpec(
        case=case,
        phrasing=0,
        category=0,
        version=11,
        license=features.LICENSE_PERMISSIVE[0],
        split="train",
        seq=0,
        plan=features.plan_for(case, 0),
    )
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")
