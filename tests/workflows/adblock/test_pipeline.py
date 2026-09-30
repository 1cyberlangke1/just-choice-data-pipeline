"""广告元素判定的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import language, pipeline, workflow as workflow_mod

from tests.workflows.adblock.conftest import FakeBackend, spec_for, text_of
from workflows.adblock import features, locales, render, workflow as adblock_workflow
from workflows.adblock.features import AdBlockCase
from workflows.adblock.spec import AdBlockSpec, spec_to_dict
from workflows.adblock.workflow import AdBlockConfig, AdBlockWorkflow

#: 小批量试跑的条数上限。train 切片一共 32 组里的六成，`_cover_pick` 会从里面挑 10 组。
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")

#: 审计报告里那五个「不自洽」计数器。
BAD_KEYS = ("content_bad", "flag_bad", "count_bad", "combo_bad", "act_bad")


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
        workflow="adblock",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=AdBlockConfig(split=split, case_limit=limit),
        backend=backend,
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[AdBlockCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return adblock_workflow.cases_for(AdBlockConfig(split=split, case_limit=limit))


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，四族题都真的
    出得来。
    """
    result = pipeline.run(make_config(tmp_path, "p1", fake_backend()))
    expected = len(cases_of("train"))
    assert result.accepted == expected
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("adblock").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == expected
    for family in ("flag", "count", "combo", "act"):
        assert report["train"]["families"][family] > 0, family
    for key in BAD_KEYS:
        assert report["train"][key] == 0, key


def test_two_runs_are_byte_identical(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：同种子跑两次。输出：两份产物。预期行为：逐字节一致。"""
    first = pipeline.run(make_config(tmp_path, "p2a", fake_backend()))
    second = pipeline.run(make_config(tmp_path, "p2b", fake_backend()))
    assert (first.run_dir / "train.jsonl").read_bytes() == (
        second.run_dir / "train.jsonl"
    ).read_bytes()


@pytest.mark.parametrize(
    ("switch", "code"),
    [
        ("too_short", "E_ASSEMBLE"),
        ("fence", "E_ASSEMBLE"),
        ("no_html", "E_ASSEMBLE"),
    ],
)
def test_always_broken_drafts_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：每一组 case 都会踩到的「写歪」。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是该有的那一个——写歪的片段是渲染期的 `E_ASSEMBLE`
    （重试耗尽后仍被拒，不静默放行）。
    """
    result = pipeline.run(make_config(tmp_path, f"p3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


def test_a_lying_trace_only_spares_the_empty_hits(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解漏报最后一条。输出：RunResult。

    预期行为：命中组合为空的那一条照样合格（没有可漏报的），其余全被 `E_JUDGE_TRACE` 拦下——
    「反解说反话」与「片段写歪」是两条独立的闸，各自拦各自那一类。
    """
    unaffected = sum(1 for case in cases_of("train") if not case.tags())
    result = pipeline.run(make_config(tmp_path, "p3-trace", fake_backend(trace_lies=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_dropping_a_fragment_only_spares_the_cases_without_one(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：少写一条该命中的形状。输出：RunResult。

    预期行为：只有「本来就一条都不命中」的那些 case 能活下来（没有形状可少写）。
    """
    unaffected = sum(1 for case in cases_of("train") if not case.tags())
    result = pipeline.run(make_config(tmp_path, "p3-drop", fake_backend(drop_fragment=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_adding_a_spare_fragment_only_spares_the_full_hits(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：多写一条不该命中的形状。输出：RunResult。

    预期行为：只有「四条全命中」的 case 无处可加，其余全被 `E_ASSEMBLE` 拦下。
    """
    unaffected = sum(1 for case in cases_of("train") if len(case.tags()) == len(features.TAG_IDS))
    result = pipeline.run(make_config(tmp_path, "p3-extra", fake_backend(extra_fragment=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_ood_split_only_uses_the_held_out_surface() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：`test_ood` 全是 `video`，同分布三个切分一个 `video` 都没有——这就是值留出。
    """
    assert {case.surface for case in cases_of("test_ood", limit=0)} == set(
        adblock_workflow.OOD_SURFACES
    )
    for split in ID_SPLITS:
        surfaces = {case.surface for case in cases_of(split, limit=0)}
        assert not (surfaces & set(adblock_workflow.OOD_SURFACES)), split


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：四个切分两两不重、并起来正好是全空间 48 组（同分布侧剔掉留出取值之后按桶分）。
    """
    seen: list[tuple[str, str]] = []
    for split in ALL_SPLITS:
        seen.extend((case.hits, case.surface) for case in cases_of(split, limit=0))
    assert len(seen) == len(set(seen)) == 48
    assert set(seen) == {(case.hits, case.surface) for case in features.build_cases(512)}


def test_every_count_and_act_level_is_reachable() -> None:
    """输入：同分布三个切分。输出：无。

    预期行为：命中条数 0~4 与四个处置档位在同分布侧全都取得到——不然那几族在训练侧会缺档，
    均衡闸与审计都会当场报出来。
    """
    counts: set[int] = set()
    acts: set[str] = set()
    for split in ID_SPLITS:
        for case in cases_of(split, limit=0):
            labels = features.label_for(features.plan_for(case, 0))
            counts.add(labels.count)
            acts.add(labels.act)
    assert counts == set(range(len(features.TAG_IDS) + 1))
    assert acts == set(features.ACTS)


def test_full_train_run_is_balanced(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：整个 train 切分跑一遍。输出：RunResult。预期行为：零拒绝、各族的答案均衡全绿。"""
    result = pipeline.run(make_config(tmp_path, "p4-full", fake_backend(), limit=0))
    assert result.rejected == []
    assert result.accepted == len(cases_of("train", limit=0))
    assert [item["ok"] for item in result.stats["balance"]] == [True] * len(
        result.stats["balance"]
    )


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow。输出：无。预期行为：抛 RuntimeError，不静默降级成模板。"""
    workflow = AdBlockWorkflow()
    spec = spec_for(AdBlockCase("A1", features.SURFACE_ARTICLE))
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_trace_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却要 trace 判别器。输出：无。预期行为：抛 RuntimeError（接线错了）。"""
    workflow = AdBlockWorkflow()
    with pytest.raises(RuntimeError, match="trace"):
        workflow.make_judges(("trace",), "zh")


def test_unknown_split_and_judge_and_locale_report_available() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：都报人话。"""
    workflow = AdBlockWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = AdBlockCase("A2+A4", features.SURFACE_FEED)
    config = AdBlockConfig(split="train", phrasing_pool=(1,))
    workflow = AdBlockWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, AdBlockSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)
    assert spec_to_dict(first)["hits"] == ["A2", "A4"]


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = AdBlockWorkflow()
    case = AdBlockCase("A3", features.SURFACE_FEED)
    state = render.to_state(
        render.ContentDraft(snippet=text_of(case.tags())), locales.load("zh")
    )
    parsed = workflow.reparse(state)
    assert parsed.text == text_of(case.tags())
    assert parsed.tags == features.TAG_IDS
    assert workflow.case_label(case) == "A3-feed"
    assert workflow.stats_dimensions(case) == {"hits": "A3", "surface": "feed"}
    assert workflow.dimension_titles()["hits"]


def test_locales_are_reported() -> None:
    """输入：无。输出：无。预期行为：支持的语言与语言文件目录一致。"""
    workflow = AdBlockWorkflow()
    assert workflow.available_locales() == locales.available() == ["en", "zh"]


def test_language_texts_reports_only_the_visible_copy() -> None:
    """输入：一个整段汉字占比很低的 zh 片段（标签名与属性名全是 ASCII）。输出：无。

    预期行为：`language_texts` 只报**元素里的可见文字**，不报整段 HTML——整段 HTML 的汉字占比会被
    标签名 / 属性名稀释到 `jc.language.ZH_FLOOR` 以下（真机 `ab-10b-zh` 10 条里 5 条踩中），
    拿它去比会把合格的中文产物报成「语言对不上」（范式 §9.49）。
    """
    snippet = (
        '<div class="feed-card sponsored-adslot" style="display:none;width:1px;">'
        '<a href="/post/10086">热门推荐内容</a></div>'
    )
    row = {"state": render.to_state(render.ContentDraft(snippet=snippet), locales.load("zh"))}
    texts = AdBlockWorkflow().language_texts(row)
    assert list(texts) == ["元素可见文字"]
    assert texts["元素可见文字"] == "热门推荐内容"
    assert language.sample_problem(texts, ("zh",)) is None
    # 整段拿去比就会被误报——这正是不能报整段的理由。
    assert language.sample_problem({"HTML 片段": snippet}, ("zh",)) is not None


def test_language_texts_is_empty_when_the_element_has_no_visible_copy() -> None:
    """输入：一个只有标签、一个字都没有的片段。输出：无。

    预期行为：报空表——那把语言尺子对空表不发表意见，正是「没有自然语言正文」该有的形状。
    """
    snippet = '<div class="feed-card-item" style="display:none; width:1px;"><img src="/a.png" /></div>'
    row = {"state": render.to_state(render.ContentDraft(snippet=snippet), locales.load("zh"))}
    assert AdBlockWorkflow().language_texts(row) == {}


def test_declared_contract_fields() -> None:
    """输入：workflow 的声明项。输出：无。预期行为：题型、后端需求与判别器名单都对得上。"""
    workflow = AdBlockWorkflow()
    assert workflow.name == "adblock"
    assert workflow.kinds == frozenset({"choice", "noul"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("hygiene", "frame", "trace")


def test_config_knobs_are_validated() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话，不拖到采样阶段。"""
    with pytest.raises(ValueError, match="phrasing_pool"):
        AdBlockConfig(split="train", phrasing_pool=())
    with pytest.raises(ValueError, match="case_limit"):
        AdBlockConfig(split="train", case_limit=-1)
