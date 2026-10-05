"""广告元素判定的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import language, pipeline

from tests.workflows._scaffold import run_config
from tests.workflows.adblock.conftest import FakeBackend, text_of
from workflows.adblock import features, locales, render, workflow as adblock_workflow
from workflows.adblock.features import AdBlockCase
from workflows.adblock.spec import AdBlockSpec
from workflows.adblock.workflow import AdBlockConfig, AdBlockWorkflow

#: 小批量试跑的条数上限。train 切片一共 32 组里的六成，`_cover_pick` 会从里面挑 10 组。
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
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
        workflow="adblock",
        split=split,
        seed=7,
        override=AdBlockConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[AdBlockCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return adblock_workflow.cases_for(AdBlockConfig(split=split, case_limit=limit))


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


def test_language_texts_reports_only_the_visible_copy() -> None:
    """输入：一个整段汉字占比很低的 zh 片段（标签名与属性名全是 ASCII）。输出：无。

    预期行为：`language_texts` 只报**元素里的可见文字**，不报整段 HTML——标签名与属性名不是
    自然语言正文，交给语言判据本身就是错的（`jc.language` 模块文档那条「只喂自然语言正文」
    的规矩，范式 §9.49）。
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
    # 判据 2026-10-02 起 zh 侧按**词**比（见 jc.language 的阈值表），所以「整段拿去比会被拦」
    # 已经不是不能报整段的理由了。它真正拦下的是**一条中文都没有**的正文——真机里就是这一条：
    assert language.sample_problem({"元素可见文字": "Sponsored by ABC"}, ("zh",)) is not None


def test_language_texts_is_empty_when_the_element_has_no_visible_copy() -> None:
    """输入：一个只有标签、一个字都没有的片段。输出：无。

    预期行为：报空表——那把语言尺子对空表不发表意见，正是「没有自然语言正文」该有的形状。
    """
    snippet = '<div class="feed-card-item" style="display:none; width:1px;"><img src="/a.png" /></div>'
    row = {"state": render.to_state(render.ContentDraft(snippet=snippet), locales.load("zh"))}
    assert AdBlockWorkflow().language_texts(row) == {}
