"""聊天回复门控的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import pipeline

from tests.workflows._scaffold import run_config
from tests.workflows.chatgate.conftest import FakeBackend, candidates_of, faithful, message_of
from workflows.chatgate import features, render
from workflows.chatgate.features import ChatCase
from workflows.chatgate.spec import ChatGateSpec
from workflows.chatgate.workflow import ChatGateConfig, ChatGateWorkflow

#: 小批量试跑的条数上限。train 切片一共 18 组，`_cover_pick` 会从里面挑 10 组。
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")


#: 一个普通 case：命中两条，同事私聊，该发第一条。
CASE = ChatCase(degree=2, scene=features.SCENE_COLLEAGUE, slot=0)


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
        workflow="chatgate",
        split=split,
        seed=7,
        override=ChatGateConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[ChatCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return ChatGateWorkflow().build_cases(ChatGateConfig(split=split, case_limit=limit))


def test_a_wrong_clean_pick_is_caught_on_every_case(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解把「哪条候选干净」说成另一条。输出：RunResult。

    预期行为：**一条都留不下**——「该发哪条候选」与消息踩了几条无关，所以连一条都不命中的 case
    也一样被拦下（这正是 `reply` 那一族与消息解耦的体现）。
    """
    result = pipeline.run(
        make_config(tmp_path, "p3-clean-lie", fake_backend(trace_clean_lies=True))
    )
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_dropping_a_fragment_only_spares_the_cases_without_one(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：少写一条该命中的片段。输出：RunResult。

    预期行为：只有「本来就一条都不命中」的那些 case 能活下来（没有片段可少写）。
    """
    unaffected = sum(1 for case in cases_of("train") if not case.tags())
    result = pipeline.run(make_config(tmp_path, "p3-drop", fake_backend(drop_fragment=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_adding_a_spare_fragment_only_spares_the_full_hits(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：多写一条不该命中的片段。输出：RunResult。

    预期行为：只有「四条全命中」的 case 无处可加，其余全被 `E_ASSEMBLE` 拦下。
    """
    unaffected = sum(1 for case in cases_of("train") if len(case.tags()) == len(features.TAG_IDS))
    result = pipeline.run(make_config(tmp_path, "p3-extra", fake_backend(extra_fragment=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_ood_split_only_uses_the_held_out_scene() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：`test_ood` 全是 `stranger`，同分布三个切分一个 `stranger` 都没有——这就是值留出。
    """
    assert {case.scene for case in cases_of("test_ood", limit=0)} == {"stranger"}
    for split in ID_SPLITS:
        scenes = {case.scene for case in cases_of(split, limit=0)}
        assert "stranger" not in scenes, split


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：四个切分两两不重、并起来正好是全空间 45 组（同分布侧剔掉留出取值之后按桶分）。
    """
    seen: list[tuple[int, str, int]] = []
    for split in ALL_SPLITS:
        seen.extend((case.degree, case.scene, case.slot) for case in cases_of(split, limit=0))
    assert len(seen) == len(set(seen)) == 45
    assert set(seen) == {
        (case.degree, case.scene, case.slot) for case in features.build_cases(512)
    }


def test_every_count_and_handling_level_is_reachable() -> None:
    """输入：同分布三个切分。输出：无。

    预期行为：命中条数 0~4 与四个处置档位在同分布侧全都取得到——不然那几族在训练侧会缺档，
    均衡闸与审计都会当场报出来。
    """
    counts: set[int] = set()
    handlings: set[str] = set()
    for split in ID_SPLITS:
        for case in cases_of(split, limit=0):
            labels = features.label_for(features.plan_for(case, 0))
            counts.add(labels.count)
            handlings.add(labels.handling)
    assert counts == set(range(len(features.TAG_IDS) + 1))
    assert handlings == set(features.HANDLING_LEVELS)


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = ChatCase(degree=3, scene=features.SCENE_GROUP, slot=2)
    config = ChatGateConfig(split="train", phrasing_pool=(1,))
    workflow = ChatGateWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, ChatGateSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = ChatGateWorkflow()
    case = ChatCase(degree=2, scene=features.SCENE_GROUP, slot=1)
    spec, sample = faithful(case)
    parsed = workflow.reparse(str(sample["state"]))
    assert parsed.tags == features.TAG_IDS
    assert parsed.flaws == features.FLAW_IDS
    assert parsed.message == message_of(spec.plan.hits)
    assert parsed.replies == tuple(candidates_of(spec.plan.reply.flaws))
    assert workflow.case_label(case) == "degree2-group-slot2"
    assert workflow.stats_dimensions(case) == {"degree": 2, "scene": "group", "slot": 1}
    assert workflow.dimension_titles()["degree"]


def test_language_texts_report_only_the_model_written_parts() -> None:
    """输入：一条忠实样本。输出：无。

    预期行为：只报模型写的那两段正文——三张表是代码按语言渲染的，报进去会把「文案与声明语言
    一致」这条尺子变成空转。
    """
    spec, sample = faithful(CASE)
    texts = ChatGateWorkflow().language_texts(sample)
    assert set(texts) == {"对方消息", "候选回复"}
    assert texts["对方消息"] == message_of(spec.plan.hits)
    assert texts["候选回复"].splitlines() == candidates_of(spec.plan.reply.flaws)


def test_state_round_trips_through_the_parser() -> None:
    """输入：一份渲染好的 state。输出：无。预期行为：解析回来的正文与渲染前的草稿逐字相同。"""
    spec, sample = faithful(CASE)
    parsed = render.parse_state(str(sample["state"]))
    assert parsed.message == message_of(spec.plan.hits)
    assert parsed.replies == tuple(candidates_of(spec.plan.reply.flaws))
    assert render.detect_locale(str(sample["state"])) == "zh"
