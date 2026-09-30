"""聊天回复门控的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod

from tests.workflows.chatgate.conftest import (
    FakeBackend,
    candidates_of,
    faithful,
    message_of,
    spec_for,
)
from workflows.chatgate import features, locales, render
from workflows.chatgate.features import ChatCase
from workflows.chatgate.spec import ChatGateSpec, spec_to_dict
from workflows.chatgate.workflow import ChatGateConfig, ChatGateWorkflow

#: 小批量试跑的条数上限。train 切片一共 18 组，`_cover_pick` 会从里面挑 10 组。
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")

#: 审计报告里那五个「不自洽」计数器。
BAD_KEYS = ("content_bad", "tag_bad", "count_bad", "route_bad", "reply_bad")

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
    return pipeline.RunConfig(
        workflow="chatgate",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=ChatGateConfig(split=split, case_limit=limit),
        backend=backend,
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[ChatCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return ChatGateWorkflow().build_cases(ChatGateConfig(split=split, case_limit=limit))


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
    report, missing = workflow_mod.load("chatgate").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == expected
    for family in ("tag", "count", "route", "reply"):
        assert report["train"]["families"][family] > 0, family
    for key in BAD_KEYS:
        assert report["train"][key] == 0, key


@pytest.mark.parametrize(
    ("switch", "code"),
    [
        ("too_short", "E_ASSEMBLE"),
        ("fence", "E_ASSEMBLE"),
        ("clean_carries_flaw", "E_ASSEMBLE"),
    ],
)
def test_always_broken_drafts_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：每一组 case 都会踩到的「写歪」。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是该有的那一个——写歪的消息 / 候选是渲染期的 `E_ASSEMBLE`
    （重试耗尽后仍被拒，不静默放行）。
    """
    result = pipeline.run(make_config(tmp_path, f"p3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


def test_a_lying_trace_only_spares_the_empty_hits(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解漏报最后一条命中。输出：RunResult。

    预期行为：命中组合为空的那一条照样合格（没有可漏报的），其余全被 `E_JUDGE_TRACE` 拦下——
    「反解说反话」与「消息写歪」是两条独立的闸，各自拦各自那一类。
    """
    unaffected = sum(1 for case in cases_of("train") if not case.tags())
    result = pipeline.run(make_config(tmp_path, "p3-trace", fake_backend(trace_lies=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


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


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow。输出：无。预期行为：抛 RuntimeError，不静默降级成模板。"""
    workflow = ChatGateWorkflow()
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec_for(CASE), random.Random(0), "zh")


def test_trace_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却要 trace 判别器。输出：无。预期行为：抛 RuntimeError（接线错了）。"""
    workflow = ChatGateWorkflow()
    with pytest.raises(RuntimeError, match="trace"):
        workflow.make_judges(("trace",), "zh")


def test_unknown_split_and_judge_and_locale_report_available() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：都报人话。"""
    workflow = ChatGateWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


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
    assert spec_to_dict(first)["hits"] == list(case.tags())
    assert spec_to_dict(first)["clean"] == case.slot


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


def test_locales_are_reported() -> None:
    """输入：无。输出：无。预期行为：支持的语言与语言文件目录一致。"""
    workflow = ChatGateWorkflow()
    assert workflow.available_locales() == locales.available() == ["en", "zh"]


def test_declared_contract_fields() -> None:
    """输入：workflow 的声明项。输出：无。预期行为：题型、后端需求与判别器名单都对得上。"""
    workflow = ChatGateWorkflow()
    assert workflow.name == "chatgate"
    assert workflow.kinds == frozenset({"choice", "noul"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("hygiene", "frame", "trace")
    assert workflow.split_names() == ("train", "val", "test_iid", "test_ood")


def test_config_knobs_are_validated() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话，不拖到采样阶段。"""
    with pytest.raises(ValueError, match="phrasing_pool"):
        ChatGateConfig(split="train", phrasing_pool=())
    with pytest.raises(ValueError, match="case_limit"):
        ChatGateConfig(split="train", case_limit=-1)


def test_state_round_trips_through_the_parser() -> None:
    """输入：一份渲染好的 state。输出：无。预期行为：解析回来的正文与渲染前的草稿逐字相同。"""
    spec, sample = faithful(CASE)
    parsed = render.parse_state(str(sample["state"]))
    assert parsed.message == message_of(spec.plan.hits)
    assert parsed.replies == tuple(candidates_of(spec.plan.reply.flaws))
    assert render.detect_locale(str(sample["state"])) == "zh"
