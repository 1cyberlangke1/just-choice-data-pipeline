"""字幕对质检的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import language, pipeline

from tests.workflows._scaffold import run_config
from tests.workflows.subqa.conftest import BASE_LINES, cue_of, render_lines
from workflows.subqa import features, locales, render, workflow as subqa_workflow
from workflows.subqa.features import SubQaCase
from workflows.subqa.spec import SubQaSpec
from workflows.subqa.workflow import SubQaConfig, SubQaWorkflow

#: 小批量试跑的条数上限。train 切片一共 19 组里的前十来个，`_cover_pick` 会从里面挑 10 组。
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
        workflow="subqa",
        split=split,
        seed=7,
        override=SubQaConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[SubQaCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return subqa_workflow.cases_for(SubQaConfig(split=split, case_limit=limit))


def test_ood_split_only_uses_the_held_out_track() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：`test_ood` 全是 `lecture`，同分布三个切分一个 `lecture` 都没有——这就是值留出。
    """
    assert {case.track for case in cases_of("test_ood", limit=0)} == set(
        subqa_workflow.OOD_TRACKS
    )
    for split in ID_SPLITS:
        tracks = {case.track for case in cases_of(split, limit=0)}
        assert not (tracks & set(subqa_workflow.OOD_TRACKS)), split


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：四个切分两两不重、并起来正好是全空间 48 组（同分布侧剔掉留出取值之后按桶分）。
    """
    seen: list[tuple[str, str]] = []
    for split in ALL_SPLITS:
        seen.extend((case.hits, case.track) for case in cases_of(split, limit=0))
    assert len(seen) == len(set(seen)) == 48
    assert set(seen) == {(case.hits, case.track) for case in features.build_cases(512)}


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = SubQaCase("S2+S4", features.TRACK_SERIES)
    config = SubQaConfig(split="train", phrasing_pool=(1,))
    workflow = SubQaWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, SubQaSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = SubQaWorkflow()
    case = SubQaCase("S3", features.TRACK_SERIES)
    state = render.to_state(render.ContentDraft(cue=cue_of(case.tags())), locales.load("zh"))
    parsed = workflow.reparse(state)
    assert parsed.text == cue_of(case.tags())
    assert parsed.tags == features.TAG_IDS
    assert workflow.case_label(case) == "S3-series"
    assert workflow.stats_dimensions(case) == {"hits": "S3", "track": "series"}
    assert workflow.dimension_titles()["hits"]


def test_language_texts_reports_only_the_two_natural_language_fields() -> None:
    """输入：一条译文与备注都是中文的字幕对（键名、原文与时间码全是 ASCII）。输出：无。

    预期行为：`language_texts` 只报**两段自然语言正文**——译文与译者备注。键名、时间码与 `-->`
    是格式，`Source:` 那一行的原文按设计就是英文（源语言固定），把它们一起报上去只会把汉字占比
    稀释掉（范式 §9.50）。
    """
    cue = cue_of(("S1", "S3"))
    row = {"state": render.to_state(render.ContentDraft(cue=cue), locales.load("zh"))}
    texts = SubQaWorkflow().language_texts(row)
    assert list(texts) == ["译文", "译者备注"]
    assert texts["译文"] == BASE_LINES["target"].removeprefix("Target: ")
    assert texts["译者备注"] == BASE_LINES["note"].removeprefix("Note: ")
    assert language.sample_problem(texts, ("zh",)) is None


def test_language_texts_catches_a_translation_written_in_the_wrong_language() -> None:
    """输入：模型把 `Target:` 那一行的译文写成了英文。输出：无。

    预期行为：报上去的那一段能被语言尺子抓住——这正是「只报自然语言正文」换来的东西：报整条
    字幕对时，英文原文那一行会把译文稀释到看不出来。
    """
    cue = render_lines(
        {**BASE_LINES, "target": "Target: We should move the meeting to Thursday, if that works."}
    )
    row = {"state": render.to_state(render.ContentDraft(cue=cue), locales.load("zh"))}
    texts = SubQaWorkflow().language_texts(row)
    assert language.sample_problem(texts, ("zh",)) is not None


def test_language_texts_is_empty_when_there_is_no_natural_language() -> None:
    """输入：一条既没有 `Target:` 也没有 `Note:` 那一行的字幕对。输出：无。

    预期行为：报空表——那把语言尺子对空表不发表意见，正是「没有自然语言正文」该有的形状。
    """
    cue = render_lines(
        {key: value for key, value in BASE_LINES.items() if key not in ("target", "note")}
    )
    row = {"state": render.to_state(render.ContentDraft(cue=cue), locales.load("zh"))}
    assert SubQaWorkflow().language_texts(row) == {}


def test_timecode_knob_reaches_the_prompt() -> None:
    """输入：只留一个开始时间码的采样参数。输出：无。

    预期行为：`plan_sample` 抽出来的时间码真的传到了提示词里（范式 §9.61 的样本级变量：它不进
    case 空间、不进 `stats_dimensions`，只改 `Timing:` 那一行的开始那一侧）。
    """
    case = SubQaCase("S3", features.TRACK_MOVIE)
    workflow = SubQaWorkflow()
    config = SubQaConfig(split="train", timecodes=("00:07:19,481",))
    spec = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    assert isinstance(spec, SubQaSpec)
    assert spec.timecode == "00:07:19,481"
    prompt = render.build_prompt(locales.load("zh"), spec.case, spec.timecode)
    assert spec.timecode in prompt
    assert workflow.stats_dimensions(case) == {"hits": "S3", "track": "movie"}
