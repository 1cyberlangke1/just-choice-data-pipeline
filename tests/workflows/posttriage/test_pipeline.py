"""社交帖子分诊的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod
from jc.jsontypes import JsonValue

from tests.workflows.posttriage.conftest import FakeBackend, spec_for, text_of
from workflows.posttriage import features, locales, render, workflow as posttriage_workflow
from workflows.posttriage.features import PostTriageCase
from workflows.posttriage.spec import PostTriageSpec, spec_to_dict
from workflows.posttriage.workflow import PostTriageConfig, PostTriageWorkflow

#: 小批量试跑的条数上限。train 切片一共 9 组（16 组的 60%），所以实际跑满的是 9 条——
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")

#: 审计报告里那五个「不自洽」计数器。
BAD_KEYS = ("content_bad", "flag_bad", "count_bad", "combo_bad", "route_bad")


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
        workflow="posttriage",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=PostTriageConfig(split=split, case_limit=limit),
        backend=backend,
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[PostTriageCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return posttriage_workflow.cases_for(PostTriageConfig(split=split, case_limit=limit))


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
    report, missing = workflow_mod.load("posttriage").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == expected
    for family in ("flag", "count", "combo", "route"):
        assert report["train"]["families"][family] > 0, family
    for key in BAD_KEYS:
        assert report["train"][key] == 0, key


@pytest.mark.parametrize(
    ("switch", "code"),
    [
        ("too_short", "E_ASSEMBLE"),
        ("fence", "E_ASSEMBLE"),
        ("decoy_extra", "E_ASSEMBLE"),
    ],
)
def test_always_broken_drafts_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：每一组 case 都会踩到的「写歪」。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是 `E_ASSEMBLE`——写歪的正文在渲染期被拦下（重试耗尽后
    仍被拒，不静默放行）。
    """
    result = pipeline.run(make_config(tmp_path, f"p3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


def test_a_lying_reader_only_spares_the_cases_with_nothing_to_lie_about(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解时把命中组合说反（漏报最后一条）。输出：RunResult。

    预期行为：一条命中都没有的 case 没得可漏，照样合格；其余全被 `E_JUDGE_TRACE` 拦下。

    为什么单拎出来：`trace_lies` 的「说反」是**漏报最后一条**，一条都没命中的 case 上它是空操作
    ——把它塞进「整批都会被拦下」那条断言里，就会依赖「train 切片里恰好没有空命中的 case」这种
    跟切分桶有关的巧合（第一版就是这么写的，实测 train 里正好有一个）。
    """
    unaffected = sum(1 for case in cases_of("train") if not case.hits())
    result = pipeline.run(make_config(tmp_path, "p3-lie", fake_backend(trace_lies=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_dropping_a_fragment_only_spares_the_cases_without_one(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：少写一条该命中的片段。输出：RunResult。

    预期行为：本来就没有命中片段的 case 照样合格，其余全被 `E_ASSEMBLE` 拦下——一个都不放过。
    """
    unaffected = sum(1 for case in cases_of("train") if not case.hits())
    result = pipeline.run(make_config(tmp_path, "p3-drop", fake_backend(drop_fragment=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_adding_a_spare_fragment_only_spares_the_full_hits(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：多写一条不该命中的片段。输出：RunResult。

    预期行为：四条全中的 case 没有「多余的一条」可加，其余全被拦下。
    """
    unaffected = sum(
        1 for case in cases_of("train") if len(case.hits()) == len(features.SIGNAL_IDS)
    )
    result = pipeline.run(make_config(tmp_path, "p3-extra", fake_backend(extra_fragment=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_missing_decoy_on_the_ood_split_is_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：留出侧（`decoy=lower`）却不写诱饵。输出：RunResult。预期行为：整批都被拦下。"""
    result = pipeline.run(
        make_config(tmp_path, "p3-decoy", fake_backend(decoy_missing=True), split="test_ood")
    )
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：四份互不相交，合起来正好是全空间 32 组（同分布侧 16 组全是 `decoy=none`，
    留出侧 16 组全是 `decoy=lower`）。
    """
    keys: dict[tuple[tuple[str, JsonValue], ...], str] = {}
    for split in ALL_SPLITS:
        cases = cases_of(split, limit=0)
        assert cases, split
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in keys, f"{case.describe()} 同时落在 {keys[key]} 与 {split}"
            keys[key] = split
    assert len(keys) == 32
    assert {dict(key)["decoy"] for key, split in keys.items() if split == "test_ood"} == {
        features.DECOY_LOWER
    }
    assert {dict(key)["decoy"] for key, split in keys.items() if split != "test_ood"} == {
        features.DECOY_NONE
    }


def test_every_count_and_route_level_is_reachable() -> None:
    """输入：同分布三个切分。输出：无。

    预期行为：命中条数 0~4 与四个处置档位在同分布侧全都取得到——不然那几族在训练侧会缺档，
    均衡闸与审计都会当场报出来。
    """
    counts: set[int] = set()
    routes: set[str] = set()
    for split in ID_SPLITS:
        for case in cases_of(split, limit=0):
            labels = features.label_for(features.plan_for(case, 0))
            counts.add(labels.count)
            routes.add(labels.route)
    assert counts == set(range(len(features.SIGNAL_IDS) + 1))
    assert routes == set(features.ROUTE_LEVELS)


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow。输出：无。预期行为：抛 RuntimeError，不静默降级成模板。"""
    workflow = PostTriageWorkflow()
    spec = spec_for(PostTriageCase("S1", features.DECOY_NONE))
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_trace_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却要 trace 判别器。输出：无。预期行为：抛 RuntimeError（接线错了）。"""
    workflow = PostTriageWorkflow()
    with pytest.raises(RuntimeError, match="trace"):
        workflow.make_judges(("trace",), "zh")


def test_unknown_split_and_judge_and_locale_report_available() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：都报人话。"""
    workflow = PostTriageWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：分诊计划与切分、进程无关。"""
    case = PostTriageCase("S2+S4", features.DECOY_LOWER)
    config = PostTriageConfig(split="train", phrasing_pool=(1,))
    workflow = PostTriageWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, PostTriageSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)
    assert spec_to_dict(first)["signals"] == ["S2", "S4"]


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = PostTriageWorkflow()
    case = PostTriageCase("S3", features.DECOY_LOWER)
    state = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    parsed = workflow.reparse(state)
    assert parsed.text == text_of(case)
    assert parsed.signals == features.SIGNAL_IDS
    assert workflow.case_label(case) == "S3-lower"
    assert workflow.stats_dimensions(case) == {"signals": "S3", "decoy": "lower"}
    assert workflow.dimension_titles()["signals"]


def test_locales_are_reported() -> None:
    """输入：无。输出：无。预期行为：支持的语言与语言文件目录一致。"""
    workflow = PostTriageWorkflow()
    assert workflow.available_locales() == locales.available() == ["en", "zh"]


def test_declared_contract_fields() -> None:
    """输入：workflow 的声明项。输出：无。预期行为：题型、后端需求与判别器名单都对得上。"""
    workflow = PostTriageWorkflow()
    assert workflow.name == "posttriage"
    assert workflow.kinds == frozenset({"choice", "noul"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("hygiene", "frame", "trace")


def test_language_texts_reports_the_post() -> None:
    """输入：一个忠实样本。输出：无。

    预期行为：只报模型写的那段帖子正文——判据表是代码写的，报出去会让「产物文本与声明语言一致」
    这条尺子去量代码写的表头。
    """
    workflow = PostTriageWorkflow()
    case = PostTriageCase("S1", features.DECOY_NONE)
    sample = _sample(case)
    declared = workflow.language_texts(sample)
    assert list(declared) == ["帖子正文"]
    assert declared["帖子正文"].startswith("今天")


def _sample(case: PostTriageCase) -> dict[str, JsonValue]:
    """输入：一个 case。输出：只带 state 的样本。预期行为：正文完全兑现计划。"""
    return {"state": render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))}
