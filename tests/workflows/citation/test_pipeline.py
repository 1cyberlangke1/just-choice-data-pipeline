"""引文核查的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod
from jc.jsontypes import JsonValue

from workflows.citation import features, locales, render, workflow as citation_workflow
from workflows.citation.spec import CitationSpec, spec_to_dict

from tests.workflows.citation.conftest import SENTENCES, FakeBackend, spec_for

CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train", limit: int = CASE_LIMIT
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。

    预期行为：**关掉语言闸**（`language_gate=False`）——本包的假后端写的是英文标题与正文
    （长度按两种语言的预算挑的，见 `conftest`），zh 侧因此不是汉字。真机 zh 产物不是这样，
    这是夹具的局限；语言闸本身由 `tests/core` 的定点测试与认 `locale` 的假后端验。
    """
    return pipeline.RunConfig(
        workflow="citation",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=citation_workflow.CitationConfig(split=split, case_limit=limit),
        backend=backend,
        language_gate=False,
    )


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，而且
    `section` 一族真的出得来（小批量选 case 是覆盖优先的，能查到的引文不会被漏掉）。
    """
    result = pipeline.run(make_config(tmp_path, "c1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("citation").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    assert report["train"]["families"]["section"] > 0
    for counter in (
        "doc_bad",
        "quote_bad",
        "claim_bad",
        "verdict_bad",
        "in_doc_bad",
        "section_bad",
        "in_claimed_bad",
    ):
        assert report["train"][counter] == 0, counter


@pytest.mark.parametrize(
    ("switch", "code"),
    [
        ("duplicate", "E_ASSEMBLE"),
        ("drop_sentence", "E_ASSEMBLE"),
        ("no_digit", "E_ASSEMBLE"),
        ("short_sentence", "E_ASSEMBLE"),
        ("fence_line", "E_ASSEMBLE"),
        ("title_fence", "E_ASSEMBLE"),
        ("trace_lies", "E_JUDGE_TRACE"),
    ],
)
def test_bad_drafts_and_bad_traces_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：各种作弊开关。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是该有的那一个——写歪的正文是渲染期的 `E_ASSEMBLE`
    （重试耗尽后仍被拒，不静默放行），反解说反话是 `E_JUDGE_TRACE`。
    """
    result = pipeline.run(make_config(tmp_path, f"c3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：四份互不相交，合起来正好是全空间 36 组（同分布侧剔除留出取值后 18 组，
    留出侧 18 组）。
    """
    keys: dict[tuple[tuple[str, JsonValue], ...], str] = {}
    for split in ALL_SPLITS:
        cases = citation_workflow.cases_for(citation_workflow.CitationConfig(split=split))
        assert cases, split
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in keys, f"{case.describe()} 同时落在 {keys[key]} 与 {split}"
            keys[key] = split
    assert len(keys) == 36
    ood = {
        tuple(case.as_dict().items())
        for case in citation_workflow.cases_for(
            citation_workflow.CitationConfig(split="test_ood")
        )
    }
    assert {key[3][1] for key in ood} == set(citation_workflow.OOD_TARGETS)


def test_id_splits_keep_every_answer_reachable() -> None:
    """输入：同分布三个切分的全部 case。输出：无。

    预期行为：三种结论、三节、两个布尔值在同分布侧都取得到——留出取值要是把某一族的答案压成
    常量，这一条会红。
    """
    labels = [
        features.label_for(features.plan_for(case, phrasing))
        for split in ID_SPLITS
        for case in citation_workflow.cases_for(citation_workflow.CitationConfig(split=split))
        for phrasing in (0, 1)
    ]
    assert {item.verdict for item in labels} == set(features.VERDICTS)
    assert {item.section for item in labels if item.in_doc} == set(range(features.SECTIONS))
    assert {item.in_doc for item in labels} == {True, False}
    assert {item.at_claimed for item in labels} == {True, False}


def test_full_train_run_is_balanced(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：整个 train 切分跑一遍。输出：RunResult。预期行为：各族的答案均衡全绿。"""
    result = pipeline.run(make_config(tmp_path, "c4-full", fake_backend(), limit=0))
    assert result.accepted == len(
        citation_workflow.cases_for(citation_workflow.CitationConfig(split="train"))
    )
    assert [item["ok"] for item in result.stats["balance"]] == [True] * len(
        result.stats["balance"]
    )


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow。输出：无。预期行为：抛 RuntimeError，不静默降级成模板。"""
    workflow = citation_workflow.CitationWorkflow()
    case = features.CitationCase("exact", "same", "first", "first")
    spec = spec_for(case)
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_trace_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却要 trace 判别器。输出：无。预期行为：抛 RuntimeError（接线错了）。"""
    workflow = citation_workflow.CitationWorkflow()
    with pytest.raises(RuntimeError, match="trace"):
        workflow.make_judges(("trace",), "zh")


def test_unknown_split_and_judge_and_locale_report_available() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：都报人话。"""
    workflow = citation_workflow.CitationWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：引文计划与切分、进程无关。"""
    case = features.CitationCase("partial", "other", "last", "last")
    config = citation_workflow.CitationConfig(split="train", phrasing_pool=(1,))
    workflow = citation_workflow.CitationWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, CitationSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)
    assert spec_to_dict(first)["claimed"] == first.plan.claimed


def test_topic_knob_reaches_the_prompt() -> None:
    """输入：只留一个题材方向的采样参数。输出：无。

    预期行为：`plan_sample` 抽出来的题材真的传到了提示词里（范式 §9.57 的样本级变量：它不进 case
    空间、不进 `stats_dimensions`，只改短文与标题怎么写）。首版一个变量都没有，真机 9 条 en 产物
    的标题全是同一个字符串。
    """
    case = features.CitationCase("exact", "same", "first", "first")
    workflow = citation_workflow.CitationWorkflow()
    config = citation_workflow.CitationConfig(split="train", topics=(7,))
    spec = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    assert isinstance(spec, CitationSpec)
    assert spec.topic == 7
    assert spec_to_dict(spec)["topic"] == 7
    loc = locales.load("zh")
    assert locales.pick(loc["prompt"]["topics"], 7) in render.build_prompt(loc, spec.topic)
    assert workflow.stats_dimensions(case) == {
        "form": "exact",
        "where": "same",
        "place": "first",
        "target": "first",
    }


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = citation_workflow.CitationWorkflow()
    case = features.CitationCase("exact", "same", "last", "last")
    spec = spec_for(case)
    backend = FakeBackend()
    draft = backend.generate(render.build_task(spec, locales.load("zh")))
    state = render.to_state(
        render.CitationDraft(
            title=str(draft.data["title"]),
            sentences=[str(item) for item in draft.data["sentences"]],
        ),
        spec.plan,
        locales.load("zh"),
    )
    parsed = workflow.reparse(state)
    assert parsed.sentences == SENTENCES
    assert parsed.claimed == spec.plan.claimed
    assert workflow.case_label(case) == "exact-same-last-last"
    assert workflow.stats_dimensions(case) == {
        "form": "exact",
        "where": "same",
        "place": "last",
        "target": "last",
    }
    assert workflow.dimension_titles()["form"]


def test_locales_are_reported() -> None:
    """输入：无。输出：无。预期行为：支持的语言与语言文件目录一致。"""
    workflow = citation_workflow.CitationWorkflow()
    assert workflow.available_locales() == locales.available() == ["en", "zh"]


def test_declared_contract_fields() -> None:
    """输入：workflow 的声明项。输出：无。预期行为：题型、后端需求与判别器名单都对得上。"""
    workflow = citation_workflow.CitationWorkflow()
    assert workflow.name == "citation"
    assert workflow.kinds == frozenset({"choice", "noul"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("hygiene", "frame", "trace")
