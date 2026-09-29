"""检索重排的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import jsonio, pipeline, workflow as workflow_mod
from jc.jsontypes import JsonValue

from workflows.rerank import features, locales, render, workflow as rerank_workflow
from workflows.rerank.spec import RerankSpec, spec_to_dict

from tests.workflows.rerank.conftest import FakeBackend

CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train", limit: int = CASE_LIMIT
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="rerank",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=rerank_workflow.RerankConfig(split=split, case_limit=limit),
        backend=backend,
    )


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，而且
    `top` 一族真的出得来（小批量选 case 是覆盖优先的，`relevant>0` 不会被漏掉）。
    """
    result = pipeline.run(make_config(tmp_path, "c1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("rerank").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    assert report["train"]["families"]["top"] > 0
    for counter in (
        "terms_bad",
        "candidates_bad",
        "rule_bad",
        "top_bad",
        "count_bad",
        "pair_bad",
        "order_bad",
    ):
        assert report["train"][counter] == 0, counter


def test_pair_family_is_balanced_in_the_products(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：整批产物。输出：无。预期行为：`pair` 一族的 True / False 严格各占一半。"""
    result = pipeline.run(make_config(tmp_path, "c1b", fake_backend()))
    yes = no = 0
    for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl"):
        for qid, answer in row["answers"].items():
            if qid == "pair" or qid.startswith("pair_"):
                if answer["noul"]:
                    yes += 1
                else:
                    no += 1
    assert yes == no > 0


def test_two_runs_are_byte_identical(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：同种子跑两次。输出：两份产物。预期行为：逐字节一致。"""
    first = pipeline.run(make_config(tmp_path, "c2a", fake_backend()))
    second = pipeline.run(make_config(tmp_path, "c2b", fake_backend()))
    assert (first.run_dir / "train.jsonl").read_bytes() == (
        second.run_dir / "train.jsonl"
    ).read_bytes()


@pytest.mark.parametrize(
    ("switch", "code"),
    [
        ("unrealized", "E_ASSEMBLE"),
        ("extra_hit", "E_ASSEMBLE"),
        ("duplicate", "E_ASSEMBLE"),
        ("drop_candidate", "E_ASSEMBLE"),
        ("bad_terms", "E_ASSEMBLE"),
        ("term_missing_from_query", "E_ASSEMBLE"),
        ("fence_line", "E_ASSEMBLE"),
        ("trace_lies", "E_JUDGE_TRACE"),
    ],
)
def test_bad_drafts_and_bad_traces_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：各种作弊开关。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是该有的那一个——写歪的候选是渲染期的 `E_ASSEMBLE`
    （重试耗尽后仍被拒，不静默放行），反解说反话是 `E_JUDGE_TRACE`。
    """
    result = pipeline.run(make_config(tmp_path, f"c3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：同分布三个切分互不相交、`test_ood` 与它们不相交，四份合起来正好是全空间
    （同分布侧剔除留出取值后是 30 组，留出侧 15 组，合计 45 组）。
    """
    keys: dict[tuple[tuple[str, JsonValue], ...], str] = {}
    for split in ("train", "val", "test_iid", "test_ood"):
        cases = rerank_workflow.cases_for(rerank_workflow.RerankConfig(split=split))
        assert cases
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in keys, f"{key} 同时出现在 {keys[key]} 与 {split}"
            keys[key] = split
    assert len(keys) == (
        len(features.TERM_COUNTS) * len(features.RELEVANT) * len(features.BESTS)
    )


def test_id_splits_never_carry_the_held_out_term_count() -> None:
    """输入：同分布三个切分。输出：无。预期行为：一条都不带留出的关键词个数。"""
    for split in ID_SPLITS:
        for case in rerank_workflow.cases_for(rerank_workflow.RerankConfig(split=split)):
            assert case.terms not in rerank_workflow.OOD_TERMS


def test_ood_split_is_only_the_held_out_term_count() -> None:
    """输入：`test_ood` 的 case 列表。输出：无。预期行为：全是留出的关键词个数，其余维照旧全。"""
    cases = rerank_workflow.cases_for(rerank_workflow.RerankConfig(split="test_ood"))
    assert {case.terms for case in cases} == set(rerank_workflow.OOD_TERMS)
    assert {case.relevant for case in cases} == set(features.RELEVANT)
    assert {case.best for case in cases} == set(features.BESTS)


def test_every_answer_value_is_reachable_without_the_held_out_value() -> None:
    """输入：同分布三个切分合起来。输出：无。

    预期行为：留出「关键词个数」之后，每一族该出现的取值都还到得了——`count` 的 0~4 五档、
    `top` 的五个候选、`pair` 的两个布尔值，一个都不缺。留出取值要是把某一族的答案压成常量，
    这一条会红。
    """
    plans = [
        features.plan_for(case, phrasing)
        for split in ID_SPLITS
        for case in rerank_workflow.cases_for(rerank_workflow.RerankConfig(split=split))
        for phrasing in (0, 1)
    ]
    assert {plan.relevant for plan in plans} == set(features.RELEVANT)
    labels = [features.label_for(plan) for plan in plans]
    assert {item.top for item in labels if item.top is not None} == set(range(features.CANDIDATES))
    answers = {item.above(left, right) for item in labels for left, right in features.PAIRS}
    assert answers == {True, False}
    order_texts = {item.order[:3] for item in labels}
    assert len(order_texts) > 1


def test_full_train_run_is_balanced(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：整个 train 切分跑一遍。输出：RunResult。预期行为：各族的答案均衡全绿。"""
    result = pipeline.run(make_config(tmp_path, "c4-full", fake_backend(), limit=0))
    assert result.accepted == len(
        rerank_workflow.cases_for(rerank_workflow.RerankConfig(split="train"))
    )
    assert [item["ok"] for item in result.stats["balance"]] == [True] * len(
        result.stats["balance"]
    )


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow。输出：无。预期行为：抛 RuntimeError，不静默降级成模板。"""
    workflow = rerank_workflow.RerankWorkflow()
    case = features.RerankCase(terms=2, relevant=2, best="first")
    spec = RerankSpec(case=case, phrasing=0, split="train", seq=0, plan=features.plan_for(case, 0))
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_trace_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却要 trace 判别器。输出：无。预期行为：抛 RuntimeError（接线错了）。"""
    workflow = rerank_workflow.RerankWorkflow()
    with pytest.raises(RuntimeError, match="trace"):
        workflow.make_judges(("trace",), "zh")


def test_unknown_split_and_judge_and_locale_report_available() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：都报人话。"""
    workflow = rerank_workflow.RerankWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：名次计划与切分、进程无关。"""
    case = features.RerankCase(terms=3, relevant=4, best="last")
    config = rerank_workflow.RerankConfig(split="train", phrasing_pool=(1,))
    workflow = rerank_workflow.RerankWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, RerankSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)
    assert spec_to_dict(first)["overlaps"] == list(first.plan.overlaps)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = rerank_workflow.RerankWorkflow()
    case = features.RerankCase(terms=2, relevant=1, best="first")
    plan = features.plan_for(case, 0)
    backend = FakeBackend()
    spec = RerankSpec(case=case, phrasing=0, split="train", seq=0, plan=plan)
    draft = backend.generate(render.build_task(spec, locales.load("zh")))
    state = render.to_state(
        render.RerankDraft(
            query=str(draft.data["query"]),
            terms=[str(item) for item in draft.data["terms"]],
            candidates=[str(item) for item in draft.data["candidates"]],
        ),
        plan,
        locales.load("zh"),
    )
    parsed = workflow.reparse(state)
    assert parsed.candidates == tuple(str(item) for item in draft.data["candidates"])
    assert workflow.case_label(case) == "2词-1相关-first"
    assert workflow.stats_dimensions(case) == {"terms": 2, "relevant": 1, "best": "first"}
    assert workflow.dimension_titles()["relevant"]


def test_locales_are_reported() -> None:
    """输入：无。输出：无。预期行为：支持的语言与语言文件目录一致。"""
    workflow = rerank_workflow.RerankWorkflow()
    assert workflow.available_locales() == locales.available() == ["en", "zh"]
