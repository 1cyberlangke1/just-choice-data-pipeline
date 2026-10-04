"""检索重排的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import jsonio, pipeline
from jc.jsontypes import JsonValue

from workflows.rerank import features, locales, render, workflow as rerank_workflow
from workflows.rerank.spec import RerankSpec

from tests.workflows._scaffold import run_config
from tests.workflows.rerank.conftest import FakeBackend

CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train", limit: int = CASE_LIMIT
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。

    预期行为：**关掉语言闸**（`language_gate=False`）——本包的假后端按注入的关键词拼英文查询
    （`how do i fix … on the server`），zh 侧因此不是汉字。真机 zh 产物不是这样，这是夹具的
    局限；语言闸本身由 `tests/core` 的定点测试与认 `locale` 的假后端验。
    """
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="rerank",
        split=split,
        seed=7,
        override=rerank_workflow.RerankConfig(split=split, case_limit=limit),
        language_gate=False,
    )


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

