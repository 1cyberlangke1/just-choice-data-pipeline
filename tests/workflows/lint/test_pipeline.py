"""行级规则检查的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import pipeline
from jc.jsontypes import JsonValue

from workflows.lint import features, locales, render, workflow as lint_workflow
from workflows.lint.spec import LintSpec

from tests.workflows._scaffold import run_config
from tests.workflows.lint.conftest import FakeBackend

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train", limit: int = CASE_LIMIT
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="lint",
        split=split,
        seed=7,
        override=lint_workflow.LintConfig(split=split, case_limit=limit),
    )


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：同分布三个切分互不相交、`test_ood` 与它们不相交，四份合起来正好是全空间
    （同分布侧剔除留出体裁后是 90 组，留出侧 45 组，合计 135 组）。
    """
    keys: dict[tuple[tuple[str, JsonValue], ...], str] = {}
    for split in ("train", "val", "test_iid", "test_ood"):
        cases = lint_workflow.cases_for(lint_workflow.LintConfig(split=split))
        assert cases
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in keys, f"{key} 同时出现在 {keys[key]} 与 {split}"
            keys[key] = split
    assert len(keys) == 135


def test_id_splits_never_carry_the_held_out_genre() -> None:
    """输入：同分布三个切分。输出：无。预期行为：一条都不带留出的体裁取值。"""
    for split in ("train", "val", "test_iid"):
        for case in lint_workflow.cases_for(lint_workflow.LintConfig(split=split)):
            assert case.genre not in lint_workflow.OOD_GENRES


def test_ood_split_is_only_the_held_out_genre() -> None:
    """输入：`test_ood` 的 case 列表。输出：无。预期行为：全是留出体裁，且规则与命中数照旧全。"""
    cases = lint_workflow.cases_for(lint_workflow.LintConfig(split="test_ood"))
    assert {case.genre for case in cases} == set(lint_workflow.OOD_GENRES)
    assert {case.rule for case in cases} == set(features.RULES)
    assert {case.hits for case in cases} == set(features.HITS)


def test_every_answer_value_is_reachable_without_the_held_out_genre() -> None:
    """输入：同分布三个切分合起来。输出：无。

    预期行为：留出体裁之后，每一族该出现的取值都还到得了——`count` 的 0~8 九个取值、`gate`
    的三档、`spot` 的八个行号，一个都不缺。留出取值要是把某一族的答案压成常量，这一条会红。
    """
    plans = [
        features.plan_for(case, phrasing)
        for split in ("train", "val", "test_iid")
        for case in lint_workflow.cases_for(lint_workflow.LintConfig(split=split))
        for phrasing in (0, 1)
    ]
    assert {plan.hits for plan in plans} == set(features.HITS)
    assert {features.gate_of(plan.hits) for plan in plans} == set(features.GATES)
    spots = {features.label_for(plan).spot for plan in plans if plan.hits == 1}
    assert spots == set(range(1, features.SIZE + 1))


def test_full_train_run_is_balanced(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：整个 train 切分跑一遍。输出：RunResult。预期行为：三个维度的答案均衡全绿。"""
    result = pipeline.run(
        make_config(tmp_path, "c4-full", fake_backend(), limit=0)
    )
    assert result.accepted == len(lint_workflow.cases_for(lint_workflow.LintConfig(split="train")))
    assert [item["ok"] for item in result.stats["balance"]] == [True] * len(
        result.stats["balance"]
    )


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：行计划与切分、进程无关。"""
    case = features.LintCase(features.RULE_SECRET, "log", 3)
    config = lint_workflow.LintConfig(split="train", phrasing_pool=(1,))
    workflow = lint_workflow.LintWorkflow()
    first = workflow.sample(random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0))
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, LintSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = lint_workflow.LintWorkflow()
    case = features.LintCase(features.RULE_DEBUG, "code", 1)
    plan = features.plan_for(case, 0)
    lines = [
        'print("x")' if row in plan.hit_rows else "import os" for row in range(features.SIZE)
    ]
    state = render.to_state(lines, plan, locales.load("zh"))
    parsed = workflow.reparse(state)
    assert parsed.lines == tuple(lines)
    assert workflow.case_label(case) == "debug/code/1"
    assert workflow.stats_dimensions(case) == {"rule": "debug", "genre": "code", "hits": 1}
    assert workflow.dimension_titles()["hits"]
