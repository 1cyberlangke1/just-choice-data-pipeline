"""检索结果判读的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import pipeline
from workflows.search import features, workflow as search_workflow

from tests.workflows._scaffold import run_config
from tests.workflows.search.conftest import FakeBackend

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="search",
        split=split,
        seed=7,
        override=search_workflow.SearchConfig(split=split, case_limit=CASE_LIMIT),
    )


def test_render_rejects_a_source_word_in_the_title(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：标题里写了来源词的假后端。输出：RunResult。

    预期行为：渲染期就拒掉，拒绝码是 `E_ASSEMBLE`（生成阶段的统一拒绝码）——
    这类文本拼进 state 就洗不掉了。
    """
    result = pipeline.run(make_config(tmp_path, "s4", fake_backend(source_title=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_render_retries_then_succeeds(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：标题写成多行的假后端。输出：RunResult。预期行为：重试耗尽后仍被拒，不静默放行。"""
    result = pipeline.run(make_config(tmp_path, "s5", fake_backend(multi_line_title=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：同分布三个切分互不相交、`test_ood` 与它们不相交，四者并起来等于全空间——
    一个相容组合都不丢、也不重复。
    """
    wf = search_workflow.SearchWorkflow()
    parts: dict[str, set[tuple[tuple[str, str], ...]]] = {}
    for split in wf.split_names():
        cases = wf.build_cases(search_workflow.SearchConfig(split=split))
        parts[split] = {tuple(case.as_dict().items()) for case in cases}
    assert parts["train"] & parts["val"] == set()
    assert parts["train"] & parts["test_iid"] == set()
    assert parts["val"] & parts["test_iid"] == set()
    for split in ("train", "val", "test_iid"):
        assert parts[split] & parts["test_ood"] == set()
    union = set().union(*parts.values())
    assert len(union) == sum(len(part) for part in parts.values())
    assert union == {tuple(case.as_dict().items()) for case in features.build_cases(1024)}


def test_id_side_has_no_dead_options() -> None:
    """输入：同分布侧的 case 列表。输出：无。

    预期行为：`test_ood` 留出的取值在同分布侧一次都不出现，而且**每一族每一档都还到得了**
    ——留出一个取值连带杀死别的族的好几档，就是死选项。
    """
    cases = [
        case
        for split in ("train", "val", "test_iid")
        for case in search_workflow.cases_for(search_workflow.SearchConfig(split=split))
    ]
    assert cases
    for case in cases:
        assert case.freshness not in search_workflow.OOD_FRESHNESSES
    labels = [case.label() for case in cases]
    assert {label.rule for label in labels} == set(features.RULES)
    assert {label.verdict for label in labels} == set(features.VERDICTS)
    assert {label.trust for label in labels} == set(range(features.TRUST_LEVELS))
    assert {label.query_kind for label in labels} == set(features.QUERY_KINDS)
    assert {label.topicality for label in labels} == set(features.TOPICALITIES)
    assert {label.spam for label in labels} == set(features.SPAMS)


def test_ood_side_only_carries_the_held_out_value() -> None:
    """输入：`test_ood` 的 case 列表。输出：无。预期行为：每一条都带留出的那个取值。"""
    cases = search_workflow.cases_for(search_workflow.SearchConfig(split="test_ood"))
    assert cases
    for case in cases:
        assert case.freshness in search_workflow.OOD_FRESHNESSES


def test_bad_knobs_are_rejected() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话。"""
    with pytest.raises(ValueError):
        search_workflow.SearchConfig(split="train", case_limit=-1)
    with pytest.raises(ValueError):
        search_workflow.SearchConfig(split="train", phrasing_pool=())
    with pytest.raises(KeyError):
        search_workflow.SearchWorkflow().config_for("nope")

