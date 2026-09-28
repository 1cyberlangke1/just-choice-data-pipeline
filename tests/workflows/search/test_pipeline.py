"""检索结果判读的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod
from workflows.search import features, workflow as search_workflow

from tests.workflows.search.conftest import FakeBackend

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="search",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=search_workflow.SearchConfig(split=split, case_limit=CASE_LIMIT),
        backend=backend,
    )


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，
    而且 `verdict` 每一条都与「命中规则」自洽。
    """
    result = pipeline.run(make_config(tmp_path, "s1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("search").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    assert report["train"]["frame_bad"] == 0
    assert report["train"]["rules_bad"] == 0
    assert report["train"]["verdict_bad"] == 0


def test_two_runs_are_byte_identical(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：同种子跑两次。输出：两份产物。预期行为：逐字节一致。"""
    first = pipeline.run(make_config(tmp_path, "s2a", fake_backend()))
    second = pipeline.run(make_config(tmp_path, "s2b", fake_backend()))
    assert (first.run_dir / "train.jsonl").read_bytes() == (
        second.run_dir / "train.jsonl"
    ).read_bytes()


def test_trace_judge_rejects_a_misread_kind(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解说反话的假后端。输出：RunResult。

    预期行为：这些样本被拒，拒绝码是 `E_JUDGE_TRACE`——判别器抓的是模型真会犯的错。
    """
    result = pipeline.run(make_config(tmp_path, "s3", fake_backend(trace_lies_kind=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


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


def test_render_without_backend_raises() -> None:
    """输入：没注入后端的 workflow。输出：无。

    预期行为：直接报错，绝不悄悄降级成模板——那会让「文本由模型写」这条前提失效。
    """
    wf = search_workflow.SearchWorkflow()
    spec = wf.sample(
        __import__("random").Random(0),
        wf.plan_sample(
            __import__("random").Random(0),
            search_workflow.SearchConfig(split="train"),
            features.build_cases(1024)[0],
            0,
        ),
    )
    with pytest.raises(RuntimeError):
        wf.render(spec, __import__("random").Random(0), "zh")


def test_unknown_judge_name_raises() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：直接报错。"""
    with pytest.raises(KeyError):
        search_workflow.SearchWorkflow().make_judges(("nope",), "zh")
