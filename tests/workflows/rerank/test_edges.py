"""边界与错误路径：坏旋钮、坏名字、坏语言都报人话。"""

from __future__ import annotations

from typing import Any

import pytest

from jc import workflow as workflow_mod
from workflows.rerank import judges, locales, parse, render
from workflows.rerank import spec as sp

WORKFLOW = "rerank"


def test_unknown_split_is_reported() -> None:
    """输入：不存在的切分名。输出：无。预期行为：KeyError 并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        workflow_mod.load(WORKFLOW).config_for("nope")


def test_split_names_are_the_documented_ones() -> None:
    """输入：无。输出：无。预期行为：三个同分布切分 + test_ood。"""
    assert workflow_mod.load(WORKFLOW).split_names() == ("train", "val", "test_iid", "test_ood")


def test_unknown_judge_is_reported() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：KeyError 并列出可用的。"""
    with pytest.raises(KeyError, match="未知判别器"):
        workflow_mod.load(WORKFLOW).make_judges(("nope",), "zh")


def test_bind_backend_is_refused() -> None:
    """输入：给纯构造式 workflow 绑后端。输出：无。预期行为：AssertionError（接线错了）。"""
    with pytest.raises(AssertionError, match="纯构造式"):
        workflow_mod.load(WORKFLOW).bind_backend(object())


def test_workflow_declares_no_backend_need() -> None:
    """输入：无。输出：无。预期行为：`needs` 为空（零 token），三种题型与两个判别器齐备。"""
    workflow = workflow_mod.load(WORKFLOW)
    assert workflow.needs == frozenset()
    assert workflow.kinds == frozenset({"choice", "noul", "score"})
    assert workflow.judges == ("roundtrip", "label")


def test_unknown_language_lists_available() -> None:
    """输入：不存在的语言。输出：无。预期行为：FileNotFoundError 并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="zh"):
        locales.load("ja")


def test_pick_wraps_around() -> None:
    """输入：越界的问法编号。输出：无。预期行为：回绕（问法措辞不是判据）。"""
    assert locales.pick(["a", "b"], 5) == "b"


def test_reverse_rejects_duplicate_words() -> None:
    """输入：两个键共用一个词。输出：无。预期行为：报错（反解会把两个键认成一个）。"""
    with pytest.raises(ValueError, match="同时属于"):
        locales.reverse({"official": "官方", "vendor": "官方"})


def test_dimension_titles_cover_all_five_dimensions() -> None:
    """输入：无。输出：无。预期行为：五个维度都有中文标题（报告里才不会漏维度）。"""
    workflow = workflow_mod.load(WORKFLOW)
    case = workflow.build_cases(workflow.config_for("train"))[0]
    titles = workflow.dimension_titles()
    assert set(titles) == set(workflow.stats_dimensions(case))
    assert set(titles) == {"topic", "aspect", "shape", "traps", "wording"}


def test_case_label_is_readable() -> None:
    """输入：一个 case。输出：无。预期行为：标签是一行可读描述，五个维度都露出来。"""
    workflow = workflow_mod.load(WORKFLOW)
    case = workflow.build_cases(workflow.config_for("train"))[0]
    label = workflow.case_label(case)
    assert "\n" not in label
    assert case.topic in label and case.aspect in label and case.wording in label


def test_reparse_returns_a_parsed_list(make_spec: Any) -> None:
    """输入：渲染后的 state。输出：无。预期行为：`reparse` 用合并标签表反解成功。"""
    spec = make_spec()
    loc = locales.load("zh")
    parsed = workflow_mod.load(WORKFLOW).reparse(render.build(spec, loc))
    assert isinstance(parsed, parse.ParsedList)
    assert parsed.cands[0].body == render.cand_lines(spec.case, 0, loc)


def test_missing_body_is_caught_by_the_roundtrip_judge(make_spec: Any) -> None:
    """输入：把候选正文整段删掉。输出：无。预期行为：结构判别判不通过并说明反解失败。"""
    loc = locales.load("zh")
    spec = make_spec()
    head = render.build(spec, loc).split(loc["meta"]["cand_label"])[0]
    sample = {"state": head, "questions": {}, "answers": {}}
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok
    assert "反解失败" in verdict.reason


def test_config_rejects_unknown_topic_naming_the_field() -> None:
    """输入：不存在的主题。输出：无。预期行为：ValueError 点名 `topics`。"""
    with pytest.raises(ValueError, match="topics"):
        sp.RerankConfig(topics=("backup", "nope"))


def test_config_rejects_unknown_aspect_naming_the_field() -> None:
    """输入：不存在的方面。输出：无。预期行为：ValueError 点名 `aspects`。"""
    with pytest.raises(ValueError, match="aspects"):
        sp.RerankConfig(aspects=("steps", "nope"))


def test_config_rejects_an_unknown_source() -> None:
    """输入：画像里出现不存在的来源。输出：无。预期行为：ValueError 点名来源。"""
    bad = sp._shape(((1, 1, 1, "nope", "2024-07-05"),) * 4, 2, (0, 1), (1, 2), 0)
    with pytest.raises(ValueError, match="来源"):
        sp.RerankConfig(shapes=(bad,))


@pytest.mark.parametrize(
    "field", ("topics", "aspects", "shapes", "traps", "wordings", "phrasings")
)
def test_config_rejects_empty_pools_naming_the_field(field: str) -> None:
    """输入：六个旋钮各传一次空表。输出：无。预期行为：ValueError 点名那个字段。"""
    with pytest.raises(ValueError, match=field):
        sp.RerankConfig(**{field: ()})
