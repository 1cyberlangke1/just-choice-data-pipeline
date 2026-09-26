"""判别器的单元测试：独立重算、排序口径、题面编号读取、坏产物拦阻。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.search import judges, locales, parse, render, questions
from workflows.search import spec as sp
from workflows.search import workflow as wf


def page_of(spec: sp.SearchSpec, locale: str = "zh") -> parse.ParsedPage:
    """输入：规格、语言。输出：渲染后反解回来的页。"""
    loc = locales.load(locale)
    state = render.build(spec, loc)
    return parse.parse_page(state, parse.label_index(loc), (loc["meta"]["field_sep"],))


def test_recompute_matches_the_construction_on_every_id_case() -> None:
    """输入：同分布三个切分的全部 case × 两种语言。输出：无。

    预期行为：从**文本**独立重算出来的答案，与构造侧按表格推出来的答案逐题相等——
    这是「构造与判别共用一套规则」的硬证据，也是两侧实现不许走偏的那道闸。
    """
    workflow = wf.SearchWorkflow()
    checked = 0
    for locale in locales.available():
        loc = locales.load(locale)
        for split in ("train", "val", "test_iid"):
            for case in workflow.build_cases(workflow.config_for(split)):
                spec = sp.sample_spec(random.Random(0), case, split, 0, 0)
                page = page_of(spec, locale)
                expected = judges.recompute(page, loc, case.probe, case.pair)
                built = {item.family: item.answer for item in questions.build(spec, loc, 0, random.Random(0))}
                assert expected == built, (locale, case.describe())
                checked += 1
    assert checked > 4000


def test_recompute_reads_the_state_not_the_spec(make_spec: Any) -> None:
    """输入：把渲染出来的文本改一处（最佳结果那条的摘要删掉）。输出：重算结果。

    预期行为：答案跟着**文本**变——判别器看的是页面，不是 spec。
    """
    loc = locales.load("zh")
    spec = make_spec(best_flavor="full")
    page = page_of(spec)
    before = judges.recompute(page, loc, spec.case.probe, spec.case.pair)
    assert before["answerable"] is True
    broken = page.results[spec.case.best_at - 1]
    patched = parse.ParsedPage(
        query=page.query,
        want_kind=page.want_kind,
        since=page.since,
        points=page.points,
        results=tuple(
            row if row.index != broken.index else parse.ParsedResult(
                index=row.index,
                doc_id=row.doc_id,
                title=row.title,
                kind=row.kind,
                authority=row.authority,
                updated=row.updated,
                snippet="（没有摘要）",
            )
            for row in page.results
        ),
    )
    after = judges.recompute(patched, loc, spec.case.probe, spec.case.pair)
    assert after["answerable"] is False
    assert after["next_action"] == "search_more"


def test_question_indices_come_from_the_instructions() -> None:
    """输入：题面里编号不一致 / 数字个数不对的样本。输出：异常。

    预期行为：从题面读编号；不一致就报错，不猜也不取第一个——编号对不上说明渲染与出题走偏了。
    """
    sample = {
        "questions": {
            "relevance": {"type": "score", "instructions": "结果 3 的相关度是几级？", "criteria": []},
            "conflict": {"type": "noul", "instructions": "结果 1 与结果 2 是否互相矛盾？", "criteria": {}},
        }
    }
    assert judges.question_indices(sample) == (3, (1, 2))
    sample["questions"]["covered_points"] = {
        "type": "score",
        "instructions": "结果 4 覆盖了几个要点？",
        "criteria": [],
    }
    with pytest.raises(ValueError, match="编号不一致"):
        judges.question_indices(sample)
    sample["questions"]["probe_fresh"] = {
        "type": "noul",
        "instructions": "结果 1 到 5 之间有几个？",
        "criteria": {},
    }
    with pytest.raises(ValueError, match="不是恰好一个"):
        judges.question_indices(sample)


def test_topic_of_query_reports_an_unrecognised_question() -> None:
    """输入：问题里认不出主题。输出：异常。预期行为：报错点名，不静默挑一个。"""
    loc = locales.load("zh")
    page = parse.ParsedPage(query="今天天气怎么样？", want_kind="操作手册", since="2024-01-01", points=(), results=())
    with pytest.raises(ValueError, match="主题不是恰好一个"):
        judges.topic_of_query(page, loc)


def test_best_of_prefers_relevance_then_authority_then_date() -> None:
    """输入：三条精心摆好的事实。输出：最优。预期行为：三档 tie-break 依次生效。"""
    def row(index: int, *, topic: bool, kind: bool, fresh: bool, authority: str, updated: str) -> sp.TruthRow:
        return sp.TruthRow(
            index=index,
            doc_id=f"KB-{index}",
            topic_match=topic,
            kind_match=kind,
            fresh=fresh,
            authority=authority,
            updated=updated,
            snippet="",
        )

    low = row(1, topic=True, kind=True, fresh=False, authority="official", updated="2026-01-01")
    community = row(2, topic=True, kind=True, fresh=True, authority="community", updated="2026-06-01")
    official = row(3, topic=True, kind=True, fresh=True, authority="official", updated="2026-05-01")
    assert sp.best_of((low, community, official)).index == 3  # 相关度优先
    newer_blog = row(4, topic=True, kind=True, fresh=True, authority="blog", updated="2026-12-01")
    assert sp.best_of((community, official, newer_blog)).index == 3  # 权威优先于日期
    older_official = row(5, topic=True, kind=True, fresh=True, authority="official", updated="2026-02-01")
    assert sp.best_of((official, older_official)).index == 3  # 权威相同时日期晚者优先
    with pytest.raises(ValueError, match="一条结果都没有"):
        sp.best_of(())


def test_conflict_of_needs_the_metric_on_both_sides() -> None:
    """输入：几种摘要组合。输出：矛盾与否。

    预期行为：两条都提到指标、且取值集合不同才算矛盾——只有一条提到、或取值一样都不算。
    """
    def row(index: int, snippet: str) -> sp.TruthRow:
        return sp.TruthRow(
            index=index,
            doc_id=f"KB-{index}",
            topic_match=True,
            kind_match=True,
            fresh=True,
            authority="official",
            updated="2026-01-01",
            snippet=snippet,
        )

    metric, values = "金额上限", ("5000 元", "20000 元")
    assert sp.conflict_of(row(1, f"金额上限为 {values[0]}"), row(2, f"金额上限为 {values[1]}"), metric, values)
    assert not sp.conflict_of(row(1, f"金额上限为 {values[0]}"), row(2, f"金额上限为 {values[0]}"), metric, values)
    assert not sp.conflict_of(row(1, f"金额上限为 {values[0]}"), row(2, "没有指标"), metric, values)


def test_roundtrip_judge_skips_without_a_spec(make_spec: Any) -> None:
    """输入：没有 spec。输出：Verdict。预期行为：跳过（`skipped=True`），不误判。"""
    verdict = judges.RoundtripJudge().judge({"state": "随便"}, None)
    assert verdict.ok and verdict.raw["skipped"] is True


def test_roundtrip_judge_catches_a_truncated_state(make_spec: Any) -> None:
    """输入：被砍掉最后一条（标题 + 摘要）的 state。输出：Verdict。

    预期行为：判不通过并报出「结果条数」差异——少一条必须当场看出来。
    """
    loc = locales.load("zh")
    spec = make_spec()
    lines = render.build(spec, loc).splitlines()
    verdict = judges.RoundtripJudge().judge({"state": "\n".join(lines[:-2])}, spec)
    assert not verdict.ok
    assert "结果条数" in verdict.reason


def test_roundtrip_judge_catches_a_swapped_field(make_spec: Any) -> None:
    """输入：把类型与来源两格对调的 state。输出：Verdict。预期行为：判不通过。"""
    loc = locales.load("zh")
    spec = make_spec()
    state = render.build(spec, loc)
    sep = loc["meta"]["field_sep"]
    parts = state.split(sep)
    parts[2], parts[3] = parts[3], parts[2]
    verdict = judges.RoundtripJudge().judge({"state": sep.join(parts)}, spec)
    assert not verdict.ok


def test_label_judge_flags_a_wrong_answer(make_spec: Any) -> None:
    """输入：把答案改错一条的样本。输出：Verdict。预期行为：判不通过并点名题号。"""
    loc = locales.load("zh")
    spec = make_spec()
    labeled = questions.build(spec, loc, 0, random.Random(0))
    from jc import assemble

    sample = assemble.assemble_sample("search-train-0000000", render.build(spec, loc), labeled, random.Random(0))
    assert judges.LabelJudge().judge(sample).ok
    sample["answers"]["answerable"]["noul"] = not sample["answers"]["answerable"]["noul"]
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    assert "answerable" in verdict.reason
