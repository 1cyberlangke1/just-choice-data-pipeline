"""边角与报错分支：坏输入要报人话，审计要能点名坏产物。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from jc import pipeline
from workflows.search import judges, locales, parse, render
from workflows.search import spec as sp
from workflows.search import workflow as wf

from .conftest import tiny_config
from .test_judges import page_of


def test_pair_question_with_three_numbers_is_rejected() -> None:
    """输入：矛盾那族的题面里有三个数字。输出：异常。预期行为：报错并带上题面原文。"""
    sample = {
        "questions": {
            "relevance": {"type": "score", "instructions": "结果 1 的相关度？", "criteria": []},
            "conflict": {
                "type": "noul",
                "instructions": "结果 1 与结果 2 以及结果 3 是否矛盾？",
                "criteria": {},
            },
        }
    }
    with pytest.raises(ValueError, match="不是恰好两个"):
        judges.question_indices(sample)


def test_unknown_authority_text_is_rejected(make_spec: Any) -> None:
    """输入：来源那一格是没见过的文案。输出：异常。预期行为：报错点名，不静默当最低档。"""
    loc = locales.load("zh")
    spec = make_spec()
    page = page_of(spec)
    patched = parse.ParsedPage(
        query=page.query,
        want_kind=page.want_kind,
        since=page.since,
        points=page.points,
        results=(
            parse.ParsedResult(
                index=page.results[0].index,
                doc_id=page.results[0].doc_id,
                title=page.results[0].title,
                kind=page.results[0].kind,
                authority="神秘来源",
                updated=page.results[0].updated,
                snippet=page.results[0].snippet,
            ),
            *page.results[1:],
        ),
    )
    with pytest.raises(ValueError, match="来源权威"):
        judges.truth_rows_of_page(patched, loc)


def test_missing_question_index_is_rejected(make_spec: Any) -> None:
    """输入：题面点名的编号超出这一页。输出：异常。预期行为：报错点名缺了哪几个。"""
    loc = locales.load("zh")
    spec = make_spec()
    with pytest.raises(ValueError, match="不存在"):
        judges.recompute(page_of(spec), loc, 99, (1, 2))


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("query_label", "完全换了一句话"),
        ("kind_label", "另一个类型"),
        ("since_label", "1999-01-01"),
        ("points_label", "别的、要点"),
    ],
)
def test_roundtrip_diff_names_the_broken_header_field(
    make_spec: Any, label: str, value: str
) -> None:
    """输入：请求头某一格被改掉。输出：差异列表。预期行为：点名是哪一格。"""
    loc = locales.load("zh")
    spec = make_spec()
    lines = render.build(spec, loc).splitlines()
    head = loc["meta"][label]
    replaced = [
        f"{head}{loc['meta']['colon']}{value}" if line.startswith(head) else line for line in lines
    ]
    page = parse.parse_page(
        "\n".join(replaced), parse.label_index(loc), (loc["meta"]["field_sep"],)
    )
    diff = judges.roundtrip_diff(spec, page, loc)
    assert diff, label


def test_roundtrip_judge_reports_a_parse_failure(make_spec: Any) -> None:
    """输入：一条连标题行都切不开的 state。输出：Verdict。预期行为：判不通过并说反解失败。"""
    spec = make_spec()
    verdict = judges.RoundtripJudge().judge({"state": "  [1] 只有一格"}, spec)
    assert not verdict.ok and "反解失败" in verdict.reason


def test_label_judge_reports_a_recompute_failure(make_spec: Any) -> None:
    """输入：state 里认不出主题。输出：Verdict。预期行为：判不通过并说重算失败。"""
    spec = make_spec()
    state = render.build(spec, locales.load("zh")).replace("退款", "别的东西")
    sample = {"state": state, "questions": {}, "answers": {}}
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "重算失败" in verdict.reason


def test_label_judge_names_unknown_families_and_bad_indices(make_spec: Any) -> None:
    """输入：混进未知题族、越界编号、对不上的答案。输出：Verdict。

    预期行为：逐条点名——未知题族、编号越界、答案指向错、score 对不上，一个都不许静默放过。
    """
    loc = locales.load("zh")
    spec = make_spec()
    state = render.build(spec, loc)
    sample = {
        "state": state,
        "questions": {
            "made_up": {"type": "noul", "instructions": "?", "criteria": {"true": "a", "false": "b"}},
            "best_result": {"type": "choice", "instructions": "?", "criteria": {"result_1": "甲", "result_2": "乙"}},
            "next_action": {"type": "choice", "instructions": "?", "criteria": {"answer": "甲", "search_more": "乙"}},
            "relevance": {"type": "score", "instructions": f"结果 {spec.case.probe} 的相关度？", "criteria": []},
            "covered_points": {"type": "score", "instructions": f"结果 {spec.case.probe} 覆盖几个？", "criteria": []},
            "probe_fresh": {"type": "noul", "instructions": f"结果 {spec.case.probe} 够新吗？", "criteria": {"true": "a", "false": "b"}},
            "conflict": {"type": "noul", "instructions": f"结果 {spec.case.pair[0]} 与结果 {spec.case.pair[1]} 矛盾吗？", "criteria": {"true": "a", "false": "b"}},
            "same_source": {"type": "noul", "instructions": f"结果 {spec.case.pair[0]} 与结果 {spec.case.pair[1]} 同源吗？", "criteria": {"true": "a", "false": "b"}},
            "answerable": {"type": "noul", "instructions": "?", "criteria": {"true": "a", "false": "b"}},
        },
        "answers": {
            "made_up": {"type": "noul", "noul": True},
            "best_result": {"type": "choice", "choice": 0},
            "next_action": {"type": "choice", "choice": 99},
            "relevance": {"type": "score", "score": 99},
            "covered_points": {"type": "score", "score": 0},
            "probe_fresh": {"type": "noul", "noul": True},
            "conflict": {"type": "noul", "noul": True},
            "same_source": {"type": "noul", "noul": False},
            "answerable": {"type": "noul", "noul": True},
        },
    }
    verdict = judges.LabelJudge(max_problems=20).judge(sample)
    assert not verdict.ok
    joined = "; ".join(verdict.raw["problems"])
    assert "未知题族" in joined
    assert "编号越界" in joined
    assert "答案指向" in joined or "答案是" in joined


def test_strip_updated_falls_back_to_the_raw_value() -> None:
    """输入：不带更新标签的日期格。输出：原文。预期行为：认不出标签就原样返回。"""
    labels = parse.merged_labels()
    assert parse._strip_updated("2024-01-01", labels) == "2024-01-01"
    assert parse._strip_updated("更新 2024-01-01", labels) == "2024-01-01"


def test_merged_labels_reports_a_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两种语言共用同一个标签文本。输出：异常。

    预期行为：直接报错——重叠会让语言无关的反解把两种语言的字段串成一行。
    """
    zh = locales.load("zh")
    clashing = json.loads(json.dumps(zh))
    clashing["meta"]["query_label"] = zh["meta"]["query_label"]
    monkeypatch.setattr(locales, "available", lambda: ["zh", "fake"])
    monkeypatch.setattr(locales, "load", lambda lang: zh if lang == "zh" else clashing)
    with pytest.raises(ValueError, match="重复"):
        parse.merged_labels()


def test_build_cases_drops_probes_beyond_the_page() -> None:
    """输入：一个超出页长的问号。输出：无。预期行为：丢掉这个组合，不造出引用不到编号的 case。"""
    config = sp.SearchConfig(
        topics=("refund",),
        best_flavors=("full",),
        mixes=(sp.MIXES[1],),
        layouts=(((1, 2), 3),),
        probes=(1, 99),
        pair_kinds=("none",),
        query_styles=(0,),
    )
    cases = sp.build_cases(config)
    assert {case.probe for case in cases} == {1}


def test_workflow_ood_path_and_unknown_judge_name() -> None:
    """输入：`test_ood` 切分 / 一个不存在的判别器名。输出：case 列表 / 异常。

    预期行为：`test_ood` 走「全取值表 + ood_only」这条分支（不是哈希切片）；
    判别器名写错就报错点名，不静默少跑一个。
    """
    workflow = wf.SearchWorkflow()
    ood = workflow.build_cases(workflow.config_for("test_ood"))
    assert ood and all(case.unseen_dims() for case in ood)
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("made_up",), "zh")
    assert workflow.available_locales() == locales.available()


def test_audit_flags_broken_artifacts_by_name(tmp_path: Path) -> None:
    """输入：三条被改坏的产物行（编号断档 / 未知题族 / 选项越界）。输出：(报告, 缺失项)。

    预期行为：三类问题各自被点名，而不是笼统地「不通过」。
    """
    config = pipeline.RunConfig(
        workflow="search",
        split="train",
        locales=("zh",),
        run_seed=17,
        per_case=1,
        out_root=tmp_path,
        run_id="s-edge",
        config_override=tiny_config("train"),
    )
    result = pipeline.run(config)
    path = result.run_dir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
    # 第一条：把**最后一条**结果的编号推后十位（断档）；顺手塞一个空行，反解要能忽略它。
    # 改最后一条而不是第二条：题面点名的编号若在这一页里找不到，反解会先一步报错，
    # 那条断档检查就走不到了（实测踩过）。
    lines = rows[0]["state"].splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip().startswith("["):
            head, rest = lines[index].strip().split("]", 1)
            lines[index] = f"  [{int(head[1:]) + 10}]{rest}"
            break
    lines.insert(6, "")
    rows[0]["state"] = "\n".join(lines)
    # 第二条：混进一个未知题族
    qid = next(iter(rows[1]["questions"]))
    rows[1]["questions"]["made_up"] = rows[1]["questions"][qid]
    rows[1]["answers"]["made_up"] = rows[1]["answers"][qid]
    # 第三条：choice 答案编号越界
    choice_qid = next(q for q, body in rows[2]["questions"].items() if body["type"] == "choice")
    rows[2]["answers"][choice_qid]["choice"] = 99
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    report, missing = wf.SearchWorkflow().audit(result.run_dir, ("train",), "zh")
    joined = "; ".join(missing)
    assert report["train"]["index_bad"] >= 1
    assert "编号异常" in joined
    assert "未知题族" in joined


def test_audit_flags_a_missing_family(tmp_path: Path) -> None:
    """输入：把某一族题从全部产物行里删掉。输出：(报告, 缺失项)。预期行为：点名缺了哪一族。"""
    config = pipeline.RunConfig(
        workflow="search",
        split="train",
        locales=("zh",),
        run_seed=17,
        per_case=1,
        out_root=tmp_path,
        run_id="s-edge2",
        config_override=tiny_config("train"),
    )
    result = pipeline.run(config)
    path = result.run_dir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
    for row in rows:
        for qid in list(row["questions"]):
            if qid.startswith("same_source"):
                del row["questions"][qid]
                del row["answers"][qid]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    _report, missing = wf.SearchWorkflow().audit(result.run_dir, ("train",), "zh")
    assert any("same_source" in item for item in missing)
