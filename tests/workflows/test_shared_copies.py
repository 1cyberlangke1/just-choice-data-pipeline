"""多家逐字抄同一段断言的用例：收成表 + 参数化用例。

输入：各家测试包里那段逐字相同的断言（只有域名 / 题号 / 字段名不同）。输出：无。
预期行为：与原来各家各写一遍的那些用例等价——脚手架与数据都从各家原处现场取
（`importlib` 按包名导入），这里只留「哪几家 x 哪几个字段」这一份事实。原来散在 42 个
workflow 测试包里的 275 个逐字重复的测试函数收成 72 条用例（312 个参数组合）；各家的
`_run` / `_rows` / `BAD_LABELS` / `CASE` 仍留在原处，不搬家。
"""

from __future__ import annotations

import importlib
import json
import random
import re
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

import dataclasses

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from jc.backend.types import TransportError
from tests.workflows._scaffold import run_config
from jc.jsontypes import JsonValue


def module_of(name: str, dotted: str, *attrs: str) -> ModuleType:
    """输入：workflow 名、测试模块尾名（如 `test_audit`）、这条用例要用的名字。输出：那个模块。

    预期行为：要用的名字一个不缺——缺了当场报错，不静默跳过（跳过等于这条用例白写）。
    其余名字不查：各家测试包覆盖的分支不同，`_flip_choice` 只有带 choice 题的那几家才有。
    """
    module = importlib.import_module(f"tests.workflows.{name}.{dotted}")
    missing = [attr for attr in attrs if not hasattr(module, attr)]
    assert not missing, f"{name}.{dotted} 缺 {missing}"
    return module


def _lang_pairs(names: tuple[str, ...], dotted: str, attr: str) -> tuple[tuple[str, str], ...]:
    """输入：几家 workflow 名、测试模块尾名（如 `test_locales`）、语言表的名字。输出：(家, 语言) 组合。

    预期行为：语言表从各家原处现场取（收集期就取，`parametrize` 等不到函数体）；不另抄一份语言清单。
    """
    return tuple(
        (name, lang)
        for name in names
        for lang in getattr(module_of(name, dotted, attr), attr)
    )


#: 7 家逐字相同的「choice 答案与文本不自洽会被审计抓出来」：家 -> (题号, 计数器) 表。
CHOICE_MISMATCH: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "adblock": (("count", "count_bad"), ("combo", "combo_bad"), ("act", "act_bad")),
    "alertrisk": (("highs", "highs_bad"), ("act", "act_bad")),
    "chatgate": (("count", "count_bad"), ("route", "route_bad"), ("reply", "reply_bad")),
    "depcheck": (("combo", "combo_bad"), ("act", "act_bad")),
    "guardrail": (("count", "count_bad"), ("combo", "combo_bad"), ("action", "action_bad")),
    "logtriage": (("count", "count_bad"), ("combo", "combo_bad"), ("route", "route_bad")),
    "memorygate": (("count", "count_bad"), ("combo", "combo_bad"), ("gate", "gate_bad")),
}

#: 展开成 (家, 题号, 计数器) 三元组，参数化用。
CHOICE_CASES: Final[tuple[tuple[str, str, str], ...]] = tuple(
    (name, qid, counter)
    for name, pairs in CHOICE_MISMATCH.items()
    for qid, counter in pairs
)


@pytest.mark.parametrize(("name", "qid", "counter"), CHOICE_CASES)
def test_choice_answer_mismatch_is_caught(
    tmp_path: Path, name: str, qid: str, counter: str
) -> None:
    """输入：把一条 choice 的答案改成另一个选项。输出：无。

    预期行为：对应的计数器报 1，并在缺失项列表里点名——审计真的会抓「答案与文本不自洽」。
    """
    audit = module_of(
        name, "test_audit", "_run", "_rows", "_flip_choice", "_write", "_audit", "BAD_LABELS"
    )
    run_dir = audit._run(tmp_path, f"t-{qid}")
    rows = audit._rows(run_dir)
    audit._flip_choice(rows[0], qid)
    audit._write(run_dir, rows)
    report, missing = audit._audit(run_dir)
    assert report["train"][counter] == 1
    assert any(audit.BAD_LABELS[counter] in item for item in missing)


#: 6 家逐字相同的「抬头全认不出来的 state 会被审计抓出来」。
STATE_WITHOUT_TABLE: Final[tuple[str, ...]] = (
    "adblock",
    "alertrisk",
    "chatgate",
    "guardrail",
    "posttriage",
    "segmentation",
)


@pytest.mark.parametrize("name", STATE_WITHOUT_TABLE)
def test_state_without_a_structure_table_is_caught(tmp_path: Path, name: str) -> None:
    """输入：一段认不出抬头的 state。输出：无。预期行为：`content_bad` 报 1，并进缺失项。"""
    audit = module_of(name, "test_audit", "_run", "_rows", "_write", "_audit", "BAD_LABELS")
    run_dir = audit._run(tmp_path, "t-content")
    rows = audit._rows(run_dir)
    rows[0]["state"] = "这里什么抬头都没有"
    audit._write(run_dir, rows)
    report, missing = audit._audit(run_dir)
    assert report["train"]["content_bad"] == 1
    assert any(audit.BAD_LABELS["content_bad"] in item for item in missing)


#: 4 家逐字相同的「逐判据 noul 答案取反会被审计抓出来」。
FLAG_MISMATCH: Final[tuple[str, ...]] = ("adblock", "depcheck", "segmentation", "tracescore")


@pytest.mark.parametrize("name", FLAG_MISMATCH)
def test_flag_answer_mismatch_is_caught(tmp_path: Path, name: str) -> None:
    """输入：把一条逐判据 noul 的答案取反。输出：无。预期行为：`flag_bad` 报 1，并进缺失项。"""
    audit = module_of(name, "test_audit", "_run", "_rows", "_write", "_audit", "BAD_LABELS")
    run_dir = audit._run(tmp_path, "t-flag")
    rows = audit._rows(run_dir)
    answer = rows[0]["answers"]["flag"]
    answer["noul"] = not answer["noul"]
    audit._write(run_dir, rows)
    report, missing = audit._audit(run_dir)
    assert report["train"]["flag_bad"] == 1
    assert any(audit.BAD_LABELS["flag_bad"] in item for item in missing)


#: 3 家逐字相同的「反解说反话会被 `E_JUDGE_TRACE` 拦下」。
TRACE_LIES: Final[tuple[str, ...]] = ("modelroute", "toolgate", "verify")


@pytest.mark.parametrize("name", TRACE_LIES)
def test_trace_judge_rejects_a_misread_draft(tmp_path: Path, name: str) -> None:
    """输入：反解时把草稿读反的假后端。输出：无。

    预期行为：一条都留不下，拒绝码是 `E_JUDGE_TRACE`；被拒的行必须带上 state，不然
    「模型写了什么」这条唯一线索就丢了。
    """
    mod = module_of(name, "test_pipeline", "make_config", "CASE_LIMIT")
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    result = pipeline.run(mod.make_config(tmp_path, "s3", conftest.FakeBackend(trace_lies=True)))
    assert result.stats["reject_codes"] == {"E_JUDGE_TRACE": mod.CASE_LIMIT}
    assert result.accepted == 0
    assert all(row["state"] for row in result.rejected)
    assert all("注入" in row["reason"] for row in result.rejected)


@pytest.mark.parametrize("name", TRACE_LIES)
def test_trace_judge_skips_an_endpoint_failure(tmp_path: Path, name: str) -> None:
    """输入：反解时端点故障（重试与降级都试过）的假后端。输出：无。

    预期行为：草稿本身没问题，数据照收；`trace` 记成跳过而不是拒绝——端点抖动不是数据
    质量问题。
    """
    mod = module_of(name, "test_pipeline", "make_config", "CASE_LIMIT")
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    result = pipeline.run(
        mod.make_config(
            tmp_path, "s9", conftest.FakeBackend(trace_raises=TransportError("网关 504"))
        )
    )
    assert result.accepted == mod.CASE_LIMIT
    assert result.rejected == []
    assert result.stats["reject_codes"] == {}
    assert result.stats["judges"]["trace"] == {"ok": 0, "fail": 0, "skipped": mod.CASE_LIMIT}
    assert result.stats["judge_skipped"] == mod.CASE_LIMIT
    skipped = [row for _, row in jsonio.iter_jsonl(result.run_dir / "judge-skipped.jsonl")]
    assert {row["judge"] for row in skipped} == {"trace"}
    assert all("端点故障" in row["reason"] for row in skipped)


#: 3 家「把草稿写坏」的开关 -> 拒绝理由里必须点名的词（开关名与措辞各家不同，逐字抄了 3 遍）。
REFUSED_AT_RENDER: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "modelroute": (
        ("fence_request", "围栏"),
        ("multi_line_request", "多行"),
        ("empty_request", "空的"),
        ("name_request", "模型名"),
        ("label_request", "请求"),
    ),
    "toolgate": (
        ("fence_intent", "围栏"),
        ("multi_line_intent", "多行"),
        ("empty_intent", "空的"),
        ("scope_intent", "状态行"),
        ("label_intent", "段落名"),
    ),
    "verify": (
        ("fence_claim", "围栏"),
        ("multi_line_claim", "多行"),
        ("empty_claim", "空的"),
        ("evidence_claim", "证据"),
        ("label_claim", "声明"),
    ),
}

#: 展开成 (家, 开关, 理由里的词) 三元组，参数化用。
REFUSED_CASES: Final[tuple[tuple[str, str, str], ...]] = tuple(
    (name, switch, reason)
    for name, pairs in REFUSED_AT_RENDER.items()
    for switch, reason in pairs
)


@pytest.mark.parametrize(("name", "switch", "reason"), REFUSED_CASES)
def test_a_broken_draft_is_refused_at_render_time(
    tmp_path: Path, name: str, switch: str, reason: str
) -> None:
    """输入：一个把草稿写坏的假后端。输出：无。

    预期行为：坏法在**渲染期**就被拒（`E_ASSEMBLE`），一条都不留——坏文本拼进 state 就
    洗不掉了，不该等到判别器；理由里点名是哪一种坏法。
    """
    mod = module_of(name, "test_pipeline", "make_config", "CASE_LIMIT")
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    result = pipeline.run(
        mod.make_config(tmp_path, f"s4-{switch}", conftest.FakeBackend(**{switch: True}))
    )
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_ASSEMBLE": mod.CASE_LIMIT}
    assert all(reason in row["reason"] for row in result.rejected)


#: 5 家逐字相同的「提示词里带上『换一种说法』那条规则」。
PROMPT_VARY_RULE: Final[tuple[str, ...]] = (
    "adblock",
    "debugstep",
    "logtriage",
    "memorygate",
    "sqlreview",
)


@pytest.mark.parametrize("name", PROMPT_VARY_RULE)
def test_prompt_carries_the_vary_rule(name: str) -> None:
    """输入：一个 workflow 的两种语言。输出：无。预期行为：提示词里带上 `vary_rule` 那句话。"""
    mod = module_of(name, "test_render", "locales", "render", "CASE")
    for lang in mod.locales.available():
        loc = mod.locales.load(lang)
        assert str(loc["prompt"]["vary_rule"]) in mod.render.build_prompt(loc, mod.CASE)


#: 4 家逐字相同的「`source` 表与 `features.SOURCES` 对得上，文案非空且互不相同」。
SOURCE_TABLE: Final[tuple[str, ...]] = ("alertrisk", "logtriage", "promptinject", "tracescore")


@pytest.mark.parametrize("name", SOURCE_TABLE)
def test_source_table_lines_up_with_features(name: str) -> None:
    """输入：一个 workflow 的两种语言。输出：无。

    预期行为：`source` 表的键就是 `features.SOURCES`，每条文案非空且两两不同——文案抄重了
    等于两档来源在题面里没法区分。
    """
    mod = module_of(name, "test_locales", "locales", "render", "features")
    for lang in mod.locales.available():
        loc = mod.locales.load(lang)
        table = loc["source"]
        assert isinstance(table, dict)
        assert sorted(table) == sorted(mod.features.SOURCES)
        descs = [mod.render.source_copy(loc, source) for source in mod.features.SOURCES]
        assert all(desc.strip() for desc in descs)
        assert len(set(descs)) == len(descs)


#: 4 家逐字相同的「进提示词的那几维每个取值都有 brief 与 hint」。
PROMPT_DIMS: Final[tuple[str, ...]] = ("browser", "findtriage", "search", "sourceselect")


@pytest.mark.parametrize("name", PROMPT_DIMS)
def test_every_prompt_dimension_value_has_text(name: str) -> None:
    """输入：一个 workflow 的两种语言。输出：无。

    预期行为：进提示词的那几维每个取值都有非空的 brief 与 hint——空了模型就只能瞎猜。
    """
    mod = module_of(name, "test_locales", "locales", "render", "features")
    for lang in mod.locales.available():
        loc = mod.locales.load(lang)
        for dim in mod.render.DIM_ORDER:
            for value in mod.features.DIMS[dim]:
                assert str(loc["feature"][dim][value]["brief"]).strip()
                assert str(loc["feature"][dim][value]["hint"]).strip()


#: 4 家逐字相同的「中英两语言定义同一批题族与同一批键」。
SAME_FAMILIES: Final[tuple[str, ...]] = ("cachereuse", "clipcut", "planstep", "sentiment")


@pytest.mark.parametrize("name", SAME_FAMILIES)
def test_languages_define_the_same_families_and_keys(name: str) -> None:
    """输入：一个 workflow 的两种语言。输出：无。

    预期行为：题族集合、每个题族的键集合、每个题族的问法条数都一致——一边改了另一边没改，
    换语言跑就会少一个题族或换一套判据。
    """
    mod = module_of(name, "test_locales", "locales", "questions")
    reference = mod.locales.load("zh")
    other = mod.locales.load("en")
    assert set(other["fam"]) == set(reference["fam"])
    for family in mod.questions.FAMILIES:
        assert set(other["fam"][family]) == set(reference["fam"][family]), family
        assert len(other["fam"][family]["q"]) == len(reference["fam"][family]["q"]), family


#: 4 家逐字相同的「只有 choice 题族允许打乱选项」。
ONLY_CHOICE_SHUFFLED: Final[tuple[str, ...]] = ("cachereuse", "clipcut", "quoteread", "siftrank")


@pytest.mark.parametrize("name", ONLY_CHOICE_SHUFFLED)
def test_only_choice_options_may_be_shuffled(name: str) -> None:
    """输入：一个 workflow 的一个 case。输出：无。

    预期行为：`shuffle` 恰好落在 choice 题族上——给有自然顺序的集合打乱会毁掉答案。
    """
    mod = module_of(name, "test_questions", "questions", "_make_spec", "LOC")
    for item in mod.questions.build(mod._make_spec(2), mod.LOC, 0, random.Random(0)):
        assert item.shuffle is (item.question["type"] == "choice"), item.family
#: 7 家逐字相同的「均衡报告覆盖每个题族、且没有哪一族退化成常量」：家 -> (跑法标签, 题族表)。
BALANCE_FAMILIES: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "cachereuse": ("c4", ("score:reuse_band", "choice:diverge_at", "choice:diff_kind", "noul:reuse_half")),
    "clipcut": ("c4", ("score:keep_count", "choice:first_kept", "noul:blocked_any", "noul:dup_any")),
    "match": (
        "m4",
        (
            "choice:verdict",
            "choice:variant_count",
            "noul:same_name",
            "noul:same_brand",
            "noul:same_size",
            "noul:same_abv",
            "noul:same_pack",
            "score:band",
        ),
    ),
    "planstep": ("p4", ("score:remaining", "choice:next", "score:rounds", "noul:unlocks")),
    "quoteread": ("c4", ("score:tier", "choice:leader", "noul:alert", "noul:divergent")),
    "sentiment": ("s4", ("choice:tone", "choice:pivot", "score:drop", "noul:rebound")),
    "siftrank": ("c4", ("choice:agg_top", "choice:pair_order", "score:cut_size", "noul:stable")),
}


@pytest.mark.parametrize("name", sorted(BALANCE_FAMILIES))
def test_balance_report_covers_every_family_and_has_no_constant(tmp_path: Path, name: str) -> None:
    """输入：一个 workflow 的小规格跑。输出：均衡报告。

    预期行为：表里列的题族一个不少地出现在报告里，且**没有哪一族退化成常量或严重偏斜**
    （`max_dev > 0.2`）——常量答案意味着这一族在教「永远选同一个」，是坏数据而不是难题。
    """
    label, families = BALANCE_FAMILIES[name]
    mod = module_of(name, "test_pipeline", "pipeline", "make_config")
    result = mod.pipeline.run(mod.make_config(tmp_path, label))
    keys = {item["key"] for item in result.stats["balance"]}
    for family in families:
        assert family in keys
    degenerate = [item for item in result.stats["balance"] if item["max_dev"] > 0.2]
    assert not degenerate, f"有题族退化成常量或严重偏斜：{degenerate}"


#: 6 家逐字相同的「抹掉一条结构键那一行会被审计抓出来」：家 -> 被抹掉的抬头。
STRUCTURE_KEY: Final[dict[str, str]] = {
    "alertrisk": "exploit:",
    "debugstep": "Note:",
    "depcheck": "license:",
    "hirecheck": "permit:",
    "sqlreview": "note:",
    "subqa": "Note:",
}


@pytest.mark.parametrize("name", sorted(STRUCTURE_KEY))
def test_state_missing_a_structure_key_is_caught(tmp_path: Path, name: str) -> None:
    """输入：把一条样本的某个结构键那一行抹掉。输出：无。

    预期行为：`content_bad` 报 1——结构键是 state 格式的一部分，少一行那几条钉在行上的判据
    就定位不到，整条样本算不出来。
    """
    prefix = STRUCTURE_KEY[name]
    audit = module_of(name, "test_audit", "_run", "_rows", "_write", "_audit")
    run_dir = audit._run(tmp_path, "t-key")
    rows = audit._rows(run_dir)
    rows[0]["state"] = "\n".join(
        line for line in str(rows[0]["state"]).splitlines() if not line.strip().startswith(prefix)
    )
    audit._write(run_dir, rows)
    report, _ = audit._audit(run_dir)
    assert report["train"]["content_bad"] == 1
#: 10 家逐字相同的「train / val / test_iid 三份产物的 state 两两不相交」：家 -> 跑法标签前缀。
SPLIT_LABELS: Final[dict[str, str]] = {
    "bench": "bm5-",
    "cachereuse": "c6-",
    "clipcut": "c6-",
    "elementpick": "ep5-",
    "functions": "fn6-",
    "match": "m6-",
    "planstep": "p6-",
    "sentiment": "s6-",
    "sim": "sm5-",
    "skillpick": "sp5-",
}


@pytest.mark.parametrize("name", sorted(SPLIT_LABELS))
def test_three_id_splits_are_disjoint(tmp_path: Path, name: str) -> None:
    """输入：一个 workflow 的 train / val / test_iid 三次跑。输出：三份产物。

    预期行为：三份产物的 state 集合两两不相交——切分按**素材**哈希分桶，同一条不会同时
    出现在两档里（`config_for` 忘了替换 `split` 时这条会红）。
    """
    label = SPLIT_LABELS[name]
    seen: list[set[str]] = []
    for split in ("train", "val", "test_iid"):
        result = pipeline.run(pipeline_config(name, tmp_path, f"{label}{split}", split=split))
        rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / f"{split}.jsonl")]
        assert rows
        seen.append({str(row["state"]) for row in rows})
    assert not seen[0] & seen[1]
    assert not seen[0] & seen[2]
    assert not seen[1] & seen[2]
#: 7 家逐字相同的「判据表缺一条会被 FrameJudge 抓出来」：家 -> (被抹掉的编号, 理由里的那个词)。
FRAME_TABLE: Final[dict[str, tuple[str, str]]] = {
    "adblock": ("A2", "判据表"),
    "debugstep": ("D2", "判据表"),
    "logtriage": ("N2", "判据表"),
    "sqlreview": ("Q2", "判据表"),
    "subqa": ("S2", "判据表"),
    "tracescore": ("S2", "信号表"),
    "turntake": ("T2", "判据表"),
}


def test_frame_accepts_faithful_and_rejects_a_broken_table() -> None:
    """输入：忠实 state 与缺了一行的 state。输出：无。

    预期行为：判据表缺一条就报出来——那是构造 bug，标签会跟着一起错。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name in sorted(FRAME_TABLE):
        try:
            mod = module_of(name, "test_judges", "faithful", "CASE", "judges", "_drop_tag_line")
            dropped, word = FRAME_TABLE[name]
            judge = mod.judges.FrameJudge("zh")
            spec, sample = mod.faithful(mod.CASE)
            assert judge.judge(sample, spec).ok
            broken = {"state": mod._drop_tag_line(str(sample["state"]), dropped)}
            verdict = judge.judge(broken, spec)
            assert not verdict.ok
            assert word in verdict.reason
        except AssertionError as exc:
            bad.append(f"{(name,)!r}: {exc}")
    assert not bad, "\n".join(bad)


#: 8 家逐字相同的「反解漏报一条会被 TraceJudge 抓出来」：家 -> 理由里该出现的两条编号。
TRACE_LIE_IDS: Final[dict[str, tuple[str, str]]] = {
    "adblock": ("A1", "A3"),
    "chatgate": ("C3", "C4"),
    "debugstep": ("D1", "D3"),
    "logtriage": ("N1", "N3"),
    "memorygate": ("M1", "M3"),
    "promptinject": ("T1", "T3"),
    "sqlreview": ("Q1", "Q3"),
    "subqa": ("S1", "S3"),
}


def test_trace_catches_a_lie() -> None:
    """输入：反解漏报一条。输出：无。预期行为：不通过，理由里两条都列出来。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name in sorted(TRACE_LIE_IDS):
        try:
            mod = module_of(name, "test_judges", "faithful", "CASE", "judges", "FakeBackend")
            first, second = TRACE_LIE_IDS[name]
            spec, sample = mod.faithful(mod.CASE)
            verdict = mod.judges.TraceJudge(mod.FakeBackend(trace_lies=True), "zh").judge(sample, spec)
            assert not verdict.ok and not verdict.skipped
            assert first in verdict.reason and second in verdict.reason
        except AssertionError as exc:
            bad.append(f"{(name,)!r}: {exc}")
    assert not bad, "\n".join(bad)
#: 8 家逐字相同的「逐判据那一族的 True / False 正好各占一半」：家 -> (第一条判据编号, 每半边的条数)。
FLAG_BALANCE: Final[dict[str, tuple[str, int]]] = {
    "adblock": ("A1", 24),
    "debugstep": ("D1", 24),
    "logtriage": ("N1", 24),
    "memorygate": ("M1", 24),
    "sqlreview": ("Q1", 24),
    "subqa": ("S1", 24),
    "tracescore": ("S1", 24),
    "turntake": ("T1", 24),
}


def test_flag_answers_are_fifty_fifty() -> None:
    """输入：一个 workflow 的全空间 case。输出：无。

    预期行为：每条判据的 True / False 各占一半——这是按构造就成立的均衡，不是调出来的。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name in sorted(FLAG_BALANCE):
        try:
            mod = module_of(name, "test_questions", "locales", "features", "questions")
            first, half = FLAG_BALANCE[name]
            loc = mod.locales.load("zh")
            totals = dict.fromkeys(mod.features.TAG_IDS, 0)
            for case in mod.features.build_cases(512):
                out = mod.questions.build(mod.features.plan_for(case, 0), loc, 0)
                for tag, item in zip(mod.features.TAG_IDS, out, strict=False):
                    if item.answer:
                        totals[tag] += 1
            assert len(set(totals.values())) == 1, totals
            assert totals[first] == half
        except AssertionError as exc:
            bad.append(f"{(name,)!r}: {exc}")
    assert not bad, "\n".join(bad)


#: 8 家逐字相同的「文案表缺字段会被点名」：家 -> (判据编号, 认不出来的那个词, 只填一半的 name)。
TAG_COPY: Final[dict[str, tuple[str, str, str]]] = {
    "adblock": ("A1", "缺判据", "广告位命名"),
    "debugstep": ("D1", "缺判据", "停在异常上"),
    "logtriage": ("N1", "缺判据", "心跳与健康检查"),
    "memorygate": ("M1", "缺判据", "作用域膨胀"),
    "sqlreview": ("Q1", "缺判据", "无条件改整张表"),
    "subqa": ("S1", "缺判据", "原文是音效或占位符"),
    "tracescore": ("S1", "缺信号", "跑过验证"),
    "turntake": ("T1", "缺判据", "话头悬在连接词上"),
}


@pytest.mark.parametrize("name", sorted(TAG_COPY))
def test_tag_copy_reports_missing_fields(name: str) -> None:
    """输入：缺字段的语言字典。输出：无。预期行为：抛 ValueError，点名缺了哪个字段。"""
    mod = module_of(name, "test_render", "render")
    tag, word, label = TAG_COPY[name]
    with pytest.raises(ValueError, match=f"{word} {tag}"):
        mod.render.tag_copy({"tag": {}}, tag)
    broken: dict[str, JsonValue] = {"tag": {tag: {"name": label}}}
    with pytest.raises(ValueError, match="缺字段 rule"):
        mod.render.tag_copy(broken, tag)
#: 8 家逐字相同的「`build_task` 带上注入特征」：家 -> (写进 `case` 的那两个键, 命中编号表)。
BUILD_TASK_SPEC: Final[dict[str, tuple[dict[str, str], list[str]]]] = {
    "adblock": ({"hits": "A1+A3", "surface": "article"}, ["A1", "A3"]),
    "debugstep": ({"hits": "D1+D3", "runtime": "python"}, ["D1", "D3"]),
    "logtriage": ({"hits": "N1+N3", "source": "app"}, ["N1", "N3"]),
    "memorygate": ({"hits": "M1+M3", "form": "sentence"}, ["M1", "M3"]),
    "sqlreview": ({"hits": "Q1+Q3", "workload": "oltp"}, ["Q1", "Q3"]),
    "subqa": ({"hits": "S1+S3", "track": "movie"}, ["S1", "S3"]),
    "tracescore": ({"hits": "S1+S3", "source": "cli"}, ["S1", "S3"]),
    "turntake": ({"hits": "T1+T3", "channel": "im"}, ["T1", "T3"]),
}


def test_build_task_carries_the_injected_features() -> None:
    """输入：真值规格。输出：无。

    预期行为：任务 id 可重算、`spec` 里带上注入特征（假后端与真后端都能照着它答）、响应模型是
    草稿模型。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name in sorted(BUILD_TASK_SPEC):
        try:
            mod = module_of(name, "test_render", "spec_for", "CASE", "render", "locales")
            case, hits = BUILD_TASK_SPEC[name]
            task = mod.render.build_task(mod.spec_for(mod.CASE), mod.locales.load("zh"))
            assert task.task_id == f"{name}-train-0000001-render"
            assert task.domain == name
            assert task.response_model is mod.render.ContentDraft
            assert task.spec["case"] == case
            assert task.spec["hits"] == hits
            extra = next(key for key in case if key != "hits")
            assert task.spec[extra] == case[extra]
        except AssertionError as exc:
            bad.append(f"{(name,)!r}: {exc}")
    assert not bad, "\n".join(bad)
#: 7 家逐字相同的「`build` 出的题族首次出现顺序与 `FAMILIES` 一致」：家 -> 该家声明的四族。
FAMILY_ORDER: Final[dict[str, tuple[str, str, str, str]]] = {
    "adblock": ("flag", "count", "combo", "act"),
    "alertrisk": ("fact", "grade", "highs", "act"),
    "chatgate": ("tag", "count", "route", "reply"),
    "depcheck": ("flag", "grade", "combo", "act"),
    "logtriage": ("flag", "count", "combo", "route"),
    "memorygate": ("flag", "count", "combo", "gate"),
    "tracescore": ("flag", "count", "combo", "route"),
}


def test_family_order_matches_the_declared_families() -> None:
    """输入：无。输出：无。

    预期行为：`build` 出的题族**首次出现顺序**与 `FAMILIES` 一致（逐判据那一族出四道，所以比
    的是去重之后的顺序）。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name in sorted(FAMILY_ORDER):
        try:
            mod = module_of(name, "test_questions", "questions", "features", "locales", "CASE")
            out = mod.questions.build(mod.features.plan_for(mod.CASE, 0), mod.locales.load("zh"), 0)
            assert list(dict.fromkeys(item.family for item in out)) == list(mod.questions.FAMILIES)
            assert mod.questions.FAMILIES == FAMILY_ORDER[name]
        except AssertionError as exc:
            bad.append(f"{(name,)!r}: {exc}")
    assert not bad, "\n".join(bad)


#: 10 家逐字相同的「认不出抬头的 state 会被 HygieneJudge 挡下」：家 -> 理由里该出现的那个词。
HYGIENE_REASON: Final[dict[str, str]] = {
    "adblock": "片段",
    "debugstep": "会话",
    "guardrail": "正文",
    "logtriage": "日志",
    "memorygate": "记忆",
    "promptinject": "内容",
    "sqlreview": "评审记录",
    "subqa": "字幕对",
    "tracescore": "轨迹",
    "turntake": "输入",
}


def test_hygiene_rejects_a_state_without_a_content_block() -> None:
    """输入：一段没有正文抬头的 state。输出：无。预期行为：不通过，理由是认不出来。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name in sorted(HYGIENE_REASON):
        try:
            mod = module_of(name, "test_judges", "judges", "faithful", "CASE")
            spec, _ = mod.faithful(mod.CASE)
            verdict = mod.judges.HygieneJudge("zh").judge({"state": "这里什么抬头都没有"}, spec)
            assert not verdict.ok
            assert HYGIENE_REASON[name] in verdict.reason
        except AssertionError as exc:
            bad.append(f"{(name,)!r}: {exc}")
    assert not bad, "\n".join(bad)
#: 4 家纯构造式 workflow 的整条管线测试逐字相同：家 -> (跑法标签前缀, 均衡报告里的键)。
TINY_PIPELINE: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "bench": ("bm", ("choice:best", "noul:hit", "noul:pass", "score:count", "score:gap")),
    "elementpick": (
        "ep",
        ("choice:role", "choice:target", "noul:found", "noul:shadow", "score:decoy"),
    ),
    "skillpick": (
        "sp",
        ("choice:pick", "choice:scope", "noul:allow", "noul:conflict", "noul:fits"),
    ),
    "sim": ("sm", ("choice:best", "noul:can_move", "noul:finished", "score:cost", "score:gain")),
}


def pipeline_config(
    name: str, tmp_path: Path, run_id: str, *, split: str = "train"
) -> pipeline.RunConfig:
    """输入：workflow 名、临时目录、run 名、切分。输出：指向 tmp 的运行配置。

    预期行为：`TINY_PIPELINE` 里那四家已经收进本文件，没有自己的 `test_pipeline` 了，
    用共享的极小配置；其余各家自己的 `make_config` 还在（那是各包的接线），就用它。
    """
    if name in TINY_PIPELINE:
        return tiny_pipeline_config(name, tmp_path, run_id, split=split)
    module = importlib.import_module(f"tests.workflows.{name}.test_pipeline")
    return module.make_config(tmp_path, run_id, split=split)


def tiny_pipeline_config(
    name: str, tmp_path: Path, run_id: str, *, split: str = "train"
) -> pipeline.RunConfig:
    """输入：workflow 名、临时目录、run 名、切分。输出：指向 tmp 的运行配置（不需要后端）。

    预期行为：与该家 `conftest.tiny_config` 的极小采样参数拼起来——零 token，跑得动整条管线。
    种子与并发是各包的历史取值（17 / 1），显式传，不靠默认值。
    """
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    return run_config(
        tmp_path,
        run_id,
        workflow=name,
        split=split,
        seed=17,
        concurrency=1,
        override=conftest.tiny_config(split),
    )


@pytest.mark.parametrize("name", sorted(TINY_PIPELINE))
def test_every_case_yields_exactly_one_sample(tmp_path: Path, name: str) -> None:
    """输入：一次 train 跑。输出：无。预期行为：样本数 = case 数（per_case=1）。"""
    label, _keys = TINY_PIPELINE[name]
    config = tiny_pipeline_config(name, tmp_path, f"{label}3")
    result = pipeline.run(config)
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    assert len(rows) == len(workflow_mod.load(name).build_cases(config.config_override))
    for row in rows:
        assert row["id"].startswith(f"{name}-train-")
        assert row["questions"] and row["answers"]
        assert set(row["questions"]) == set(row["answers"])


@pytest.mark.parametrize("name", sorted(TINY_PIPELINE))
def test_ood_split_only_carries_unseen_values(tmp_path: Path, name: str) -> None:
    """输入：一次 test_ood 跑。输出：无。预期行为：零拒绝、审计零问题。"""
    label, _keys = TINY_PIPELINE[name]
    config = tiny_pipeline_config(name, tmp_path, f"{label}4", split="test_ood")
    result = pipeline.run(config)
    workflow = workflow_mod.load(name)
    assert result.accepted == len(workflow.build_cases(config.config_override)) > 0
    report, missing = workflow.audit(result.run_dir, ("test_ood",), "zh")
    assert missing == [] and report["test_ood"]["samples"] == result.accepted


@pytest.mark.parametrize("name", sorted(TINY_PIPELINE))
def test_states_are_distinct_within_a_split(tmp_path: Path, name: str) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：同一档里的 state 两两不同——「case 维度不同 ⇒ 文本不同」是构造不变量，
    破了它就会出现两条一模一样的样本。
    """
    label, _keys = TINY_PIPELINE[name]
    result = pipeline.run(tiny_pipeline_config(name, tmp_path, f"{label}6"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    states = [row["state"] for row in rows]
    assert len(set(states)) == len(states)


@pytest.mark.parametrize("name", sorted(TINY_PIPELINE))
def test_answer_balance_is_built_in(tmp_path: Path, name: str) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：表里那五个题族的答案分布全在容差内——均衡是**构造**出来的，不靠事后过滤。
    """
    label, keys = TINY_PIPELINE[name]
    result = pipeline.run(tiny_pipeline_config(name, tmp_path, f"{label}7"))
    offenders = [item for item in result.stats["balance"] if not item["ok"]]
    assert offenders == [], offenders
    assert {item["key"] for item in result.stats["balance"]} == set(keys)


@pytest.mark.parametrize("name", sorted(TINY_PIPELINE))
def test_english_locale_runs_end_to_end(tmp_path: Path, name: str) -> None:
    """输入：英文一次跑。输出：RunResult。预期行为：零拒绝、审计零问题。"""
    label, _keys = TINY_PIPELINE[name]
    config = dataclasses.replace(tiny_pipeline_config(name, tmp_path, f"{label}8"), locales=("en",))
    result = pipeline.run(config)
    workflow = workflow_mod.load(name)
    assert result.accepted == len(workflow.build_cases(config.config_override))
    report, missing = workflow.audit(result.run_dir, ("train",), "en")
    assert missing == [] and report["train"]["samples"] == result.accepted
#: 「`_choice` 当场拒掉不在选项里的答案名」的几家：家 -> (题族, 题面, 选项表, 坏答案名)。
BAD_ANSWER: Final[dict[str, tuple[str, str, dict[str, str], str]]] = {
    "adblock": ("count", "题面", {"c0": "0 条"}, "c9"),
    "alertrisk": ("act", "题面", {"monitor": "记入清单"}, "nope"),
    "depcheck": ("act", "题面", {"approve": "直接进依赖树"}, "nope"),
    "guardrail": ("count", "问", {"c0": "0 条"}, "c9"),
}


@pytest.mark.parametrize("name", sorted(BAD_ANSWER))
def test_bad_answer_names_are_rejected(name: str) -> None:
    """输入：不在选项里的答案名。输出：无。预期行为：抛 ValueError。"""
    mod = module_of(name, "test_questions", "questions")
    family, text, options, bad = BAD_ANSWER[name]
    with pytest.raises(ValueError, match="不在选项里"):
        mod.questions._choice(family, text, options, bad)


#: 「`_choice` 当场拒掉描述重复的选项」的几家：家 -> (题族, 题面, 两个同描述的选项, 答案名)。
DUP_OPTIONS: Final[dict[str, tuple[str, str, dict[str, str], str]]] = {
    "adblock": ("count", "题面", {"c0": "0 条", "c1": "0 条"}, "c0"),
    "alertrisk": ("act", "题面", {"monitor": "记入清单", "ticket": "记入清单"}, "monitor"),
    "depcheck": ("act", "题面", {"approve": "直接进依赖树", "watch": "直接进依赖树"}, "approve"),
    "guardrail": ("count", "问", {"c0": "0 条", "c1": "0 条"}, "c0"),
    "lint": ("count", "题面", {"c0": "一样", "c1": "一样"}, "c0"),
}


@pytest.mark.parametrize("name", sorted(DUP_OPTIONS))
def test_duplicate_option_descriptions_are_rejected(name: str) -> None:
    """输入：两个描述一样的选项。输出：无。预期行为：抛 ValueError。"""
    mod = module_of(name, "test_questions", "questions")
    family, text, options, answer = DUP_OPTIONS[name]
    with pytest.raises(ValueError, match="选项描述重复"):
        mod.questions._choice(family, text, options, answer)
#: 「逐判据那一族的题面自带判据原文」的几家：家 -> 该家语言表里放判据的那个键。
SELF_CONTAINED: Final[dict[str, str]] = {
    "adblock": "tag",
    "alertrisk": "fact",
    "depcheck": "fact",
    "tracescore": "tag",
}

#: 展开成 (家, 语言) 二元组，参数化用。
SELF_CONTAINED_CASES: Final[tuple[tuple[str, str], ...]] = tuple(
    (name, lang) for name in sorted(SELF_CONTAINED) for lang in ("en", "zh")
)


def test_every_question_is_self_contained() -> None:
    """输入：一种语言 × 第二种问法。输出：无。

    预期行为：逐判据那一族的题面里带上要问的那条判据的名字与原文——只读 state 的考生要能
    自己按字面判。（用第二种问法：第一种是短问法，只点名判据，不重述判据原文。）

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name, lang in SELF_CONTAINED_CASES:
        try:
            mod = module_of(name, "test_questions", "locales", "features", "questions", "CASE")
            loc = mod.locales.load(lang)
            built = mod.questions.build(mod.features.plan_for(mod.CASE, 0), loc, 1)
            for tag, item in zip(mod.features.TAG_IDS, built, strict=False):
                entry = loc[SELF_CONTAINED[name]][tag]
                instructions = str(item.question["instructions"])
                assert str(entry["name"]) in instructions
                assert str(entry["rule"]) in instructions
        except AssertionError as exc:
            bad.append(f"{(name, lang)!r}: {exc}")
    assert not bad, "\n".join(bad)


#: 7 家逐字相同的「诚实的反解要过、并把反解结果记进 raw」：家 -> raw 里期望的命中组合。
TRACE_AGREES: Final[dict[str, tuple[str, ...]]] = {
    "adblock": ("A1", "A3"),
    "debugstep": ("D1", "D3"),
    "logtriage": ("N1", "N3"),
    "memorygate": ("M1", "M3"),
    "promptinject": ("T1", "T3"),
    "sqlreview": ("Q1", "Q3"),
    "subqa": ("S1", "S3"),
}


def test_trace_agrees_with_the_plan() -> None:
    """输入：诚实的反解。输出：无。预期行为：通过，raw 里留下反解出来的命中组合。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name in sorted(TRACE_AGREES):
        try:
            mod = module_of(name, "test_judges", "judges", "faithful", "CASE", "FakeBackend")
            spec, sample = mod.faithful(mod.CASE)
            verdict = mod.judges.TraceJudge(mod.FakeBackend(), "zh").judge(sample, spec)
            assert verdict.ok and not verdict.skipped
            assert verdict.raw == {"traced_hits": list(TRACE_AGREES[name])}
        except AssertionError as exc:
            bad.append(f"{(name,)!r}: {exc}")
    assert not bad, "\n".join(bad)


#: 5 家逐字相同的「某一族的题一条都没出会被审计记一笔」：家 -> 拿掉哪一族。
FAMILY_NEVER_SHOWS_UP: Final[dict[str, str]] = {
    "bench": "best",
    "elementpick": "role",
    "functions": "stated_style",
    "sim": "finished",
    "skillpick": "pick",
}


@pytest.mark.parametrize("name", sorted(FAMILY_NEVER_SHOWS_UP))
def test_audit_flags_a_family_that_never_shows_up(tmp_path: Path, name: str) -> None:
    """输入：把某一族的题全删掉。输出：无。预期行为：审计记一笔「一条都没出」。"""
    family = FAMILY_NEVER_SHOWS_UP[name]
    mod = module_of(name, "test_edges", "WF", "_sample_row", "jsonio")
    row = mod._sample_row()
    del row["questions"][family]
    del row["answers"][family]
    mod.jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = mod.WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["families"][family] == 0
    assert any(family in item and "一条都没出" in item for item in missing)


#: 5 家逐字相同的「四条判据、16 组命中组合、顺序稳定」：家 -> 期望的 `TAG_IDS`。
TAG_IDS_LOCK: Final[dict[str, tuple[str, ...]]] = {
    "adblock": ("A1", "A2", "A3", "A4"),
    "debugstep": ("D1", "D2", "D3", "D4"),
    "sqlreview": ("Q1", "Q2", "Q3", "Q4"),
    "subqa": ("S1", "S2", "S3", "S4"),
    "turntake": ("T1", "T2", "T3", "T4"),
}


@pytest.mark.parametrize("name", sorted(TAG_IDS_LOCK))
def test_tag_ids_and_hit_sets_are_complete(name: str) -> None:
    """输入：无。输出：无。预期行为：四条判据、16 组命中组合、顺序稳定。"""
    features = module_of(name, "test_features", "features").features
    ids = TAG_IDS_LOCK[name]
    assert features.TAG_IDS == ids
    assert len(features.HIT_SETS) == 16
    assert features.HIT_SETS[0] == ""
    assert features.HIT_SETS[-1] == "+".join(ids)
    assert len(set(features.HIT_SETS)) == 16


#: 5 家逐字相同的「16 组命中组合、空串在前、全命中在最后」：家 -> (第一条, 全命中)。
HIT_SET_ORDER: Final[dict[str, tuple[str, str]]] = {
    "guardrail": ("P1", "P1+P2+P3+P4"),
    "logtriage": ("N1", "N1+N2+N3+N4"),
    "memorygate": ("M1", "M1+M2+M3+M4"),
    "promptinject": ("T1", "T1+T2+T3+T4"),
    "tracescore": ("S1", "S1+S2+S3+S4"),
}


@pytest.mark.parametrize("name", sorted(HIT_SET_ORDER))
def test_hit_sets_are_all_sixteen_combinations(name: str) -> None:
    """输入：无。输出：无。预期行为：16 组、空串在前、全命中在最后、顺序稳定。"""
    features = module_of(name, "test_features", "features").features
    first, whole = HIT_SET_ORDER[name]
    assert len(features.HIT_SETS) == 16
    assert features.HIT_SETS[0] == ""
    assert features.HIT_SETS[1] == first
    assert features.HIT_SETS[-1] == whole
    assert len(set(features.HIT_SETS)) == 16


#: 4 家逐字相同的「提示词列出必给的判据原文与不许出现的反面写法」：家 -> 不许出现的那两条。
PROMPT_AVOID: Final[dict[str, tuple[str, str]]] = {
    "adblock": ("A2", "A4"),
    "logtriage": ("N2", "N4"),
    "tracescore": ("S2", "S4"),
    "turntake": ("T2", "T4"),
}


@pytest.mark.parametrize("name", sorted(PROMPT_AVOID))
def test_prompt_lists_required_rules_and_forbidden_shapes(name: str) -> None:
    """输入：一个命中两条的 case。输出：无。

    预期行为：要求出现的两条各自印上**判据原文**（那是模型唯一的规格来源，示例片段一律不给，
    见 `build_prompt`）；不许出现的那两条各自带着反面写法——少列一条，模型就可能顺手把那条也
    写上，命中组合当场多出来。
    """
    mod = module_of(name, "test_render", "locales", "render", "CASE")
    loc = mod.locales.load("zh")
    prompt = mod.render.build_prompt(loc, mod.CASE)
    for tag in mod.CASE.tags():
        assert str(mod.render.tag_copy(loc, tag)["rule"]) in prompt
        assert "example" not in mod.render.tag_copy(loc, tag)
    for tag in PROMPT_AVOID[name]:
        assert str(mod.render.tag_copy(loc, tag)["avoid"]) in prompt


#: 3 家逐字相同的「题族改名会被审计记一笔未知 + 原来那一族一条都没出」：家 -> (run_id, 题族)。
UNKNOWN_FAMILY: Final[dict[str, tuple[str, str]]] = {
    "browser": ("a9", "confirm"),
    "compact": ("a5", "kind"),
    "routing": ("a7", "issue"),
}


@pytest.mark.parametrize("name", sorted(UNKNOWN_FAMILY))
def test_audit_flags_an_unknown_family_and_a_missing_one(tmp_path: Path, name: str) -> None:
    """输入：把一个题族改名成不认识的。输出：无。

    预期行为：报出 `unknown_families` 一笔，同时点名原来那一族一条都没出——产物串版本时
    这两条会一起出现。
    """
    run_id, family = UNKNOWN_FAMILY[name]
    mod = module_of(
        name,
        "test_audit",
        "pipeline",
        "make_config",
        "FakeBackend",
        "read_rows",
        "write_rows",
        "audit_mod",
        "CASE_LIMIT",
    )
    result = mod.pipeline.run(mod.make_config(tmp_path, run_id, mod.FakeBackend()))
    rows = mod.read_rows(result.run_dir)
    for row in rows:
        questions_of = row["questions"]
        answers_of = row["answers"]
        assert isinstance(questions_of, dict) and isinstance(answers_of, dict)
        questions_of["bogus"] = questions_of.pop(family)
        answers_of["bogus"] = answers_of.pop(family)
    mod.write_rows(result.run_dir, rows)
    report, missing = mod.audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["unknown_families"]["bogus"] == mod.CASE_LIMIT
    assert any(f"题族 {family}" in item for item in missing)


#: 3 家逐字相同的「题面里不许留占位符」：家 -> 造规格时传的种子实参。
PLACEHOLDER_SPEC: Final[dict[str, int]] = {"cachereuse": 10, "clipcut": 7, "quoteread": 8}


def test_no_question_text_keeps_a_placeholder() -> None:
    """输入：一份语言文件。输出：无。

    预期行为：题面、选项描述、等级描述里都不许留下 `{...}` 占位符——留一个就是一道问不出来
    的题，而契约校验只管形状，管不到这个。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name in sorted(PLACEHOLDER_SPEC):
        for lang in ("zh", "en"):
            try:
                mod = module_of(name, "test_questions", "locales", "questions", "_make_spec", "random")
                loc = mod.locales.load(lang)
                spec = mod._make_spec(PLACEHOLDER_SPEC[name])
                for item in mod.questions.build(spec, loc, 0, mod.random.Random(0)):
                    texts = [item.question["instructions"]]
                    criteria = item.question.get("criteria")
                    if isinstance(criteria, dict):
                        texts.extend(str(value) for value in criteria.values())
                    elif isinstance(criteria, list):
                        texts.extend(str(value) for value in criteria)
                    for text in texts:
                        assert "{" not in text and "}" not in text, f"{lang}/{item.family}: {text}"
            except AssertionError as exc:
                bad.append(f"{(name, lang)!r}: {exc}")
    assert not bad, "\n".join(bad)


#: 4 家逐字相同的「档位表与动作表都要覆盖四个档位、文案两两不同」：家 -> 那一家动作族的键。
LEVEL_TABLES: Final[dict[str, str]] = {
    "chatgate": "route",
    "logtriage": "route",
    "memorygate": "gate",
    "promptinject": "action",
}


def test_threshold_and_route_tables_line_up_with_features() -> None:
    """输入：一种语言。输出：无。预期行为：两张表都覆盖四个档位，动作文案两两不同。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name in sorted(LEVEL_TABLES):
        for lang in ("zh", "en"):
            try:
                mod = module_of(name, "test_locales", "locales", "features")
                loc = mod.locales.load(lang)
                options = loc["fam"][LEVEL_TABLES[name]]["opt"]
                assert isinstance(options, dict)
                assert sorted(options) == sorted(mod.features.HANDLING_LEVELS)
                assert len({str(value) for value in options.values()}) == len(options)
                thresholds = loc["state"]["threshold"]
                assert isinstance(thresholds, dict)
                assert sorted(thresholds) == sorted(mod.features.HANDLING_LEVELS)
                assert len({str(value) for value in thresholds.values()}) == len(thresholds)
            except AssertionError as exc:
                bad.append(f"{(name, lang)!r}: {exc}")
    assert not bad, "\n".join(bad)


#: 4 家逐字相同的「manifest 记下后端名字与 needs、绝不记 key」：家 -> 试跑用的 run_id。
MANIFEST_BACKEND: Final[dict[str, str]] = {
    "modelroute": "s10",
    "shell": "s8",
    "toolgate": "s10",
    "verify": "s10",
}


@pytest.mark.parametrize("name", sorted(MANIFEST_BACKEND))
def test_manifest_records_backend(tmp_path: Path, name: str) -> None:
    """输入：假后端。输出：manifest。

    预期行为：记下后端名字与 needs，绝不记 key——溯源够用，凭据不入库。各家的假后端类在自家
    `conftest.py` 里，直接拿来实例化：这个文件不是那些包的测试模块，用不上它们的夹具。
    """
    mod = module_of(name, "test_pipeline", "pipeline", "make_config", "json")
    backend = importlib.import_module(f"tests.workflows.{name}.conftest").FakeBackend
    result = mod.pipeline.run(mod.make_config(tmp_path, MANIFEST_BACKEND[name], backend()))
    manifest = mod.json.loads((result.run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == {"name": "fake", "starting_mode": ""}
    assert manifest["needs"] == ["text"]
    assert manifest["workflow"] == name
    assert manifest["cases"]


#: 2 家逐字相同的「answer_value 认三种题型、坏形态一律给 None」：家 -> (choice 判据, score 判据)。
ANSWER_VALUE_CRITERIA: Final[dict[str, tuple[dict[str, str], list[str]]]] = {
    "bench": ({"a": "第 1 个", "b": "第 2 个"}, ["答对 0 道", "答对 1 道", "答对 2 道"]),
    "sim": ({"a": "甲", "b": "乙"}, ["甲", "乙", "丙"]),
}


@pytest.mark.parametrize("name", sorted(ANSWER_VALUE_CRITERIA))
def test_answer_value_handles_every_type(name: str) -> None:
    """输入：三种题型的答案 + 各种坏形态。输出：无。

    预期行为：choice 给选项名、score 给档位下标、noul 给布尔；编号越界、类型不对
    （布尔不是编号）、criteria 形状不对都给 None——那是对不上，不是崩。
    """
    mod = module_of(name, "test_judges", "judges")
    choice_criteria, score_criteria = ANSWER_VALUE_CRITERIA[name]

    choice = {"type": "choice", "criteria": choice_criteria}
    assert mod.judges.answer_value(choice, {"choice": 1}) == "b"
    assert mod.judges.answer_value(choice, {"choice": 2}) is None
    assert mod.judges.answer_value(choice, {"choice": True}) is None
    assert mod.judges.answer_value(choice, {"choice": "1"}) is None
    assert mod.judges.answer_value({"type": "choice", "criteria": []}, {"choice": 0}) is None

    score = {"type": "score", "criteria": score_criteria}
    assert mod.judges.answer_value(score, {"score": 2}) == 2
    assert mod.judges.answer_value(score, {"score": 3}) is None
    assert mod.judges.answer_value(score, {"score": False}) is None
    assert mod.judges.answer_value({"type": "score", "criteria": {}}, {"score": 0}) is None

    noul = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    assert mod.judges.answer_value(noul, {"noul": True}) is True
    assert mod.judges.answer_value(noul, {"noul": "true"}) is None
    assert mod.judges.answer_value({"type": "score"}, {}) is None
    assert mod.judges.answer_value({"type": "bogus"}, {}) is None


#: 3 家逐字相同的「语言名单来自语言目录；题型三选；needs 为空（纯构造式）」：家 -> 那个 workflow 实例。
AVAILABLE_LOCALES: Final[tuple[str, ...]] = ("bench", "elementpick", "sim")


@pytest.mark.parametrize("name", AVAILABLE_LOCALES)
def test_available_locales_and_declarations(name: str) -> None:
    """输入：无。输出：无。

    预期行为：语言名单来自语言目录；题型是 choice / noul / score，`needs` 为空（纯构造式）。
    """
    mod = module_of(name, "test_api", "WF", "locales")
    assert mod.WF.available_locales() == mod.locales.available()
    assert mod.WF.kinds == frozenset({"choice", "noul", "score"})
    assert mod.WF.needs == frozenset()
    assert mod.WF.name == name
    assert mod.WF.judges == ("roundtrip", "label")


#: 2 家逐字相同的「反解草稿缺字段要记成跳过、不当成数据不合格」：家 -> 理由里该出现的字段名。
TRACE_MUTE_FIELD: Final[dict[str, str]] = {"citation": "in_doc", "rerank": "relevant"}


@pytest.mark.parametrize("name", sorted(TRACE_MUTE_FIELD))
def test_trace_skips_when_the_draft_has_no_field(name: str) -> None:
    """输入：反解草稿缺字段。输出：无。预期行为：记成「没判成」，不当成数据不合格。"""
    mod = module_of(name, "test_judges", "FakeBackend", "Draft", "faithful", "CASE", "judges")

    class Mute(mod.FakeBackend):
        """只回一个空字典的假后端。"""

        def generate(self, task: object) -> object:
            """输入：任务。输出：空草稿。"""
            return mod.Draft(
                data={}, raw_text="", backend="fake", provider="fake", model="fake", mode="fake"
            )

    spec, sample = mod.faithful(mod.CASE)
    verdict = mod.judges.TraceJudge(Mute(), "zh").judge(sample, spec)
    assert verdict.skipped
    assert TRACE_MUTE_FIELD[name] in verdict.reason


#: 2 家逐字相同的「引文在记录里站不住的那条误报要被丢掉」：家 -> (命中两条, 误报那条, 误报的引文)。
TRACE_DROP_QUOTES: Final[dict[str, tuple[str, str, str, str]]] = {
    "alertrisk": ("R1", "R3", "R2", "exploit: auth=none -- 记录里根本没这一行"),
    "depcheck": ("D1", "D3", "D2", "status: archived -- 记录里根本没这一行"),
}


@pytest.mark.parametrize("name", sorted(TRACE_DROP_QUOTES))
def test_trace_drops_quotes_that_do_not_stand_up(name: str) -> None:
    """输入：反解多报一条，但引文在记录里不存在。输出：无。

    预期行为：那条误报被丢掉，剩下的与计划一致——所以整条通过，理由里点名被丢掉的那条。
    """
    mod = module_of(name, "test_judges", "faithful", "CASE", "quote_for", "judges", "StubBackend")
    hit_a, hit_b, miss, miss_quote = TRACE_DROP_QUOTES[name]
    spec, sample = mod.faithful(mod.CASE)
    data: dict[str, JsonValue] = {
        "tags": [
            {"tag": hit_a, "quote": mod.quote_for(hit_a)},
            {"tag": hit_b, "quote": mod.quote_for(hit_b)},
            {"tag": miss, "quote": miss_quote},
        ],
        "level": spec.labels.level,
    }
    verdict = mod.judges.TraceJudge(mod.StubBackend(data), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw["traced_hits"] == [hit_a, hit_b]
    assert "已按没命中处理" in verdict.reason


#: 2 家逐字相同的「四族问法池非空 + 判据族有真假两行 + count/combo 选项表齐全」：家 -> 判据族的键。
FAMILY_SHAPE: Final[dict[str, str]] = {"adblock": "flag", "promptinject": "tag"}


@pytest.mark.parametrize("name", sorted(FAMILY_SHAPE))
@pytest.mark.parametrize("lang", ("zh", "en"))
def test_every_family_has_questions_and_the_right_shape(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四族的问法池非空；逐判据那一族有 true / false 判据；count 与 combo 的选项表齐全。
    """
    mod = module_of(name, "test_locales", "locales", "questions", "features")
    loc = mod.locales.load(lang)
    for family in mod.questions.FAMILIES:
        spec = loc["fam"][family]
        assert isinstance(spec["q"], list) and spec["q"], f"{family} 没有问法"
    criteria_family = FAMILY_SHAPE[name]
    assert str(loc["fam"][criteria_family]["criteria_true"]).strip()
    assert str(loc["fam"][criteria_family]["criteria_false"]).strip()
    assert sorted(loc["fam"]["count"]["opt"]) == [
        f"c{value}" for value in range(len(mod.features.TAG_IDS) + 1)
    ]
    assert str(loc["fam"]["combo"]["none"]).strip()
    assert str(loc["fam"]["combo"]["joiner"])


#: 3 家逐字相同的「编号越界的产物要被审计记一笔」：家 -> 那条 choice 题的题号。
OUT_OF_RANGE_INDEX: Final[dict[str, str]] = {
    "bench": "best",
    "elementpick": "target",
    "skillpick": "pick",
}


@pytest.mark.parametrize("name", sorted(OUT_OF_RANGE_INDEX))
def test_audit_flags_an_out_of_range_choice_index(tmp_path: Path, name: str) -> None:
    """输入：编号越界的产物。输出：无。预期行为：审计记一笔「编号越界」。"""
    mod = module_of(name, "test_edges", "_sample_row", "jsonio", "WF")
    row = mod._sample_row()
    row["answers"][OUT_OF_RANGE_INDEX[name]]["choice"] = 99
    mod.jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = mod.WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


#: 3 家逐字相同的「编号不是整数也记一笔「编号越界」」：家 -> 那条 choice 题的题号。
NON_INTEGER_INDEX: Final[dict[str, str]] = {
    "bench": "best",
    "elementpick": "role",
    "skillpick": "scope",
}


@pytest.mark.parametrize("name", sorted(NON_INTEGER_INDEX))
def test_audit_flags_a_non_integer_choice_index(tmp_path: Path, name: str) -> None:
    """输入：编号不是整数的产物。输出：无。预期行为：同样记一笔「编号越界」。"""
    mod = module_of(name, "test_edges", "_sample_row", "jsonio", "WF")
    row = mod._sample_row()
    row["answers"][NON_INTEGER_INDEX[name]]["choice"] = True
    mod.jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, _ = mod.WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1


#: 2 家逐字相同的「判据原文整段拼在每道题的问句后面」：家（只按名字取各家的 `_build` 与 `LOC`）。
RULEBOOK: Final[tuple[str, ...]] = ("bench", "sim")


@pytest.mark.parametrize("name", RULEBOOK)
def test_instructions_carry_the_whole_rulebook(name: str) -> None:
    """输入：一份规格出的全部题目。输出：无。

    预期行为：判据原文整段拼在每道题的问句后面（范式 §9.8）——只读 state 与题面的读者
    必须能自己算出来。
    """
    mod = module_of(name, "test_questions", "_build", "LOC")
    _spec, items = mod._build()
    for item in items:
        instructions = item.question["instructions"]
        policy = str(mod.LOC["policy"]["text"])
        assert instructions.endswith(policy)
        assert instructions != policy  # 判据前面真的带了问句
        first = str(mod.LOC["fam"][item.family]["q"][0])
        if "{n}" not in first:  # 带题号的问句由各家自己的那条用例锁
            assert instructions.startswith(first)


#: 4 家逐字相同的「反解草稿缺字段记成跳过」：家 -> 假后端回的残缺 trace 数据。
TRACE_SKIPPED: Final[dict[str, dict[str, str]]] = {
    "browser": {"intent": "fill"},
    "compact": {"kind": "user"},
    "routing": {"issue": "bug"},
    "search": {"query_kind": "informational"},
}


@pytest.mark.parametrize("name", sorted(TRACE_SKIPPED))
def test_trace_marks_a_broken_draft_as_skipped(name: str) -> None:
    """输入：反解草稿缺字段。输出：无。预期行为：记成跳过。"""
    mod = module_of(name, "test_judges", "judges", "sample", "state_of", "spec")
    backend = importlib.import_module(f"tests.workflows.{name}.conftest").FakeBackend(
        trace_data=TRACE_SKIPPED[name]
    )
    verdict = mod.judges.TraceJudge(backend, "zh").judge(mod.sample(mod.state_of()), mod.spec())
    assert verdict.ok and verdict.skipped


#: 2 家逐字相同的「train / val / test_iid 的 tag 组合两两不相交」：家（直接按名字 load）。
ID_SPLITS: Final[tuple[str, ...]] = ("mail", "shell")


@pytest.mark.parametrize("name", ID_SPLITS)
def test_id_splits_share_no_tag_combination(name: str) -> None:
    """输入：无。输出：无。

    预期行为：train / val / test_iid 的 tag 组合两两不相交——切分单位是组合，
    同一个组合不许同时出现在训练与测试里。
    """
    workflow = workflow_mod.load(name)
    seen: dict[tuple[tuple[str, str], ...], str] = {}
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases, f"{split} 一个 case 都没有"
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in seen, f"组合同时出现在 {seen[key]} 与 {split}：{key}"
            seen[key] = split


#: 2 家逐字相同的「四族矛盾 + 未知题族都要被审计报出来」：家 -> (题面造法, 答案造法, 布尔, 四个题族)。
AUDIT_CONTRADICTION: Final[dict[str, tuple[str, str, bool, tuple[str, ...]]]] = {
    "modelroute": ("atlas-xl", "usable", False, ("route", "blocked", "cost", "slack")),
    "verify": ("verified", "nothing_missing", True, ("verdict", "gap", "confidence", "committed")),
}


@pytest.mark.parametrize("name", sorted(AUDIT_CONTRADICTION))
def test_audit_catches_contradictions_and_unknown_families(tmp_path: Path, name: str) -> None:
    """输入：一条四族答案都与 state 矛盾、且带未知题族的产物行。输出：(报告, 缺失项)。

    预期行为：四族矛盾各报一条、未知题族报一条、题族缺失也要报——审计的用处就是
    「文本与标签脱节」当场暴露，而不是靠人肉抽查。
    """
    mod = module_of(name, "test_pipeline", "_questions", "_answers", "_row", "_state", "jsonio")
    state_key, case_key, flag, families = AUDIT_CONTRADICTION[name]
    questions_map = mod._questions(state_key, case_key)
    questions_map["made_up_family"] = {"type": "noul", "instructions": "y"}
    answers = mod._answers(3, flag)
    answers["made_up_family"] = {"type": "noul", "noul": True}
    mod.jsonio.write_jsonl(tmp_path / "train.jsonl", [mod._row(mod._state(), questions_map, answers)])
    _, missing = workflow_mod.load(name).audit(tmp_path, ("train",), "zh")
    for family in families:
        assert any(f"{family} 答案与文本重算的结果矛盾" in item for item in missing)
    assert any("未知题族" in item for item in missing)


#: 2 家逐字相同的「反解提示词把两族原子题连同选项原样再问一遍」：家 -> 那两族。
TRACE_PROMPT: Final[dict[str, tuple[str, str]]] = {
    "compact": ("kind", "fact"),
    "routing": ("issue", "urgency"),
}


@pytest.mark.parametrize("name", sorted(TRACE_PROMPT))
def test_build_trace_prompt_asks_both_families_with_the_shared_options(name: str) -> None:
    """输入：一段 state。输出：无。

    预期行为：反解提示词把两族原子题连同选项原样再问一遍——选项与数据集里的一致，
    否则「答对了」与「标签对了」不是一回事。
    """
    mod = module_of(name, "test_judges", "locales", "judges", "state_of")
    loc = mod.locales.load("zh")
    prompt = mod.judges.build_trace_prompt(mod.state_of(), loc)
    assert mod.state_of() in prompt
    assert mod.judges.TRACE_FAMILIES == TRACE_PROMPT[name]
    for family in mod.judges.TRACE_FAMILIES:
        assert str(loc["fam"][family]["q"][0]) in prompt
        for criteria_name in loc["fam"][family]["criteria"]:
            assert str(criteria_name) in prompt


#: 2 家逐字相同的「选项描述按规范顺序拼、空组合用那句「哪条都不算」」：家 -> 单条组合用的判据。
COMBO_TEXT: Final[dict[str, str]] = {"sqlreview": "Q2", "subqa": "S2"}


@pytest.mark.parametrize("name", sorted(COMBO_TEXT))
def test_combo_text_follows_the_canonical_tag_order(name: str) -> None:
    """输入：空组合、单条组合、以及全空间计划里的真值与干扰组合。输出：无。

    预期行为：选项描述按**传进来的顺序**取判据名拼，空组合用该语言里那句「哪条都不算」；而
    `plan.hits` 与三个干扰组合本来就按 `TAG_IDS` 的规范顺序排好（`features.plan_for` 保证），
    所以一组判据只有一种写法、四个选项描述两两不同（`_choice` 会当场验这一点）。
    """
    mod = module_of(name, "test_questions", "locales", "questions", "features")
    loc = mod.locales.load("zh")
    tag = COMBO_TEXT[name]
    assert mod.questions.combo_text(loc, ()) == str(loc["fam"]["combo"]["none"])
    assert mod.questions.combo_text(loc, (tag,)).count(str(loc["tag"][tag]["name"])) == 1
    for case in mod.features.build_cases(512):
        plan = mod.features.plan_for(case, 0)
        for combo in (plan.hits, *plan.distractors):
            assert combo == tuple(item for item in mod.features.TAG_IDS if item in combo), combo


#: 2 家逐字相同的「引文三件事同时成立才算数」：家 -> (命中那条, 别族那条, 不存在的编号, 只抄形状, 只抄取值)。
QUOTE_HOLDS: Final[dict[str, tuple[str, str, str, str, str]]] = {
    "alertrisk": ("R1", "R2", "R9", "exposure: reachable=internal", "reachable=public"),
    "depcheck": ("D1", "D2", "D9", "license: MIT", "AGPL-3.0-only"),
}


@pytest.mark.parametrize("name", sorted(QUOTE_HOLDS))
def test_quote_holds_needs_the_quote_and_the_whole_record(name: str) -> None:
    """输入：四种引文。输出：无。

    预期行为：引文原样在记录里、引文自己满足判据、整屏记录也满足判据——三件事同时成立才算数；
    只把形状塞进引文（记录里没有那一行）是误报。
    """
    mod = module_of(name, "test_judges", "packet_of", "judges", "quote_for", "json")
    good, other, ghost, shape_only, value_only = QUOTE_HOLDS[name]
    body = mod.packet_of((good,))
    assert mod.judges.quote_holds(good, mod.quote_for(good), body)
    # 引文被装进 JSON 数组字符串（范式 §9.88）
    assert mod.judges.quote_holds(good, mod.json.dumps([mod.quote_for(good)]), body)
    assert not mod.judges.quote_holds(good, shape_only, body)
    assert not mod.judges.quote_holds(other, mod.quote_for(other), body)
    assert not mod.judges.quote_holds(ghost, mod.quote_for(good), body)
    assert not mod.judges.quote_holds(good, value_only, body)  # 只抄取值、不带键名，不算


#: 2 家逐字相同的「端点故障不算数据质量问题」：家（各家的假后端类与 run_id 都一样）。
ENDPOINT_FAILURE: Final[tuple[str, ...]] = ("alertrisk", "hirecheck")


@pytest.mark.parametrize("name", ENDPOINT_FAILURE)
def test_endpoint_failure_is_not_a_data_problem(tmp_path: Path, name: str) -> None:
    """输入：反解时端点故障。输出：RunResult。

    预期行为：样本照样合格（没判成的账单独记），端点抖动不许被写成数据质量问题。
    """
    mod = module_of(name, "test_pipeline", "pipeline", "make_config", "cases_of", "ProviderUnavailable")
    backend = importlib.import_module(f"tests.workflows.{name}.conftest").FakeBackend(
        trace_raises=mod.ProviderUnavailable("端点挂了")
    )
    result = mod.pipeline.run(mod.make_config(tmp_path, "p4", backend))
    assert result.accepted == len(mod.cases_of("train"))
    assert result.rejected == []
    assert result.stats["judges"]["trace"]["fail"] == 0


#: 2 家逐字相同的「noul 族带 true / false 判据、答案是布尔」：家 -> (规格种子, 那个 noul 族)。
NOUL_CRITERIA_ONE: Final[dict[str, tuple[int, str]]] = {
    "cachereuse": (9, "reuse_half"),
    "siftrank": (6, "stable"),
}


@pytest.mark.parametrize("name", sorted(NOUL_CRITERIA_ONE))
def test_noul_criteria_has_true_and_false(name: str) -> None:
    """输入：一条规格。输出：无。

    预期行为：noul 带 `criteria`（官方支持描述，那我们就带上）——真 / 假两行都在，
    答案必须是布尔而不是字符串 `"true"`。
    """
    mod = module_of(name, "test_questions", "questions", "_make_spec", "LOC", "random")
    seed, family = NOUL_CRITERIA_ONE[name]
    item = next(
        item
        for item in mod.questions.build(mod._make_spec(seed), mod.LOC, 0, mod.random.Random(0))
        if item.family == family
    )
    assert set(item.question["criteria"]) == {"true", "false"}
    assert isinstance(item.answer, bool)


#: 2 家逐字相同的「每个 noul 族都带 true / false 判据」：家 -> 规格种子。
NOUL_CRITERIA_ALL: Final[dict[str, int]] = {"clipcut": 6, "quoteread": 7}


@pytest.mark.parametrize("name", sorted(NOUL_CRITERIA_ALL))
def test_noul_criteria_has_true_and_false_in_every_family(name: str) -> None:
    """输入：一条规格。输出：无。

    预期行为：两个 noul 族都带 `criteria`（官方支持描述，那我们就带上）——真 / 假两行都在，
    答案必须是布尔而不是字符串 `"true"`。
    """
    mod = module_of(name, "test_questions", "questions", "_make_spec", "LOC", "random")
    built = mod.questions.build(
        mod._make_spec(NOUL_CRITERIA_ALL[name]), mod.LOC, 0, mod.random.Random(0)
    )
    for item in built:
        if item.question["type"] != "noul":
            continue
        assert set(item.question["criteria"]) == {"true", "false"}, item.family
        assert isinstance(item.answer, bool), item.family


#: 2 家逐字相同的「判据表缺一条要报出来」：家 -> 被删掉的那条判据。
FRAME_ROW_MISSING: Final[dict[str, str]] = {"memorygate": "M2", "promptinject": "T2"}


@pytest.mark.parametrize("name", sorted(FRAME_ROW_MISSING))
def test_frame_judge_flags_a_dropped_criteria_row(name: str) -> None:
    """输入：忠实 state 与缺了一行的 state。输出：无。

    预期行为：判据表缺一条就报出来——那是构造 bug，标签会跟着一起错。
    """
    mod = module_of(name, "test_judges", "judges", "locales", "faithful", "CASE", "_drop_tag_line")
    judge = mod.judges.FrameJudge("zh")
    loc = mod.locales.load("zh")
    spec, sample = mod.faithful(mod.CASE)
    assert judge.judge(sample, spec).ok
    broken = {"state": mod._drop_tag_line(str(sample["state"]), loc, FRAME_ROW_MISSING[name])}
    verdict = judge.judge(broken, spec)
    assert not verdict.ok
    assert "判据表" in verdict.reason


#: 3 家逐字相同的「整族删掉要判不通过」：家 -> 被删掉的那族题号。
LABEL_MISSING_QUESTION: Final[dict[str, str]] = {
    "bench": "best",
    "elementpick": "role",
    "skillpick": "scope",
}


@pytest.mark.parametrize("name", sorted(LABEL_MISSING_QUESTION))
def test_label_judge_flags_a_missing_question(name: str) -> None:
    """输入：把某一族整族删掉。输出：无。预期行为：题号对不上，判不通过。"""
    mod = module_of(name, "test_judges", "_sample", "judges")
    sample, spec = mod._sample()
    gone = LABEL_MISSING_QUESTION[name]
    trimmed = {qid: item for qid, item in sample["questions"].items() if qid != gone}
    verdict = mod.judges.LabelJudge(locale="zh").judge({**sample, "questions": trimmed}, spec)
    assert verdict.ok is False and "题号与规格不符" in verdict.reason


#: 2 家逐字相同的「禁用语清单非空、每条都认得出」：家 -> 把禁用语包进正文的模板。
BANNED_PRESENT: Final[dict[str, str]] = {
    "browser": "这是{phrase}的说法",
    "search": "这是{phrase}的页面",
}


@pytest.mark.parametrize("name", sorted(BANNED_PRESENT))
def test_banned_phrases_are_present_and_usable(name: str) -> None:
    """输入：两语言禁用语清单。输出：无。

    预期行为：每条禁用语都能被 `render.banned_echo` 认出来，而且清单非空——
    空清单等于这道防线不存在。
    """
    mod = module_of(name, "test_locales", "locales", "render")
    for lang in mod.locales.available():
        loc = mod.locales.load(lang)
        phrases = loc["banned_intent_phrases"]
        assert phrases, f"{lang} 没有禁用语"
        for phrase in phrases:
            wrapped = BANNED_PRESENT[name].format(phrase=phrase)
            assert str(phrase) in mod.render.banned_echo(wrapped, loc)


#: 3 家逐字相同的「同一规格两次 build 逐字段一致」：家 -> 规格种子。
BUILD_PURE: Final[dict[str, int]] = {"cachereuse": 11, "clipcut": 9, "quoteread": 10}


@pytest.mark.parametrize("name", sorted(BUILD_PURE))
def test_build_is_a_pure_function(name: str) -> None:
    """输入：同一规格、两个不同 RNG。输出：无。预期行为：题目逐字段一致（不掷骰子）。"""
    mod = module_of(name, "test_questions", "questions", "_make_spec", "LOC", "random")
    spec = mod._make_spec(BUILD_PURE[name])
    first = mod.questions.build(spec, mod.LOC, 0, mod.random.Random(0))
    second = mod.questions.build(spec, mod.LOC, 0, mod.random.Random(999))
    assert first == second


#: 2 家逐字相同的「问法按编号取遍池子」：家 -> (那个族, 规格种子)。
PHRASING_ONE: Final[dict[str, tuple[str, int]]] = {
    "cachereuse": ("reuse_band", 12),
    "siftrank": ("agg_top", 10),
}


@pytest.mark.parametrize("name", sorted(PHRASING_ONE))
def test_phrasing_index_cycles_through_the_pool(name: str) -> None:
    """输入：池子里每个问法编号。输出：无。

    预期行为：问法按编号取，取遍池子里的每一条——同一 case 的多条样本靠它换说法，
    编号给同一句话等于这一维白设。
    """
    mod = module_of(name, "test_questions", "questions", "_make_spec", "LOC", "random")
    family, seed = PHRASING_ONE[name]
    pool = mod.LOC["fam"][family]["q"]
    seen = {
        mod.questions.build(mod._make_spec(seed), mod.LOC, index, mod.random.Random(0))[0]
        .question["instructions"]
        for index in range(len(pool))
    }
    assert len(seen) == len(pool)


#: 2 家逐字相同的「问法按编号取遍池子（档位族）」：家 -> (那个族, 规格种子)。
PHRASING_ALL: Final[dict[str, tuple[str, int]]] = {
    "clipcut": ("keep_count", 10),
    "quoteread": ("tier", 11),
}


@pytest.mark.parametrize("name", sorted(PHRASING_ALL))
def test_phrasing_index_cycles_through_the_score_pool(name: str) -> None:
    """输入：池子里每个问法编号。输出：无。

    预期行为：问法按编号取，取遍池子里的每一条——同一 case 的多条样本靠它换说法，
    编号给同一句话等于这一维白设。
    """
    mod = module_of(name, "test_questions", "questions", "_make_spec", "LOC", "random")
    family, seed = PHRASING_ALL[name]
    pool = mod.LOC["fam"][family]["q"]
    seen = {
        mod.questions.build(mod._make_spec(seed), mod.LOC, index, mod.random.Random(0))[0]
        .question["instructions"]
        for index in range(len(pool))
    }
    assert len(seen) == len(pool)


#: 3 家逐字相同的「抬头认不出来要报错、不静默当空值」：家 -> (规格种子, 抬头原文)。
PARSE_HEADER: Final[dict[str, tuple[int, str]]] = {
    "clipcut": (4, "片段筛选请求号"),
    "quoteread": (4, "行情读数请求号"),
    "siftrank": (7, "排序提示"),
}


@pytest.mark.parametrize("name", sorted(PARSE_HEADER))
def test_parse_rejects_an_unknown_header(name: str) -> None:
    """输入：抬头认不出来的 state。输出：无。预期行为：报「认不出来」，不静默当空值。"""
    mod = module_of(name, "test_render_parse", "locales", "render", "parse", "_make_spec", "pytest")
    seed, header = PARSE_HEADER[name]
    loc = mod.locales.load("zh")
    state = mod.render.build(mod._make_spec(seed), loc).replace(header, "随便什么", 1)
    with mod.pytest.raises(ValueError, match="认不出来"):
        mod.parse.parse_state(state, mod.parse.tables_of(loc))


#: 2 家逐字相同的「禁用语不许撞夹具正文」：家 -> 枚举 case 的上限。
BANNED_FIXTURE: Final[dict[str, int]] = {"compact": 1024, "routing": 2048}


@pytest.mark.parametrize("name", sorted(BANNED_FIXTURE))
def test_banned_phrases_do_not_fire_on_the_fixture_bodies(name: str) -> None:
    """输入：夹具里那几段正文。输出：无。

    预期行为：一条都不许撞禁用语——撞了就说明禁用语写得过宽，真跑时会把本来能用的
    数据在渲染期白拒掉。
    """
    mod = module_of(name, "test_locales", "features", "locales", "render", "body_for")
    cases = mod.features.build_cases(BANNED_FIXTURE[name])
    for lang in mod.locales.available():
        loc = mod.locales.load(lang)
        for tags in cases:
            body = mod.body_for(tags.as_dict())
            assert mod.render.banned_echo(body, loc) == [], body


#: 2 家逐字相同的「criteria 形状不对也算编号越界」：家 -> (那一族题号, 坏 criteria 的键, 它的文本)。
MALFORMED_CRITERIA: Final[dict[str, tuple[str, str, str]]] = {
    "bench": ("count", "n0", "答对 0 道"),
    "sim": ("gain", "g0", "收益 0"),
}


@pytest.mark.parametrize("name", sorted(MALFORMED_CRITERIA))
def test_audit_flags_malformed_criteria_shapes(tmp_path: Path, name: str) -> None:
    """输入：choice 的 criteria 是数组 / score 的 criteria 是对象。输出：无。

    预期行为：形状不对也算「编号越界」——产物坏了要看得见，不能在审计里当成没下标可查。
    """
    mod = module_of(name, "test_edges", "_sample_row", "jsonio", "WF")
    qid, key, text = MALFORMED_CRITERIA[name]
    row = mod._sample_row()
    row["questions"]["best"]["criteria"] = ["甲", "乙"]
    row["questions"][qid]["criteria"] = {key: text}
    mod.jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, _ = mod.WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 2


#: 2 家逐字相同的「题面那段判据与 state 里那段同源」：家。
RULES_BLOCK: Final[tuple[str, ...]] = ("planstep", "sentiment")


@pytest.mark.parametrize("name", RULES_BLOCK)
def test_rules_block_is_the_rendered_lines_without_the_prefix(name: str) -> None:
    """输入：语言字典。输出：无。

    预期行为：题面那段判据与 state 里那段**同源**——去掉行首前缀就是同一串文本。
    """
    mod = module_of(name, "test_questions", "locales", "questions", "render")
    loc = mod.locales.load("zh")
    assert mod.questions.rules_block(loc) == "\n".join(str(line) for line in loc["rules"]["lines"])
    rendered = mod.render.rules_lines(loc)
    assert [line.removeprefix(str(loc["meta"]["rules_prefix"])) for line in rendered] == [
        str(line) for line in loc["rules"]["lines"]
    ]


#: 2 家逐字相同的「最后一档吃掉所有更大的数」：家 -> 那家的档位表。
ACT_LEVEL: Final[dict[str, tuple[str, ...]]] = {
    "sqlreview": ("apply", "explain_first", "dba_review", "reject"),
    "subqa": ("ship", "spot_check", "review", "reject"),
}


@pytest.mark.parametrize("name", sorted(ACT_LEVEL))
def test_act_level_swallows_everything_above_the_last_bucket(name: str) -> None:
    """输入：0~5 条命中。输出：无。预期行为：最后一档吃掉所有更大的数，档位名按 `ACTS` 取。"""
    mod = module_of(name, "test_features", "features")
    acts = ACT_LEVEL[name]
    assert [mod.features.act_level(count) for count in range(6)] == [*acts[:3], acts[3], acts[3], acts[3]]
    assert mod.features.ACTS == acts


#: 2 家逐字相同的「每道题的题面都以判据原文结尾」：家 -> 规格种子。
INSTRUCTIONS_END: Final[dict[str, int]] = {"clipcut": 8, "quoteread": 9}


@pytest.mark.parametrize("name", sorted(INSTRUCTIONS_END))
def test_instructions_end_with_the_rules_block(name: str) -> None:
    """输入：一条规格。输出：无。

    预期行为：每道题的题面都以判据原文结尾——判据必须随题面一起给出去（范式 §9.8），
    藏在判别器里等于只读 state 的读者算不出来。
    """
    mod = module_of(name, "test_questions", "questions", "_make_spec", "LOC", "random")
    rules = mod.questions.rules_block(mod.LOC)
    assert rules
    for item in mod.questions.build(
        mod._make_spec(INSTRUCTIONS_END[name]), mod.LOC, 0, mod.random.Random(0)
    ):
        assert item.question["instructions"].endswith(rules), item.family


#: 2 家逐字相同的「反解说反话的样本要被拒」：家 -> 试跑用的 run_id。
MISREAD_KIND: Final[dict[str, str]] = {"compact": "c3", "search": "s3"}


@pytest.mark.parametrize("name", sorted(MISREAD_KIND))
def test_trace_judge_rejects_a_misread_kind(tmp_path: Path, name: str) -> None:
    """输入：反解说反话的假后端。输出：RunResult。

    预期行为：这些样本被拒，拒绝码是 `E_JUDGE_TRACE`——判别器抓的是模型真会犯的错。
    """
    mod = module_of(name, "test_pipeline", "pipeline", "make_config")
    backend = importlib.import_module(f"tests.workflows.{name}.conftest").FakeBackend(trace_lies_kind=True)
    result = mod.pipeline.run(mod.make_config(tmp_path, MISREAD_KIND[name], backend))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


#: 2 家逐字相同的「noul 两行、score 是有序阶梯」：家 -> 那个 noul 族。
NOUL_SCORE_SHAPES: Final[dict[str, str]] = {"compact": "pin_overrode", "routing": "needs_human"}


@pytest.mark.parametrize("name", sorted(NOUL_SCORE_SHAPES))
def test_noul_and_score_criteria_shapes(name: str) -> None:
    """输入：一组特征。输出：无。

    预期行为：noul 的选项是 `true` / `false`；score 的选项是有序阶梯（档数 == `PRIORITY_LEVELS`）。
    """
    mod = module_of(name, "test_questions", "locales", "by_family", "features")
    loc = mod.locales.load("zh")
    families = mod.by_family()
    assert list(families[NOUL_SCORE_SHAPES[name]].question["criteria"]) == ["true", "false"]
    assert list(families["priority"].question["criteria"]) == list(loc["fam"]["priority"]["levels"])
    assert len(families["priority"].question["criteria"]) == mod.features.PRIORITY_LEVELS


#: 2 家逐字相同的「答案改坏 / 规格类型不对都要判不通过」：家 -> 被改坏的那族题号。
WRONG_ANSWER: Final[dict[str, str]] = {"elementpick": "found", "skillpick": "fits"}


@pytest.mark.parametrize("name", sorted(WRONG_ANSWER))
def test_label_judge_flags_a_wrong_answer_and_a_wrong_spec(name: str) -> None:
    """输入：答案被改坏 / 规格类型不对。输出：无。预期行为：都判不通过并说清原因。"""
    mod = module_of(name, "test_judges", "_sample", "judges")
    qid = WRONG_ANSWER[name]
    sample, spec = mod._sample()
    answers = dict(sample["answers"])
    answers[qid] = {"type": "noul", "noul": not sample["answers"][qid]["noul"]}
    verdict = mod.judges.LabelJudge(locale="zh").judge({**sample, "answers": answers}, spec)
    assert verdict.ok is False and qid in verdict.reason

    verdict = mod.judges.LabelJudge(locale="zh").judge(sample, object())
    assert verdict.ok is False and "规格类型不对" in verdict.reason


#: 2 家逐字相同的「test_ood 切分能跑通、零拒绝」：家 -> 试跑用的 run_id。
OOD_SPLIT: Final[dict[str, str]] = {"functions": "fn5", "match": "m5"}


@pytest.mark.parametrize("name", sorted(OOD_SPLIT))
def test_ood_split_runs(tmp_path: Path, name: str) -> None:
    """输入：`test_ood` 切分的极小配置。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上。
    """
    mod = module_of(name, "test_pipeline", "pipeline", "make_config", "_case_count", "jsonio")
    config = mod.make_config(tmp_path, OOD_SPLIT[name], split="test_ood")
    result = mod.pipeline.run(config)
    assert result.accepted == mod._case_count(config) > 0
    assert result.rejected == []
    rows = [row for _, row in mod.jsonio.iter_jsonl(result.run_dir / "test_ood.jsonl")]
    assert len(rows) == result.accepted


#: 2 家逐字相同的「state 反解不出来要报出来」：家。
UNPARSABLE_STATE: Final[tuple[str, ...]] = ("cachereuse", "siftrank")


@pytest.mark.parametrize("name", UNPARSABLE_STATE)
def test_audit_flags_an_unparsable_state(tmp_path: Path, name: str) -> None:
    """输入：state 认不出的那条产物。输出：无。预期行为：报「反解不出来」并计入不一致。"""
    mod = module_of(name, "test_edges", "tiny_config", "build_row", "WORKFLOW", "jsonio")
    config = mod.tiny_config("train")
    rows = [
        mod.build_row(config, case, index)
        for index, case in enumerate(mod.WORKFLOW.build_cases(config))
    ]
    rows[0]["state"] = "乱七八糟\n"
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    mod.jsonio.write_jsonl(run_dir / "train.jsonl", rows)
    report, missing = mod.WORKFLOW.audit(run_dir, ("train",), "zh")
    assert any("反解不出来" in item for item in missing)
    assert report["train"]["label_mismatch"] >= 1


# ================================================================================================
# 以下由跨文件重复组收成（生成脚本：tmp/_merge_a.py）
# ================================================================================================


#: 3 家同款——输入：一种语言。输出：无。
TEST_TAG_TABLE_LINES_UP_WITH_FEATURES: Final[tuple[str, ...]] = ('adblock', 'logtriage', 'tracescore')


@pytest.mark.parametrize("name", TEST_TAG_TABLE_LINES_UP_WITH_FEATURES)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_tag_table_lines_up_with_features(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：判据表的键与 `TAG_IDS` 一一对应，每条三个字段都非空、判据名与表行两两不同，
    而且表行里**编号排在分隔符前面**（解析就是按这个切回编号的）。
    """
    m = module_of(name, 'test_locales', 'features', 'locales', 'render')
    features, locales, render = m.features, m.locales, m.render
    loc = locales.load(lang)
    table = loc["tag"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.TAG_IDS)
    for tag in features.TAG_IDS:
        entry = render.tag_copy(loc, tag)
        for key in ("name", "rule", "avoid"):
            assert str(entry[key]).strip(), f"{lang}.{tag}.{key}"
    names = [str(table[tag]["name"]) for tag in features.TAG_IDS]
    assert len(set(names)) == len(names)
    sep = str(loc["state"]["tag_sep"])
    lines = [render.tag_line(loc, tag) for tag in features.TAG_IDS]
    assert len(set(lines)) == len(lines)
    for tag, line in zip(features.TAG_IDS, lines, strict=True):
        assert line.partition(sep)[0] == tag


#: 4 家同款——输入：全空间的 48 组 case。输出：无。
TEST_TWO_LABEL_PATHS_AGREE_ON_EVERY_FAITHFUL_TEXT: Final[tuple[str, ...]] = ('logtriage', 'memorygate', 'promptinject', 'tracescore')


@pytest.mark.parametrize("name", TEST_TWO_LABEL_PATHS_AGREE_ON_EVERY_FAITHFUL_TEXT)
def test_two_label_paths_agree_on_every_faithful_text(name: str) -> None:
    """输入：全空间的 48 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——
    日志兑现了计划，两条路就不该打架。
    """
    m = module_of(name, 'test_features', 'features', 'text_of')
    features, text_of = m.features, m.text_of
    cases = features.build_cases(512)
    assert len(cases) == 48  # 16 组命中 × 3 种通道
    for case in cases:
        text = text_of(case)
        assert features.hits_of(text) == case.hits, case.describe()
        assert features.observed_labels(text).hits == features.label_for(
            features.plan_for(case, 0)
        ).hits


#: 5 家同款——输入：整段贴 / 改了一个词的两种轨迹。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_PASTED_RULES_ONLY_FLAGS_VERBATIM_TEXT_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'adblock': ('A4', '隐形或极小', '隐形极小'),
    'chatgate': ('C1', '行不行', '行不行呀'),
    'logtriage': ('N1', 'healthz', 'healthzzz'),
    'tracescore': ('S1', 'pytest', 'pytestx'),
    'turntake': ('T4', '等一下', '等两下'),
}


@pytest.mark.parametrize("name", TEST_PASTED_RULES_ONLY_FLAGS_VERBATIM_TEXT_VALUES)
def test_pasted_rules_only_flags_verbatim_text(name: str) -> None:
    """输入：整段贴 / 改了一个词的两种轨迹。输出：无。

    预期行为：只认**逐字相同**的整段判据原文——改一个词就不再算，否则正常产物会被误伤。
    """
    _v0, _v1, _v2 = TEST_PASTED_RULES_ONLY_FLAGS_VERBATIM_TEXT_VALUES[name]
    m = module_of(name, 'test_render', 'locales', 'BASE', 'render')
    locales, BASE, render = m.locales, m.BASE, m.render
    loc = locales.load("zh")
    rule = str(render.tag_copy(loc, _v0)["rule"])
    assert render.pasted_rules(rule, loc) == [rule]
    # 换掉的是**夹在中间**的那个词：把末尾的词加长反而仍包含原串（子串匹配不要求到结尾），
    # 那样这条断言就测了个寂寞。
    assert render.pasted_rules(rule.replace(_v1, _v2), loc) == []
    assert render.pasted_rules(BASE, loc) == []


#: 2 家同款——输入：把一条**计划是「公网可达」**的样本的 `exposure:` 那一行换成内网。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_A_REWRITTEN_LINE_IS_CAUGHT_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'alertrisk': ('R1', 'exposure: reachable=internal -- 只有内网能访问', 'exposure:', 'fact_bad', 'highs_bad'),
    'depcheck': ('D1', 'license: MIT', 'license:', 'flag_bad', 'combo_bad'),
}


@pytest.mark.parametrize("name", TEST_A_REWRITTEN_LINE_IS_CAUGHT_VALUES)
def test_a_rewritten_line_is_caught(name: str, tmp_path: Path) -> None:
    """输入：把一条**计划是「公网可达」**的样本的 `exposure:` 那一行换成内网。输出：无。

    预期行为：`fact_bad` / `grade_bad` / `highs_bad` / `act_bad` 里至少有一个报出来——审计走的是
    「从文本重推标签」这条独立路径，把形状擦掉之后重推出来的命中组合就与题面答案对不上了。
    """
    _v0, _v1, _v2, _v3, _v4 = TEST_A_REWRITTEN_LINE_IS_CAUGHT_VALUES[name]
    m = module_of(name, 'test_audit', '_run', '_rows', '_write', '_audit', 'features', 'render')
    _run, _rows, _write, _audit, features, render = m._run, m._rows, m._write, m._audit, m.features, m.render
    run_dir = _run(tmp_path, "t-rewrite")
    rows = _rows(run_dir)
    target = next(
        row
        for row in rows
        if _v0 in features.hit_set(render.parse_state(row["state"]).text or "")
    )
    lines = [
        _v1
        if line.strip().startswith(_v2)
        else line
        for line in str(target["state"]).splitlines()
    ]
    target["state"] = "\n".join(lines)
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    counts = report["train"]
    assert counts[_v3] + counts["grade_bad"] + counts[_v4] + counts["act_bad"] > 0


#: 2 家同款——输入：六种形态不对的反解草稿。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_TRACE_SKIPS_DRAFTS_IT_CANNOT_READ_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'alertrisk': ('R9', '不认识的因子编号'),
    'depcheck': ('D9', '不认识的标记编号'),
}


@pytest.mark.parametrize("name", TEST_TRACE_SKIPS_DRAFTS_IT_CANNOT_READ_VALUES)
def test_trace_skips_drafts_it_cannot_read(name: str) -> None:
    """输入：六种形态不对的反解草稿。输出：无。

    预期行为：都记「没判成」（`skipped=True`）——那是反解格式没对齐，不是数据本身错。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    _v0, _v1 = TEST_TRACE_SKIPS_DRAFTS_IT_CANNOT_READ_VALUES[name]
    m = module_of(name, 'test_judges', 'faithful', 'CASE', 'judges', 'StubBackend')
    faithful, CASE, judges, StubBackend = m.faithful, m.CASE, m.judges, m.StubBackend
    bad: list[str] = []
    for data, needle in [
        ({}, "缺 tags"),
        ({"tags": "不是列表", "level": 3}, "缺 tags"),
        ({"tags": ["不是对象"], "level": 3}, "不是「编号 + 引文」对象"),
        ({"tags": [{"tag": _v0, "quote": "x"}], "level": 3}, _v1),
        ({"tags": [], "level": 9}, "1~5 的整数"),
        ({"tags": [], "level": "三"}, "1~5 的整数"),
    ]:
        try:
            spec, sample = faithful(CASE)
            verdict = judges.TraceJudge(StubBackend(data), "zh").judge(sample, spec)
            assert verdict.skipped and verdict.ok
            assert needle in verdict.reason
        except AssertionError as exc:
            bad.append(f"{(data, needle)!r}: {exc}")
    assert not bad, "\n".join(bad)


#: 2 家同款——输入：一种语言。输出：无。
TEST_FACT_TABLE_LINES_UP_WITH_FEATURES: Final[tuple[str, ...]] = ('alertrisk', 'depcheck')


@pytest.mark.parametrize("name", TEST_FACT_TABLE_LINES_UP_WITH_FEATURES)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_fact_table_lines_up_with_features(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：因子表的键与 `TAG_IDS` 一一对应，每条三个字段都非空、因子名与表行两两不同，
    而且表行里**编号排在分隔符前面**（解析就是按这个切回编号的）。
    """
    m = module_of(name, 'test_locales', 'features', 'locales', 'render')
    features, locales, render = m.features, m.locales, m.render
    loc = locales.load(lang)
    table = loc["fact"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.TAG_IDS)
    for tag in features.TAG_IDS:
        entry = render.fact_copy(loc, tag)
        for key in ("name", "rule", "avoid"):
            assert str(entry[key]).strip(), f"{lang}.{tag}.{key}"
    names = [str(table[tag]["name"]) for tag in features.TAG_IDS]
    assert len(set(names)) == len(names)
    sep = str(loc["state"]["tag_sep"])
    lines = [render.fact_line(loc, tag) for tag in features.TAG_IDS]
    assert len(set(lines)) == len(lines)
    for tag, line in zip(features.TAG_IDS, lines, strict=True):
        assert line.partition(sep)[0] == tag


#: 3 家同款——输入：一份语言文件。输出：无。
TEST_LABELS_ARE_NOT_MUTUAL_PREFIXES: Final[tuple[str, ...]] = ('clipcut', 'planstep', 'quoteread')


@pytest.mark.parametrize(("name", "lang"), _lang_pairs(TEST_LABELS_ARE_NOT_MUTUAL_PREFIXES, "test_locales", "LANGS"))
def test_labels_are_not_mutual_prefixes(name: str, lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：抬头与区块抬头两两**不互为前缀**——反解靠最长前缀认标签，互为前缀时短的
    会把长的吃掉（「片段筛选请求」被认成「片段筛选请求号」）。
    """
    m = module_of(name, 'test_locales', 'LANGS', 'parse', 'locales')
    LANGS, parse, locales = m.LANGS, m.parse, m.locales
    tables = parse.tables_of(locales.load(lang))
    labels = [*tables.request, *tables.sections]
    for left in labels:
        for right in labels:
            if left != right:
                assert not right.startswith(left), f"{lang}: {left!r} / {right!r}"


#: 3 家同款——输入：几种引文。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_QUOTE_HOLDS_NEEDS_BOTH_THE_QUOTE_AND_THE_CRITERION_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'logtriage': ('N1', 'N1', 'N1', 'N1', 'N1', 'N1', 'N1', 'N1', 'N4', 'N1', '事务已回滚', 'N9', '事务已回滚'),
    'memorygate': ('M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M4', 'M1', '项目组', 'M9', '项目组'),
    'tracescore': ('S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S4', 'S1', '收工。', 'S9', '收工。'),
}


@pytest.mark.parametrize("name", TEST_QUOTE_HOLDS_NEEDS_BOTH_THE_QUOTE_AND_THE_CRITERION_VALUES)
def test_quote_holds_needs_both_the_quote_and_the_criterion(name: str) -> None:
    """输入：几种引文。输出：无。

    预期行为：引文得原样在日志里、且自己满足那条判据；空白差异不算抄错。
    """
    _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8, _v9, _v10, _v11, _v12 = TEST_QUOTE_HOLDS_NEEDS_BOTH_THE_QUOTE_AND_THE_CRITERION_VALUES[name]
    m = module_of(name, 'test_judges', 'BASE', 'FRAGMENTS', 'judges', 'quote_for')
    BASE, FRAGMENTS, judges, quote_for = m.BASE, m.FRAGMENTS, m.judges, m.quote_for
    body = BASE + FRAGMENTS[_v0]
    assert judges.quote_holds(_v1, quote_for(_v2), body)
    # 引文首尾多带了换行：折叠之后一致，照样算抄对了（模型常这么答）。
    assert judges.quote_holds(_v3, f"\n{quote_for(_v4)}\n", body)
    assert judges.quote_holds(_v5, json.dumps([quote_for(_v6)]), body)  # 引文被装进 JSON 数组字符串（范式 §9.88）
    assert not judges.quote_holds(_v7, quote_for(_v8), body)  # 抄不出原文
    assert not judges.quote_holds(_v9, _v10, body)  # 抄出来了但不合判据
    assert not judges.quote_holds(_v11, _v12, body)  # 不认识的编号


#: 2 家同款——输入：一份语言文件。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_META_HAS_THE_EIGHT_STRUCTURAL_KEYS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'planstep': ('tasks', 'rules', 'cell_sep', 'deps_sep', 'deps_prefix', 'deps_none'),
    'sentiment': ('topic', 'words', 'turns', 'rules', 'cell_sep', 'word_sep'),
}


@pytest.mark.parametrize(("name", "lang"), _lang_pairs(TEST_META_HAS_THE_EIGHT_STRUCTURAL_KEYS_VALUES, "test_locales", "LANGS"))
def test_meta_has_the_eight_structural_keys(name: str, lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`[meta]` 里抬头 / 两个区块抬头 / 格分隔符 / 前置分隔符 / 前置前缀 / 无前置
    写法 / 规则行前缀齐全——渲染要按它们拼行，缺一个就直接 KeyError。
    """
    _v0, _v1, _v2, _v3, _v4, _v5 = TEST_META_HAS_THE_EIGHT_STRUCTURAL_KEYS_VALUES[name]
    m = module_of(name, 'test_locales', 'LANGS', 'locales')
    LANGS, locales = m.LANGS, m.locales
    meta = locales.load(lang)["meta"]
    assert isinstance(meta, dict)
    assert set(meta) == {
        "request",
        _v0,
        _v1,
        _v2,
        _v3,
        _v4,
        _v5,
        "rules_prefix",
    }


#: 2 家同款——输入：同一屏记录、两套句面。输出：无。
TEST_PHRASING_ONLY_CHANGES_THE_WORDING: Final[tuple[str, ...]] = ('bench', 'sim')


@pytest.mark.parametrize("name", TEST_PHRASING_ONLY_CHANGES_THE_WORDING)
def test_phrasing_only_changes_the_wording(name: str) -> None:
    """输入：同一屏记录、两套句面。输出：无。

    预期行为：题族、答案、选项**完全一样**，只有问句措辞不同——句面是留出维，但留出它
    不会把任何一族的分布压歪。
    """
    m = module_of(name, 'test_questions', '_make_spec', 'LOC', 'questions')
    _make_spec, LOC, questions = m._make_spec, m.LOC, m.questions
    spec = _make_spec(layout="l2")
    first = questions.build(spec.rows, spec, LOC, 0)
    second = questions.build(spec.rows, spec, LOC, 1)
    assert [item.family for item in first] == [item.family for item in second]
    assert [item.answer for item in first] == [item.answer for item in second]
    assert [item.question["criteria"] for item in first] == [
        item.question["criteria"] for item in second
    ]
    assert any(
        left.question["instructions"] != right.question["instructions"]
        for left, right in zip(first, second, strict=True)
    )


#: 3 家同款——输入：多写一条不该命中的片段。输出：RunResult。
TEST_ADDING_A_SPARE_FRAGMENT_ONLY_SPARES_THE_FULL_HITS: Final[tuple[str, ...]] = ('adblock', 'chatgate', 'promptinject')


@pytest.mark.parametrize("name", TEST_ADDING_A_SPARE_FRAGMENT_ONLY_SPARES_THE_FULL_HITS)
def test_adding_a_spare_fragment_only_spares_the_full_hits(name: str, tmp_path: Path) -> None:
    """输入：多写一条不该命中的片段。输出：RunResult。

    预期行为：四条全中的 case 没有「多余的一条」可加，其余全被拦下。
    """
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    fake_backend = conftest.FakeBackend
    m = module_of(name, 'test_pipeline', 'FakeBackend', 'make_config', 'cases_of', 'features')
    FakeBackend, make_config, cases_of, features = m.FakeBackend, m.make_config, m.cases_of, m.features
    unaffected = sum(
        1 for case in cases_of("train") if len(case.tags()) == len(features.TAG_IDS)
    )
    result = pipeline.run(make_config(tmp_path, "p3-extra", fake_backend(extra_fragment=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


#: 3 家同款——输入：少写一条该命中的形状。输出：RunResult。
TEST_DROPPING_A_FRAGMENT_ONLY_SPARES_THE_CASES_WITHOUT_ONE: Final[tuple[str, ...]] = ('adblock', 'chatgate', 'promptinject')


@pytest.mark.parametrize("name", TEST_DROPPING_A_FRAGMENT_ONLY_SPARES_THE_CASES_WITHOUT_ONE)
def test_dropping_a_fragment_only_spares_the_cases_without_one(name: str, tmp_path: Path) -> None:
    """输入：少写一条该命中的形状。输出：RunResult。

    预期行为：只有「本来就一条都不命中」的那些 case 能活下来（没有形状可少写）。
    """
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    fake_backend = conftest.FakeBackend
    m = module_of(name, 'test_pipeline', 'FakeBackend', 'make_config', 'cases_of')
    FakeBackend, make_config, cases_of = m.FakeBackend, m.make_config, m.cases_of
    unaffected = sum(1 for case in cases_of("train") if not case.tags())
    result = pipeline.run(make_config(tmp_path, "p3-drop", fake_backend(drop_fragment=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


#: 3 家同款——输入：无。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_SOURCES_ARE_THREE_DISTINCT_SHAPES_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'logtriage': ('app', 'infra', 'access'),
    'promptinject': ('web', 'tool', 'file'),
    'tracescore': ('cli', 'editor', 'chat'),
}


@pytest.mark.parametrize("name", TEST_SOURCES_ARE_THREE_DISTINCT_SHAPES_VALUES)
def test_sources_are_three_distinct_shapes(name: str) -> None:
    """输入：无。输出：无。

    预期行为：三个通道键两两不同，且**全空间里每种通道都被枚举到**——留出维只砍取值，
    不该把别的取值一起砍掉（`build_cases` 默认维度表覆盖 `SOURCES`）。
    """
    _v0, _v1, _v2 = TEST_SOURCES_ARE_THREE_DISTINCT_SHAPES_VALUES[name]
    m = module_of(name, 'test_features', 'features')
    features = m.features
    assert features.SOURCES == (_v0, _v1, _v2)
    assert len(set(features.SOURCES)) == len(features.SOURCES)
    assert features.DIMS["source"] == features.SOURCES
    cases = features.build_cases(512)
    assert {case.source for case in cases} == set(features.SOURCES)


#: 3 家同款——输入：两种语言 × 一段 state。输出：无。
TEST_STATE_PRINTS_EACH_CRITERION_RULE: Final[tuple[str, ...]] = ('logtriage', 'memorygate', 'tracescore')


@pytest.mark.parametrize("name", TEST_STATE_PRINTS_EACH_CRITERION_RULE)
def test_state_prints_each_criterion_rule(name: str) -> None:
    """输入：两种语言 × 一段 state。输出：无。

    预期行为：判据表每行都带上该判据的**原文**——只印判据名时，反解模型认不出「已自愈的重试与
    降级」指的是什么形状（范式 §9.44，在 `memorygate` 上量过 1/8 → 8/8）。
    """
    m = module_of(name, 'test_render', 'locales', 'features', 'render', 'text_of', 'CASE')
    locales, features, render, text_of, CASE = m.locales, m.features, m.render, m.text_of, m.CASE
    for lang in locales.available():
        loc = locales.load(lang)
        state = render.to_state(render.ContentDraft(text=text_of(CASE)), loc)
        for tag in features.TAG_IDS:
            assert str(render.tag_copy(loc, tag)["rule"]) in state


#: 2 家同款——输入：一种语言 × 全空间 48 组 case。输出：无。
TEST_PROMPTS_AND_QUESTIONS_LEAVE_NO_PLACEHOLDER: Final[tuple[str, ...]] = ('adblock', 'guardrail')


@pytest.mark.parametrize("name", TEST_PROMPTS_AND_QUESTIONS_LEAVE_NO_PLACEHOLDER)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompts_and_questions_leave_no_placeholder(name: str, lang: str) -> None:
    """输入：一种语言 × 全空间 48 组 case。输出：无。

    预期行为：提示词、反解提示词、题面与选项渲染完之后**一个花括号都不剩**——漏填会在 `format`
    上抛错，而多写的花括号会原样漏给模型，让它以为那是要填的槽。
    """
    m = module_of(name, 'test_locales', 'locales', 'features', 'judges', 'render', 'questions')
    locales, features, judges, render, questions = m.locales, m.features, m.judges, m.render, m.questions
    loc = locales.load(lang)
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case)
        assert "{" not in prompt and "}" not in prompt, case.describe()
        for item in questions.build(features.plan_for(case, 0), loc, 0):
            instructions = str(item.question["instructions"])
            assert "{" not in instructions and "}" not in instructions
    trace = judges.build_trace_prompt("[state]", loc)
    assert "{" not in trace and "}" not in trace


#: 2 家同款——输入：一种语言。输出：无。
TEST_RULE_AND_AVOID_COPY_NEVER_CARRIES_BRACES: Final[tuple[str, ...]] = ('sqlreview', 'subqa')


@pytest.mark.parametrize("name", TEST_RULE_AND_AVOID_COPY_NEVER_CARRIES_BRACES)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_rule_and_avoid_copy_never_carries_braces(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：判据原文与反面文案里**一个花括号都不许有**——这两段会被 `build_prompt` 拼进
    提示词，而花括号是填槽标记（`test_prompts_and_questions_leave_no_placeholder` 会扫
    「一个都不剩」）。`subqa` 上踩过一次：判据原文里写了那个花括号指令的具体形状，提示词的
    填槽自检当场把它当成没填的槽报出来。要举例就写「一对花括号里夹着反斜杠开头的短代码」这种
    说法，别把花括号本身写进去。
    """
    m = module_of(name, 'test_locales', 'features', 'locales', 'render')
    features, locales, render = m.features, m.locales, m.render
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        entry = render.tag_copy(loc, tag)
        for key in ("rule", "avoid"):
            text = str(entry[key])
            assert "{" not in text and "}" not in text, f"{lang}.{tag}.{key}: {text!r}"


#: 2 家同款——输入：全空间 24 组 case × 两种语言。输出：无。
TEST_CHOICE_OPTIONS_NEVER_DUPLICATE: Final[tuple[str, ...]] = ('alertrisk', 'depcheck')


@pytest.mark.parametrize("name", TEST_CHOICE_OPTIONS_NEVER_DUPLICATE)
def test_choice_options_never_duplicate(name: str) -> None:
    """输入：全空间 24 组 case × 两种语言。输出：无。

    预期行为：两族 choice 的选项描述两两不同——两句话一样，考生没法分辨，是构造 bug。
    """
    m = module_of(name, 'test_questions', 'locales', 'features', 'questions')
    locales, features, questions = m.locales, m.features, m.questions
    for lang in locales.available():
        loc = locales.load(lang)
        for case in features.build_cases(512):
            for item in questions.build(features.plan_for(case, 0), loc, 0):
                if item.question["type"] != "choice":
                    continue
                criteria = item.question["criteria"]
                assert isinstance(criteria, dict)
                values = [str(value) for value in criteria.values()]
                assert len(set(values)) == len(values), f"{lang}/{case.describe()}/{item.family}"


#: 2 家同款——输入：无。输出：题族清单。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FAMILIES_ARE_THE_FOUR_KNOWN_ONES_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'clipcut': ('keep_count', 'first_kept', 'blocked_any', 'dup_any', 'keep_count', 'first_kept', 'blocked_any', 'dup_any'),
    'quoteread': ('tier', 'leader', 'alert', 'divergent', 'tier', 'leader', 'alert', 'divergent'),
}


@pytest.mark.parametrize("name", TEST_FAMILIES_ARE_THE_FOUR_KNOWN_ONES_VALUES)
def test_families_are_the_four_known_ones(name: str) -> None:
    """输入：无。输出：题族清单。

    预期行为：四个题族、顺序稳定（先段数、再位置、最后两个布尔），三张分类表正好把四族分完
    ——多一族少一族都会让下面的「答案在选项里」漏查。
    """
    _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7 = TEST_FAMILIES_ARE_THE_FOUR_KNOWN_ONES_VALUES[name]
    m = module_of(name, 'test_questions', 'questions')
    questions = m.questions
    assert questions.FAMILIES == (_v0, _v1, _v2, _v3)
    assert questions.SCORE_FAMILIES == (_v4,)
    assert questions.CHOICE_FAMILIES == (_v5,)
    assert questions.NOUL_FAMILIES == (_v6, _v7)
    assert set(questions.FAMILIES) == (
        set(questions.CHOICE_FAMILIES)
        | set(questions.SCORE_FAMILIES)
        | set(questions.NOUL_FAMILIES)
    )


#: 2 家同款——输入：一组特征、语言字典。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_PROMPT_LISTS_ONLY_THE_TEXT_DIMENSIONS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'compact': ('这一块是什么', '有没有承载持久事实', '轮次位置', '是否钉住', '剩余预算'),
    'routing': ('这条工单是什么问题', '它有多急', '客户等级', '合规标记', '产品域'),
}


@pytest.mark.parametrize("name", TEST_PROMPT_LISTS_ONLY_THE_TEXT_DIMENSIONS_VALUES)
def test_prompt_lists_only_the_text_dimensions(name: str) -> None:
    """输入：一组特征、语言字典。输出：无。

    预期行为：提示词只提由文本承载的两维，绝不提轮次位置 / 是否钉住 / 剩余预算——
    提了模型就会把它们写进正文，与「本块位置」那一行抢同一件事（范式 §9.13）。
    """
    _v0, _v1, _v2, _v3, _v4 = TEST_PROMPT_LISTS_ONLY_THE_TEXT_DIMENSIONS_VALUES[name]
    m = module_of(name, 'test_render', 'TAGS', 'locales', 'render')
    TAGS, locales, render = m.TAGS, m.locales, m.render
    loc = locales.load("zh")
    prompt = render.build_prompt(TAGS, loc, 0)
    feature_block = prompt.split("\n\n")[1]
    assert _v0 in feature_block
    assert _v1 in feature_block
    assert _v2 not in feature_block
    assert _v3 not in feature_block
    assert _v4 not in feature_block
    assert "{" not in prompt


#: 3 家同款——输入：一种语言。输出：无。预期行为：写出去再读回来，判据表与日志一字不差。
TEST_STATE_ROUND_TRIPS: Final[tuple[str, ...]] = ('logtriage', 'memorygate', 'tracescore')


@pytest.mark.parametrize("name", TEST_STATE_ROUND_TRIPS)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：写出去再读回来，判据表与日志一字不差。
    """
    m = module_of(name, 'test_render', 'text_of', 'CASE', 'locales', 'render', 'features')
    text_of, CASE, locales, render, features = m.text_of, m.CASE, m.locales, m.render, m.features
    loc = locales.load(lang)
    text = text_of(CASE)
    state = render.to_state(render.ContentDraft(text=text), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.tags == features.TAG_IDS
    assert parsed.text == text
    assert render.detect_locale(state) == lang


#: 3 家同款——输入：正文里出现一行「像判据表」的内容。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_CONTENT_MENTIONING_A_RULE_DOES_NOT_BREAK_THE_PARSE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'logtriage': ('\n- N9：这一行是正文，不是判据表', '- N9：这一行是正文，不是判据表'),
    'memorygate': ('\n- M9：这一行是正文，不是判据表', '- M9：这一行是正文，不是判据表'),
    'tracescore': ('\n- S9：这一行是正文，不是信号表', '- S9：这一行是正文，不是信号表'),
}


@pytest.mark.parametrize("name", TEST_CONTENT_MENTIONING_A_RULE_DOES_NOT_BREAK_THE_PARSE_VALUES)
def test_content_mentioning_a_rule_does_not_break_the_parse(name: str) -> None:
    """输入：正文里出现一行「像判据表」的内容。输出：无。

    预期行为：日志抬头之后的一切都当正文——否则堆栈里出现一行 `- N9：假的` 就会把判据表数歪。
    """
    _v0, _v1 = TEST_CONTENT_MENTIONING_A_RULE_DOES_NOT_BREAK_THE_PARSE_VALUES[name]
    m = module_of(name, 'test_render', 'locales', 'text_of', 'CASE', 'render', 'features')
    locales, text_of, CASE, render, features = m.locales, m.text_of, m.CASE, m.render, m.features
    loc = locales.load("zh")
    text = text_of(CASE) + _v0
    parsed = render.parse_with(render.to_state(render.ContentDraft(text=text), loc), loc)
    assert parsed.tags == features.TAG_IDS
    assert _v1 in str(parsed.text)


#: 2 家同款——输入：无。输出：题族清单。（与 `test_families_are_the_four_known_ones` 同名不同形：字面量个数不一样，拆成两份）
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FAMILIES_ARE_THE_FOUR_KNOWN_ONES_2_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'sentiment': ('tone', 'pivot', 'drop', 'rebound', 'tone', 'pivot', 'drop', 'rebound'),
    'siftrank': ('agg_top', 'pair_order', 'cut_size', 'stable', 'agg_top', 'pair_order', 'cut_size', 'stable'),
}


@pytest.mark.parametrize("name", TEST_FAMILIES_ARE_THE_FOUR_KNOWN_ONES_2_VALUES)
def test_families_are_the_four_known_ones_2(name: str) -> None:
    """输入：无。输出：题族清单。

    预期行为：四个题族、顺序稳定（先整段倾向、再转折位置、再落差、最后回升），三张分类表
    正好把四族分完——多一族少一族都会让下面的「答案在选项里」漏查。
    """
    _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7 = TEST_FAMILIES_ARE_THE_FOUR_KNOWN_ONES_2_VALUES[name]
    m = module_of(name, 'test_questions', 'questions')
    questions = m.questions
    assert questions.FAMILIES == (_v0, _v1, _v2, _v3)
    assert questions.CHOICE_FAMILIES == (_v4, _v5)
    assert questions.SCORE_FAMILIES == (_v6,)
    assert questions.NOUL_FAMILIES == (_v7,)
    assert set(questions.FAMILIES) == (
        set(questions.CHOICE_FAMILIES)
        | set(questions.SCORE_FAMILIES)
        | set(questions.NOUL_FAMILIES)
    )


#: 2 家同款——输入：四个切分的配置。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_SPLIT_SIZES_ADD_UP_TO_THE_TRAINING_SIDE_MATRIX_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'cachereuse': ('cachereuse', 3, 4, 2),
    'match': ('match', 7, 12, 3),
}


@pytest.mark.parametrize("name", TEST_SPLIT_SIZES_ADD_UP_TO_THE_TRAINING_SIDE_MATRIX_VALUES)
def test_split_sizes_add_up_to_the_training_side_matrix(name: str) -> None:
    """输入：四个切分的配置。输出：无。

    预期行为：三个同分布切分都非空（切片按**位置**而不是桶号过滤，否则哈希方差会让 val
    只剩个位数），且加起来正好等于训练侧取值表的全枚举：3 步数 × 4 档 × 2 分叉原因
    × 2 写法 × 2 来源。
    """
    _v0, _v1, _v2, _v3 = TEST_SPLIT_SIZES_ADD_UP_TO_THE_TRAINING_SIDE_MATRIX_VALUES[name]
    workflow = workflow_mod.load(_v0)
    sizes = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in workflow.split_names()
    }
    assert min(sizes.values()) > 0
    assert sum(sizes[split] for split in ("train", "val", "test_iid")) == _v1 * _v2 * 2 * _v3 * 2


#: 2 家同款——输入：一种语言。输出：无。
TEST_FORMAT_SPEC_ONLY_OFFERS_PLACEHOLDERS: Final[tuple[str, ...]] = ('debugstep', 'sqlreview')


@pytest.mark.parametrize("name", TEST_FORMAT_SPEC_ONLY_OFFERS_PLACEHOLDERS)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_spec_only_offers_placeholders(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：格式规格那五行**只有占位符、没有一个能整段照抄的成品**——每一行的取值都得模型自己
    编，规格里只印键名与 `<...>`；行数正好五个结构键。
    """
    m = module_of(name, 'test_render', 'locales', 'render', 'features')
    locales, render, features = m.locales, m.render, m.features
    loc = locales.load(lang)
    lines = render.format_lines(loc)
    assert len(lines) == len(features.KEY_LINES)
    for key, line in zip(features.KEY_LINES, lines, strict=True):
        assert line.startswith(render.LINE_PREFIX)
        assert key in line.lower(), f"{lang}.{key}"
        assert "<" in line and ">" in line, f"{lang}.{key}"


#: 3 家同款——输入：留出侧（`decoy=dashed`）却不写诱饵。输出：RunResult。预期行为：整批都被拦下。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_MISSING_DECOY_ON_THE_OOD_SPLIT_IS_REJECTED_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'guardrail': ('g3-decoy',),
    'posttriage': ('p3-decoy',),
    'receiptcat': ('p3-decoy',),
}


@pytest.mark.parametrize("name", TEST_MISSING_DECOY_ON_THE_OOD_SPLIT_IS_REJECTED_VALUES)
def test_missing_decoy_on_the_ood_split_is_rejected(name: str, tmp_path: Path) -> None:
    """输入：留出侧（`decoy=dashed`）却不写诱饵。输出：RunResult。预期行为：整批都被拦下。
    """
    _v0, = TEST_MISSING_DECOY_ON_THE_OOD_SPLIT_IS_REJECTED_VALUES[name]
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    fake_backend = conftest.FakeBackend
    m = module_of(name, 'test_pipeline', 'FakeBackend', 'make_config')
    FakeBackend, make_config = m.FakeBackend, m.make_config
    result = pipeline.run(
        make_config(tmp_path, _v0, fake_backend(decoy_missing=True), split="test_ood")
    )
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


#: 2 家同款——输入：几种引文。输出：无。（与 `test_quote_holds_needs_both_the_quote_and_the_criterion` 同名不同形：字面量个数不一样，拆成两份）
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_QUOTE_HOLDS_NEEDS_BOTH_THE_QUOTE_AND_THE_CRITERION_2_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'adblock': ('A1', 'A4', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A4', 'A1', 'A9'),
    'turntake': ('T1', 'T4', 'T1', 'T1', 'T1', 'T1', 'T1', 'T1', 'T1', 'T4', 'T1', 'T9'),
}


@pytest.mark.parametrize("name", TEST_QUOTE_HOLDS_NEEDS_BOTH_THE_QUOTE_AND_THE_CRITERION_2_VALUES)
def test_quote_holds_needs_both_the_quote_and_the_criterion_2(name: str) -> None:
    """输入：几种引文。输出：无。

    预期行为：引文得原样在片段里、且自己满足那条判据；空白差异不算抄错。
    """
    _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8, _v9, _v10, _v11 = TEST_QUOTE_HOLDS_NEEDS_BOTH_THE_QUOTE_AND_THE_CRITERION_2_VALUES[name]
    m = module_of(name, 'test_judges', 'text_of', 'judges', 'quote_for')
    text_of, judges, quote_for = m.text_of, m.judges, m.quote_for
    body = text_of((_v0, _v1))
    assert judges.quote_holds(_v2, quote_for(_v3), body)
    # 引文首尾多带了换行：折叠之后一致，照样算抄对了（模型常这么答）。
    assert judges.quote_holds(_v4, f"\n{quote_for(_v5)}\n", body)
    assert judges.quote_holds(_v6, json.dumps([quote_for(_v7)]), body)  # 引文被装进 JSON 数组字符串（范式 §9.88）
    assert not judges.quote_holds(_v8, quote_for(_v9), body)  # 抄出来了但不合判据
    assert not judges.quote_holds(_v10, "会议室还是老地方", body)  # 抄出来了但不合判据
    assert not judges.quote_holds(_v11, "会议室还是老地方", body)  # 不认识的编号


#: 2 家同款——输入：把一条 `grade`（score）的等级下标改掉。输出：无。
TEST_SCORE_ANSWER_MISMATCH_IS_CAUGHT: Final[tuple[str, ...]] = ('alertrisk', 'depcheck')


@pytest.mark.parametrize("name", TEST_SCORE_ANSWER_MISMATCH_IS_CAUGHT)
def test_score_answer_mismatch_is_caught(name: str, tmp_path: Path) -> None:
    """输入：把一条 `grade`（score）的等级下标改掉。输出：无。

    预期行为：`grade_bad` 报 1——审计拿答案下标取回等级描述，再与按文本查矩阵算出来的那一档比。
    """
    m = module_of(name, 'test_audit', '_run', '_rows', '_write', '_audit', 'features', 'BAD_LABELS')
    _run, _rows, _write, _audit, features, BAD_LABELS = m._run, m._rows, m._write, m._audit, m.features, m.BAD_LABELS
    run_dir = _run(tmp_path, "t-grade")
    rows = _rows(run_dir)
    answer = rows[0]["answers"]["grade"]
    answer["score"] = (answer["score"] + 1) % len(features.LEVELS)
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["grade_bad"] == 1
    assert any(BAD_LABELS["grade_bad"] in item for item in missing)


#: 2 家同款——输入：一种语言。输出：无。（与 `test_rule_and_avoid_copy_never_carries_braces` 同名不同形：字面量个数不一样，拆成两份）
TEST_RULE_AND_AVOID_COPY_NEVER_CARRIES_BRACES_2: Final[tuple[str, ...]] = ('alertrisk', 'depcheck')


@pytest.mark.parametrize("name", TEST_RULE_AND_AVOID_COPY_NEVER_CARRIES_BRACES_2)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_rule_and_avoid_copy_never_carries_braces_2(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：因子原文与反面文案里**一个花括号都不许有**——这两段会被 `build_prompt` 拼进提示词，
    而花括号是填槽标记（`test_prompts_and_questions_leave_no_placeholder` 会扫「一个都不剩」）。
    """
    m = module_of(name, 'test_locales', 'features', 'locales', 'render')
    features, locales, render = m.features, m.locales, m.render
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        entry = render.fact_copy(loc, tag)
        for key in ("rule", "avoid"):
            text = str(entry[key])
            assert "{" not in text and "}" not in text, f"{lang}.{tag}.{key}: {text!r}"


#: 2 家同款——输入：一种语言。输出：无。
TEST_FACT_LINES_LINE_UP_WITH_FEATURES: Final[tuple[str, ...]] = ('alertrisk', 'depcheck')


@pytest.mark.parametrize("name", TEST_FACT_LINES_LINE_UP_WITH_FEATURES)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_fact_lines_line_up_with_features(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：三条因子行两两不同、编号排在分隔符前面（解析就是按这个切回编号的）、判据原文在里面。
    """
    m = module_of(name, 'test_render', 'locales', 'features', 'render')
    locales, features, render = m.locales, m.features, m.render
    loc = locales.load(lang)
    sep = str(loc["state"]["tag_sep"])
    lines = [render.fact_line(loc, tag) for tag in features.TAG_IDS]
    assert len(set(lines)) == len(lines)
    for tag, line in zip(features.TAG_IDS, lines, strict=True):
        assert line.partition(sep)[0] == tag
        assert str(render.fact_copy(loc, tag)["rule"]) in line


#: 2 家同款——输入：一个语言。输出：无。
TEST_BANNED_PHRASES_ARE_THE_TABLE_NAMES: Final[tuple[str, ...]] = ('review', 'rubric')


@pytest.mark.parametrize("name", TEST_BANNED_PHRASES_ARE_THE_TABLE_NAMES)
@pytest.mark.parametrize("lang", ["zh", "en"])
def test_banned_phrases_are_the_table_names(name: str, lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：禁写清单里就是那几个表名——正文是一段代码，收进 `test` / `size` 这类
    标识符会大面积误伤（见 `render.banned_echo` 的口径）。
    """
    m = module_of(name, 'test_locales', 'locales', 'render')
    locales, render = m.locales, m.render
    loc = locales.load(lang)
    banned = [str(item) for item in loc["banned_intent_phrases"]]
    assert banned
    # 比对时统一小写：英文的段落名首字母大写，而禁写清单按小写匹配（`banned_echo` 也是）
    titles = {str(loc["meta"][key]).lower() for key in render.SECTION_KEYS}
    assert {item.lower() for item in banned} == titles


#: 2 家同款——输入：一种语言。输出：无。
TEST_NOTE_SPEC_FORBIDS_RESTATING_THE_FORMAT: Final[tuple[str, ...]] = ('sqlreview', 'subqa')


@pytest.mark.parametrize("name", TEST_NOTE_SPEC_FORBIDS_RESTATING_THE_FORMAT)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_note_spec_forbids_restating_the_format(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：`Note:` 那一行的规格里**明说不要复述格式与要求**。真机上量到过：模型把备注写成
    「译文保留了音效标记，时间码保持零时长以便闪现提醒」这种复述合成规格的句子（`sq-10b` zh 侧
    9 条里 3 条、en 侧 10 条里 2 条）。备注是这一族两段自然语言正文之一，写成复述规格的句子之后
    整条样本读起来就是「写手在照着规格交作业」，而不是一对真字幕。
    """
    m = module_of(name, 'test_render', 'locales')
    locales = m.locales
    loc = locales.load(lang)
    spec = str(loc["prompt"]["format"]["note"])
    assert "格式" in spec or "format" in spec.lower()
    assert "不要" in spec or "do **not**" in spec.lower()


#: 3 家同款——输入：无。输出：无。
TEST_EVERY_TAG_APPEARS_EIGHT_TIMES: Final[tuple[str, ...]] = ('adblock', 'promptinject', 'tracescore')


@pytest.mark.parametrize("name", TEST_EVERY_TAG_APPEARS_EIGHT_TIMES)
def test_every_tag_appears_eight_times(name: str) -> None:
    """输入：无。输出：无。

    预期行为：每条判据在 16 组里各出现 8 次——这是 `flag` 那一族的答案按构造就是 50% 的原因。
    """
    m = module_of(name, 'test_features', 'features')
    features = m.features
    for tag in features.TAG_IDS:
        hits = sum(1 for value in features.HIT_SETS if tag in features.parse_hits(value))
        assert hits == 8, tag


#: 3 家同款——输入：全空间 48 组 case 的忠实样本。输出：无。预期行为：一条都不报。
TEST_HYGIENE_ACCEPTS_EVERY_FAITHFUL_SAMPLE: Final[tuple[str, ...]] = ('adblock', 'chatgate', 'guardrail')


@pytest.mark.parametrize("name", TEST_HYGIENE_ACCEPTS_EVERY_FAITHFUL_SAMPLE)
def test_hygiene_accepts_every_faithful_sample(name: str) -> None:
    """输入：全空间 48 组 case 的忠实样本。输出：无。预期行为：一条都不报。
    """
    m = module_of(name, 'test_judges', 'CASES', 'judges', 'faithful', 'CASE')
    CASES, judges, faithful, CASE = m.CASES, m.judges, m.faithful, m.CASE
    judge = judges.HygieneJudge("zh")
    for case in CASES:
        spec, sample = faithful(case)
        verdict = judge.judge(sample, spec)
        assert verdict.ok and not verdict.skipped, f"{case.describe()}: {verdict.reason}"
    assert "兑现" in judge.judge(*reversed(faithful(CASE))).reason


#: 2 家同款——输入：一种语言。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FORMAT_SPEC_OFFERS_NO_COPYABLE_VALUE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'alertrisk': ('hosts=\\d', 'reachable=public', 'auth=none'),
    'depcheck': ('count=[1-9]', 'AGPL-3.0-only', '0.9.1'),
}


@pytest.mark.parametrize("name", TEST_FORMAT_SPEC_OFFERS_NO_COPYABLE_VALUE_VALUES)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_spec_offers_no_copyable_value(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：格式规格里**不许出现填好的成品**——给一个带数字的 `hosts=`、或者给一个真的
    `reachable=public`，模型照抄下去正好取到那一档（范式 §9.52）。规格只印键名与 `<...>` 占位符。
    """
    _v0, _v1, _v2 = TEST_FORMAT_SPEC_OFFERS_NO_COPYABLE_VALUE_VALUES[name]
    m = module_of(name, 'test_locales', 'locales', 'render')
    locales, render = m.locales, m.render
    loc = locales.load(lang)
    for line in render.format_lines(loc):
        assert not re.search(_v0, line), line
        assert _v1 not in line, line
        assert _v2 not in line, line


#: 2 家同款——输入：一组特征。输出：无。
TEST_ONLY_THE_SCORE_AND_NOUL_FAMILIES_ARE_UNSHUFFLED: Final[tuple[str, ...]] = ('browser', 'compact')


@pytest.mark.parametrize("name", TEST_ONLY_THE_SCORE_AND_NOUL_FAMILIES_ARE_UNSHUFFLED)
def test_only_the_score_and_noul_families_are_unshuffled(name: str) -> None:
    """输入：一组特征。输出：无。

    预期行为：有序阶梯与布尔题不打乱（打乱破坏语义），三个 choice 题族都允许打乱
    （防「答案总在第 n 个」的位置捷径）。
    """
    m = module_of(name, 'test_questions', 'by_family', 'questions')
    by_family, questions = m.by_family, m.questions
    families = by_family()
    for family in questions.FAMILIES:
        if family in questions.SCORE_FAMILIES or family in questions.NOUL_FAMILIES:
            assert families[family].shuffle is False
        else:
            assert families[family].shuffle is True


#: 2 家同款——输入：三个问法编号。输出：无。预期行为：同一族的题面在三个编号上都取得到。
TEST_PHRASINGS_ROTATE: Final[tuple[str, ...]] = ('browser', 'compact')


@pytest.mark.parametrize("name", TEST_PHRASINGS_ROTATE)
def test_phrasings_rotate(name: str) -> None:
    """输入：三个问法编号。输出：无。预期行为：同一族的题面在三个编号上都取得到。
    """
    m = module_of(name, 'test_questions', 'questions', 'TAGS', 'locales')
    questions, TAGS, locales = m.questions, m.TAGS, m.locales
    for family in questions.FAMILIES:
        texts = {
            next(
                item.question["instructions"]
                for item in questions.build(TAGS, locales.load("zh"), phrasing)
                if item.family == family
            )
            for phrasing in (0, 1, 2)
        }
        assert len(texts) == 3, family


#: 2 家同款——输入：每种语言的判据原文。输出：无。
TEST_POLICY_PRINTS_EVERY_TABLE: Final[tuple[str, ...]] = ('elementpick', 'skillpick')


@pytest.mark.parametrize(("name", "lang"), _lang_pairs(TEST_POLICY_PRINTS_EVERY_TABLE, "test_locales", "LANGS"))
def test_policy_prints_every_table(name: str, lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：六个角色、两档可见性、两档可点性都在判据里印出来——考生只能从题面读规则，
    漏印一项就等于把那一族的答案变成不可解。
    """
    m = module_of(name, 'test_locales', 'VOCAB_TABLES', 'LANGS', 'locales')
    VOCAB_TABLES, LANGS, locales = m.VOCAB_TABLES, m.LANGS, m.locales
    loc = locales.load(lang)
    policy = str(loc["policy"]["text"])
    for table in VOCAB_TABLES:
        for text in loc[table].values():
            assert str(text) in policy, (lang, table, text)


#: 3 家同款——输入：两门语言的字段名写成一样。输出：无。预期行为：报人话（合并表有歧义）。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_MERGED_REJECTS_COLLIDING_LABELS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'elementpick': ('meta', 'screen_label', 'meta', 'screen_label'),
    'sim': ('label', 'request', 'label', 'request'),
    'skillpick': ('meta', 'tag_label', 'meta', 'tag_label'),
}


@pytest.mark.parametrize("name", TEST_MERGED_REJECTS_COLLIDING_LABELS_VALUES)
def test_merged_rejects_colliding_labels(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言的字段名写成一样。输出：无。预期行为：报人话（合并表有歧义）。
    """
    _v0, _v1, _v2, _v3 = TEST_MERGED_REJECTS_COLLIDING_LABELS_VALUES[name]
    m = module_of(name, 'test_render_parse', '_patched_merged', 'parse')
    _patched_merged, parse = m._patched_merged, m.parse
    _patched_merged(
        monkeypatch,
        lambda en, zh: en[_v0].update({_v1: zh[_v2][_v3]}),
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


#: 2 家同款——输入：语言文件。输出：无。
TEST_META_PUNCTUATION_COMES_FROM_THE_LOCALE: Final[tuple[str, ...]] = ('hybrid', 'invoice')


@pytest.mark.parametrize(("name", "lang"), _lang_pairs(TEST_META_PUNCTUATION_COMES_FROM_THE_LOCALE, "test_locales", "LOCALES"))
def test_meta_punctuation_comes_from_the_locale(name: str, lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：冒号与分隔符由语言文件给；英文里不能有中日韩标点——
    否则英文日程会带着「：」，看着就不像英文数据。
    """
    m = module_of(name, 'test_locales', 'LOCALES', 'locales')
    LOCALES, locales = m.LOCALES, m.locales
    loc = locales.load(lang)
    assert loc["meta"]["colon"] and loc["meta"]["sep"]
    if lang == "en":
        for key in ("colon", "sep"):
            assert not (set("：；。") & set(loc["meta"][key]))


#: 2 家同款——输入：一种语言。输出：无。
TEST_AVOID_LISTS_TRIP_EXACTLY_THEIR_OWN_TAG: Final[tuple[str, ...]] = ('logtriage', 'tracescore')


@pytest.mark.parametrize("name", TEST_AVOID_LISTS_TRIP_EXACTLY_THEIR_OWN_TAG)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_avoid_lists_trip_exactly_their_own_tag(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每条判据的「不许出现」文案**只**命中它自己那一条——它是提示词里给模型看的反面
    清单，同时也写进 state 的判据表，同时踩中两条就会把模型往错的方向带（范式 §9.45 起，
    提示词里唯一的具体文字就是判据原文与这份反面清单，所以它更要钉住）。
    """
    m = module_of(name, 'test_locales', 'features', 'locales', 'render')
    features, locales, render = m.features, m.locales, m.render
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        avoid = str(render.tag_copy(loc, tag)["avoid"])
        assert features.hit_set(avoid) == (tag,), f"{lang}.{tag}: {avoid!r}"


#: 2 家同款——输入：一条规格。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_INSTRUCTIONS_CARRY_THE_WHOLE_RULE_BLOCK_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'planstep': (4,),
    'sentiment': ('interview',),
}


@pytest.mark.parametrize("name", TEST_INSTRUCTIONS_CARRY_THE_WHOLE_RULE_BLOCK_VALUES)
def test_instructions_carry_the_whole_rule_block(name: str) -> None:
    """输入：一条规格。输出：无。

    预期行为：四道题的题面都自带**四条判据原文**（与 state 里那一段同源）——只读 state 与
    题面的读者必须能自己把答案算出来，判据藏在生成器里等于没给。
    """
    _v0, = TEST_INSTRUCTIONS_CARRY_THE_WHOLE_RULE_BLOCK_VALUES[name]
    m = module_of(name, 'test_questions', 'locales', 'questions', '_make_spec')
    locales, questions, _make_spec = m.locales, m.questions, m._make_spec
    loc = locales.load("zh")
    rules = questions.rules_block(loc)
    for line in loc["rules"]["lines"]:
        assert str(line) in rules
    for item in questions.build(_make_spec(2, _v0), loc, 0, random.Random(0)):
        assert rules in str(item.question["instructions"]), item.family


#: 2 家同款——输入：一种语言。输出：无。预期行为：两张表都覆盖四个档位，动作文案两两不同。
TEST_ROUTE_AND_THRESHOLD_TABLES_LINE_UP_WITH_FEATURES: Final[tuple[str, ...]] = ('posttriage', 'tracescore')


@pytest.mark.parametrize("name", TEST_ROUTE_AND_THRESHOLD_TABLES_LINE_UP_WITH_FEATURES)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_route_and_threshold_tables_line_up_with_features(name: str, lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：两张表都覆盖四个档位，动作文案两两不同。
    """
    m = module_of(name, 'test_locales', 'locales', 'features')
    locales, features = m.locales, m.features
    loc = locales.load(lang)
    options = loc["fam"]["route"]["opt"]
    assert isinstance(options, dict)
    assert sorted(options) == sorted(features.ROUTE_LEVELS)
    assert len({str(value) for value in options.values()}) == len(options)
    thresholds = loc["state"]["threshold"]
    assert isinstance(thresholds, dict)
    assert sorted(thresholds) == sorted(features.ROUTE_LEVELS)
    assert len({str(value) for value in thresholds.values()}) == len(thresholds)


#: 2 家同款——输入：四个切分。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_ID_AND_OOD_TOGETHER_COVER_EVERY_COMPATIBLE_CASE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'toolgate': ('toolgate', 10000),
    'verify': ('verify', 1000),
}


@pytest.mark.parametrize("name", TEST_ID_AND_OOD_TOGETHER_COVER_EVERY_COMPATIBLE_CASE_VALUES)
def test_id_and_ood_together_cover_every_compatible_case(name: str) -> None:
    """输入：四个切分。输出：无。

    预期行为：ID 侧 100 组 + OOD 侧 86 组**恰好**等于全部 186 个相容组合——
    留出不是「少给组合」，是把同一批组合换个切分，一个都不丢。
    """
    _v0, _v1 = TEST_ID_AND_OOD_TOGETHER_COVER_EVERY_COMPATIBLE_CASE_VALUES[name]
    m = module_of(name, 'test_pipeline', 'features')
    features = m.features
    workflow = workflow_mod.load(_v0)
    seen: set[tuple[tuple[str, str], ...]] = set()
    for split in workflow.split_names():
        for case in workflow.build_cases(workflow.config_for(split)):
            seen.add(tuple(case.as_dict().items()))
    assert seen == {tuple(case.as_dict().items()) for case in features.build_cases(_v1)}


#: 2 家同款——输入：全空间 24 组 case。输出：无。预期行为：`act` 的答案是等级查档位表算出来的那一档。
TEST_ACT_ANSWERS_FOLLOW_THE_LEVEL_TABLE: Final[tuple[str, ...]] = ('alertrisk', 'depcheck')


@pytest.mark.parametrize("name", TEST_ACT_ANSWERS_FOLLOW_THE_LEVEL_TABLE)
def test_act_answers_follow_the_level_table(name: str) -> None:
    """输入：全空间 24 组 case。输出：无。预期行为：`act` 的答案是等级查档位表算出来的那一档。
    """
    m = module_of(name, 'test_questions', 'locales', 'features', 'questions')
    locales, features, questions = m.locales, m.features, m.questions
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "act")
        assert item.answer == features.act_for(plan.level)
        criteria = item.question["criteria"]
        assert isinstance(criteria, dict)
        assert criteria[item.answer] == questions.act_option(loc, item.answer)
        assert sorted(criteria) == sorted(features.ACTS)


#: 2 家同款——输入：档位下标越界的产物。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_AUDIT_FLAGS_AN_OUT_OF_RANGE_SCORE_INDEX_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'bench': ('count',),
    'sim': ('gain',),
}


@pytest.mark.parametrize("name", TEST_AUDIT_FLAGS_AN_OUT_OF_RANGE_SCORE_INDEX_VALUES)
def test_audit_flags_an_out_of_range_score_index(name: str, tmp_path: Path) -> None:
    """输入：档位下标越界的产物。输出：无。

    预期行为：score 也查下标（条数只有六档、差距只有两档），越界同样记一笔「编号越界」。
    """
    _v0, = TEST_AUDIT_FLAGS_AN_OUT_OF_RANGE_SCORE_INDEX_VALUES[name]
    m = module_of(name, 'test_edges', '_sample_row', 'WF')
    _sample_row, WF = m._sample_row, m.WF
    row = _sample_row()
    row["answers"][_v0]["score"] = 9
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


#: 2 家同款——输入：维度表与全空间枚举。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FULL_SPACE_DROPS_EXACTLY_THE_TWO_FORBIDDEN_PAIRS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'compact': (180, 144),
    'review': (144, 108),
}


@pytest.mark.parametrize("name", TEST_FULL_SPACE_DROPS_EXACTLY_THE_TWO_FORBIDDEN_PAIRS_VALUES)
def test_full_space_drops_exactly_the_two_forbidden_pairs(name: str) -> None:
    """输入：维度表与全空间枚举。输出：无。

    预期行为：5×2×3×2×3 = 180 组；两对禁用对各砍掉 3×2×3 = 18 组
    （`工具输出 × 持久事实`、`系统提示词 × 没有持久事实`），正好剩 144 组。
    """
    _v0, _v1 = TEST_FULL_SPACE_DROPS_EXACTLY_THE_TWO_FORBIDDEN_PAIRS_VALUES[name]
    m = module_of(name, 'test_features', 'features', 'all_tags')
    features, all_tags = m.features, m.all_tags
    total = 1
    for values in features.DIMS.values():
        total *= len(values)
    assert total == _v0
    assert len(all_tags()) == _v1


#: 2 家同款——输入：一个禁用对（正反两种写法）。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_ALLOW_PAIR_IS_ORDER_FREE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'compact': ('fact', 'fact', 'kind', 'tool_output', 'kind', 'tool_output', 'fact', 'fact', 'fact', 'chatter', 'kind', 'system', 'kind', 'system', 'fact', 'chatter', 'fact', 'chatter', 'kind', 'tool_output', 'fact', 'fact', 'kind', 'user', 'fact', 'chatter', 'kind', 'user'),
    'review': ('secret', 'yes', 'change_kind', 'deleted', 'change_kind', 'deleted', 'secret', 'yes', 'debug', 'yes', 'change_kind', 'deleted', 'change_kind', 'deleted', 'debug', 'yes', 'secret', 'yes', 'change_kind', 'modified', 'secret', 'no', 'change_kind', 'deleted', 'debug', 'yes', 'size', 'small'),
}


@pytest.mark.parametrize("name", TEST_ALLOW_PAIR_IS_ORDER_FREE_VALUES)
def test_allow_pair_is_order_free(name: str) -> None:
    """输入：一个禁用对（正反两种写法）。输出：无。

    预期行为：归一化按 `DIMS` 的声明顺序做，所以换顺序也一样被否掉——按字符串比大小
    会漏掉一半写法，那是这个函数最容易写错的地方。
    """
    _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8, _v9, _v10, _v11, _v12, _v13, _v14, _v15, _v16, _v17, _v18, _v19, _v20, _v21, _v22, _v23, _v24, _v25, _v26, _v27 = TEST_ALLOW_PAIR_IS_ORDER_FREE_VALUES[name]
    m = module_of(name, 'test_features', 'features')
    features = m.features
    assert not features.allow_pair(_v0, _v1, _v2, _v3)
    assert not features.allow_pair(_v4, _v5, _v6, _v7)
    assert not features.allow_pair(_v8, _v9, _v10, _v11)
    assert not features.allow_pair(_v12, _v13, _v14, _v15)
    assert features.allow_pair(_v16, _v17, _v18, _v19)
    assert features.allow_pair(_v20, _v21, _v22, _v23)
    assert features.allow_pair(_v24, _v25, _v26, _v27)


#: 3 家同款——输入：两门语言把同一个动作写成同一个词。输出：无。预期行为：报人话。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_MERGED_REJECTS_COLLIDING_VOCABULARIES_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'elementpick': ('label', 'save', 'label', 'save'),
    'sim': ('action', 'advance', 'action', 'advance'),
    'skillpick': ('tag', 'read_code', 'tag', 'read_code'),
}


@pytest.mark.parametrize("name", TEST_MERGED_REJECTS_COLLIDING_VOCABULARIES_VALUES)
def test_merged_rejects_colliding_vocabularies(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言把同一个动作写成同一个词。输出：无。预期行为：报人话。
    """
    _v0, _v1, _v2, _v3 = TEST_MERGED_REJECTS_COLLIDING_VOCABULARIES_VALUES[name]
    m = module_of(name, 'test_render_parse', '_patched_merged', 'parse')
    _patched_merged, parse = m._patched_merged, m.parse
    _patched_merged(
        monkeypatch,
        lambda en, zh: en[_v0].update({_v1: zh[_v2][_v3]}),
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


#: 2 家同款——输入：全部特征组合。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_EVERY_CASE_ASKS_EVERY_FAMILY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'modelroute': (48,),
    'shell': (32,),
}


@pytest.mark.parametrize(("name", "lang"), _lang_pairs(TEST_EVERY_CASE_ASKS_EVERY_FAMILY_VALUES, "test_questions", "LOCALES"))
def test_every_case_asks_every_family(name: str, lang: str) -> None:
    """输入：全部特征组合。输出：无。

    预期行为：五个题族对**任何**组合都问得出来——本 workflow 没有条件题族，
    所以每条样本的题族数恒定。
    """
    _v0, = TEST_EVERY_CASE_ASKS_EVERY_FAMILY_VALUES[name]
    m = module_of(name, 'test_questions', 'LOCALES', 'features', 'questions', '_build')
    LOCALES, features, questions, _build = m.LOCALES, m.features, m.questions, m._build
    cases = features.build_cases(_v0)
    assert cases
    for case in cases:
        assert [item.family for item in _build(case, lang)] == list(questions.FAMILIES)


#: 2 家同款——输入：全空间组合。输出：无。
TEST_EVERY_COMPATIBLE_COMBINATION_CAN_BE_WRITTEN: Final[tuple[str, ...]] = ('review', 'rubric')


@pytest.mark.parametrize("name", TEST_EVERY_COMPATIBLE_COMBINATION_CAN_BE_WRITTEN)
def test_every_compatible_combination_can_be_written(name: str) -> None:
    """输入：全空间组合。输出：无。

    预期行为：假后端能给每一格写出相称的改动，而且**过得了渲染期的形状检查**——
    夹具缺格或区间定错会让「正文承载两维」这条前提在测试里悄悄失效。
    """
    m = module_of(name, 'test_features', 'all_tags', 'locales', 'body_for', 'render')
    all_tags, locales, body_for, render = m.all_tags, m.locales, m.body_for, m.render
    loc = locales.load("zh")
    for tags in all_tags():
        body = body_for(tags.as_dict())
        assert body.strip()
        render.check_draft({"body": body}, tags, loc)


#: 3 家同款——输入：两个互为前缀的候选标签。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FIELD_VALUE_PREFERS_THE_LONGEST_LABEL_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'elementpick': ('屏号：S-1', '屏', '屏号', 'S-1', '屏：S-1', '屏', '屏号', 'S-1'),
    'sim': ('单号：S-1', '单', '单号', 'S-1', '单：S-1', '单', '单号', 'S-1'),
    'skillpick': ('标签号：Q-1', '标签', '标签号', 'Q-1', '标签：Q-1', '标签', '标签号', 'Q-1'),
}


@pytest.mark.parametrize("name", TEST_FIELD_VALUE_PREFERS_THE_LONGEST_LABEL_VALUES)
def test_field_value_prefers_the_longest_label(name: str) -> None:
    """输入：两个互为前缀的候选标签。输出：无。

    预期行为：取最长的那个——否则「屏」会把「屏号」那一格提前吃掉。
    """
    _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7 = TEST_FIELD_VALUE_PREFERS_THE_LONGEST_LABEL_VALUES[name]
    m = module_of(name, 'test_render_parse', 'read_field_value')
    read_field_value = m.read_field_value
    assert read_field_value(_v0, (_v1, _v2)) == _v3
    assert read_field_value(_v4, (_v5, _v6)) == _v7


#: 3 家同款——输入：把两个角色写成同一个原文。输出：无。预期行为：报人话（反解会认错键）。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_LABELS_OF_REJECTS_TWO_KEYS_SHARING_ONE_TEXT_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'elementpick': ('role', 'role', 'role', 'toggle', 'role', 'button'),
    'sim': ('cost', 'cost', 'cost', 'c2', 'cost', 'c1'),
    'skillpick': ('scope', 'scope', 'scope', 'write', 'scope', 'read'),
}


@pytest.mark.parametrize("name", TEST_LABELS_OF_REJECTS_TWO_KEYS_SHARING_ONE_TEXT_VALUES)
def test_labels_of_rejects_two_keys_sharing_one_text(name: str) -> None:
    """输入：把两个角色写成同一个原文。输出：无。预期行为：报人话（反解会认错键）。
    """
    _v0, _v1, _v2, _v3, _v4, _v5 = TEST_LABELS_OF_REJECTS_TWO_KEYS_SHARING_ONE_TEXT_VALUES[name]
    m = module_of(name, 'test_render_parse', 'locales', 'parse')
    locales, parse = m.locales, m.parse
    loc = dict(locales.load("zh"))
    loc[_v0] = dict(loc[_v1])
    loc[_v2][_v3] = loc[_v4][_v5]
    with pytest.raises(ValueError, match="共用同一个原文"):
        parse.labels_of(loc)


#: 2 家同款——输入：state 读不出来的产物。输出：无。
TEST_AUDIT_FLAGS_A_STATE_IT_CANNOT_READ: Final[tuple[str, ...]] = ('bench', 'functions')


@pytest.mark.parametrize("name", TEST_AUDIT_FLAGS_A_STATE_IT_CANNOT_READ)
def test_audit_flags_a_state_it_cannot_read(name: str, tmp_path: Path) -> None:
    """输入：state 读不出来的产物。输出：无。

    预期行为：记一条「反解不出来」而不是静默跳过——静默跳过会把一条坏样本算成好样本。
    """
    m = module_of(name, 'test_edges', '_sample_row', 'WF')
    _sample_row, WF = m._sample_row, m.WF
    row = _sample_row()
    row["state"] = "今天天气不错。"
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("反解不出来" in item for item in missing)


#: 2 家同款——输入：反解答了一个不存在的判据编号。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_TRACE_SKIPS_AN_UNKNOWN_TAG_ID_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'adblock': ('不认识的判据编号',),
    'tracescore': ('不认识的信号编号',),
}


@pytest.mark.parametrize("name", TEST_TRACE_SKIPS_AN_UNKNOWN_TAG_ID_VALUES)
def test_trace_skips_an_unknown_tag_id(name: str) -> None:
    """输入：反解答了一个不存在的判据编号。输出：无。

    预期行为：记成「没判成」——那是反解格式没对齐，不是这条数据错。
    """
    _v0, = TEST_TRACE_SKIPS_AN_UNKNOWN_TAG_ID_VALUES[name]
    m = module_of(name, 'test_judges', 'faithful', 'CASE', 'judges', '_UnknownIdBackend')
    faithful, CASE, judges, _UnknownIdBackend = m.faithful, m.CASE, m.judges, m._UnknownIdBackend
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(_UnknownIdBackend(), "zh").judge(sample, spec)
    assert verdict.skipped and verdict.ok
    assert _v0 in verdict.reason


#: 2 家同款——输入：无。输出：无。预期行为：四条 noul，一条判据一道，答案与真值一致。
TEST_FLAG_FAMILY_IS_ONE_NOUL_PER_TAG: Final[tuple[str, ...]] = ('adblock', 'tracescore')


@pytest.mark.parametrize("name", TEST_FLAG_FAMILY_IS_ONE_NOUL_PER_TAG)
def test_flag_family_is_one_noul_per_tag(name: str) -> None:
    """输入：无。输出：无。预期行为：四条 noul，一条判据一道，答案与真值一致。
    """
    m = module_of(name, 'test_questions', 'CASE', 'features', 'questions', 'locales')
    CASE, features, questions, locales = m.CASE, m.features, m.questions, m.locales
    plan = features.plan_for(CASE, 0)
    out = questions.build(plan, locales.load("zh"), 0)
    flags = [item for item in out if item.family == "flag"]
    assert len(flags) == len(features.TAG_IDS)
    for tag, item in zip(features.TAG_IDS, flags, strict=True):
        assert item.question["type"] == "noul"
        assert item.answer is (tag in plan.hits)


#: 2 家同款——输入：全空间 8 种命中组合。输出：无。
TEST_EVERY_COMBINATION_IS_REACHABLE_BY_ITS_PACKET: Final[tuple[str, ...]] = ('alertrisk', 'depcheck')


@pytest.mark.parametrize("name", TEST_EVERY_COMBINATION_IS_REACHABLE_BY_ITS_PACKET)
def test_every_combination_is_reachable_by_its_packet(name: str) -> None:
    """输入：全空间 8 种命中组合。输出：无。

    预期行为：`packet_of` 造出来的记录按字面重算，得到的正好是那一组——两条路径（计划 / 文本）
    在这一层就已经对得上，后面 `check_draft` 与审计才有意义。
    """
    m = module_of(name, 'test_features', 'features', 'packet_of')
    features, packet_of = m.features, m.packet_of
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        assert features.hits_of(packet_of(tags)) == value, value


#: 2 家同款——输入：五个等级。输出：无。预期行为：最后一档吃掉 4 / 5 两级。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_ACT_FOR_COLLAPSES_THE_TOP_LEVELS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'alertrisk': ('monitor', 'ticket', 'patch_now', 'isolate', 'isolate'),
    'depcheck': ('approve', 'watch', 'review', 'replace', 'replace'),
}


@pytest.mark.parametrize("name", TEST_ACT_FOR_COLLAPSES_THE_TOP_LEVELS_VALUES)
def test_act_for_collapses_the_top_levels(name: str) -> None:
    """输入：五个等级。输出：无。预期行为：最后一档吃掉 4 / 5 两级。
    """
    _v0, _v1, _v2, _v3, _v4 = TEST_ACT_FOR_COLLAPSES_THE_TOP_LEVELS_VALUES[name]
    m = module_of(name, 'test_features', 'features')
    features = m.features
    assert [features.act_for(level) for level in features.LEVELS] == [
        _v0,
        _v1,
        _v2,
        _v3,
        _v4,
    ]


#: 3 家同款——输入：带围栏 / 多行 / 空那句话的 state。输出：无。预期行为：都判不通过。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_HYGIENE_CATCHES_LEFTOVERS_MULTILINE_AND_EMPTY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'browser': ('【用户请求】', '```\n【用户请求】', 'request', 'request'),
    'compact': ('【本块内容】', '```\n【本块内容】', 'content', 'content'),
    'routing': ('【工单正文】', '```\n【工单正文】', 'body', 'body'),
}


@pytest.mark.parametrize("name", TEST_HYGIENE_CATCHES_LEFTOVERS_MULTILINE_AND_EMPTY_VALUES)
def test_hygiene_catches_leftovers_multiline_and_empty(name: str) -> None:
    """输入：带围栏 / 多行 / 空那句话的 state。输出：无。预期行为：都判不通过。
    """
    _v0, _v1, _v2, _v3 = TEST_HYGIENE_CATCHES_LEFTOVERS_MULTILINE_AND_EMPTY_VALUES[name]
    m = module_of(name, 'test_judges', 'judges', 'sample', 'state_of')
    judges, sample, state_of = m.judges, m.sample, m.state_of
    judge = judges.HygieneJudge("zh")
    assert not judge.judge(sample(state_of().replace(_v0, _v1))).ok
    assert not judge.judge(sample(state_of({_v2: "第一行\n第二行"}))).ok
    assert not judge.judge(sample(state_of({_v3: "   "}))).ok


#: 2 家同款——输入：一条规格。输出：无。预期行为：四道题、族名一一对应、题型正确。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_BUILD_RETURNS_ONE_QUESTION_PER_FAMILY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'cachereuse': ('reuse_band', 'reuse_half'),
    'siftrank': ('cut_size', 'stable'),
}


@pytest.mark.parametrize("name", TEST_BUILD_RETURNS_ONE_QUESTION_PER_FAMILY_VALUES)
def test_build_returns_one_question_per_family(name: str) -> None:
    """输入：一条规格。输出：无。预期行为：四道题、族名一一对应、题型正确。
    """
    _v0, _v1 = TEST_BUILD_RETURNS_ONE_QUESTION_PER_FAMILY_VALUES[name]
    m = module_of(name, 'test_questions', 'LOC', 'questions', '_make_spec')
    LOC, questions, _make_spec = m.LOC, m.questions, m._make_spec
    built = questions.build(_make_spec(1), LOC, 0, random.Random(0))
    assert [item.family for item in built] == list(questions.FAMILIES)
    types = {item.family: item.question["type"] for item in built}
    for family in questions.CHOICE_FAMILIES:
        assert types[family] == "choice"
    assert types[_v0] == "score"
    assert types[_v1] == "noul"


#: 3 家同款——输入：兑现计划的样本。输出：无。预期行为：通过。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FRAME_ACCEPTS_A_FAITHFUL_SAMPLE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'citation': ('一致',),
    'guardrail': ('骨架',),
    'posttriage': ('骨架',),
}


@pytest.mark.parametrize("name", TEST_FRAME_ACCEPTS_A_FAITHFUL_SAMPLE_VALUES)
def test_frame_accepts_a_faithful_sample(name: str) -> None:
    """输入：兑现计划的样本。输出：无。预期行为：通过。
    """
    _v0, = TEST_FRAME_ACCEPTS_A_FAITHFUL_SAMPLE_VALUES[name]
    m = module_of(name, 'test_judges', 'faithful', 'CASE', 'judges')
    faithful, CASE, judges = m.faithful, m.CASE, m.judges
    spec, sample = faithful(CASE)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert _v0 in verdict.reason


#: 2 家同款——输入：两份语言文件。输出：无。预期行为：标签原文互不重叠（合并表才不炸）。
TEST_LANGUAGES_DO_NOT_SHARE_LABELS: Final[tuple[str, ...]] = ('clipcut', 'planstep')


@pytest.mark.parametrize("name", TEST_LANGUAGES_DO_NOT_SHARE_LABELS)
def test_languages_do_not_share_labels(name: str) -> None:
    """输入：两份语言文件。输出：无。预期行为：标签原文互不重叠（合并表才不炸）。
    """
    m = module_of(name, 'test_locales', 'parse', 'locales')
    parse, locales = m.parse, m.locales
    zh = parse.tables_of(locales.load("zh"))
    en = parse.tables_of(locales.load("en"))
    assert not set(zh.request) & set(en.request)
    assert not set(zh.sections) & set(en.sections)
    merged = parse.merged_tables()
    assert set(merged.request) == set(zh.request) | set(en.request)
    assert set(merged.sections) == set(zh.sections) | set(en.sections)


#: 2 家同款——输入：一条规格。输出：无。预期行为：四道题、族名一一对应、题型正确。（与 `test_build_returns_one_question_per_family` 同名不同形：字面量个数不一样，拆成两份）
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_BUILD_RETURNS_ONE_QUESTION_PER_FAMILY_2_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'clipcut': ('keep_count', 'first_kept', 'blocked_any', 'dup_any'),
    'quoteread': ('tier', 'leader', 'alert', 'divergent'),
}


@pytest.mark.parametrize("name", TEST_BUILD_RETURNS_ONE_QUESTION_PER_FAMILY_2_VALUES)
def test_build_returns_one_question_per_family_2(name: str) -> None:
    """输入：一条规格。输出：无。预期行为：四道题、族名一一对应、题型正确。
    """
    _v0, _v1, _v2, _v3 = TEST_BUILD_RETURNS_ONE_QUESTION_PER_FAMILY_2_VALUES[name]
    m = module_of(name, 'test_questions', 'LOC', 'questions', '_make_spec')
    LOC, questions, _make_spec = m.LOC, m.questions, m._make_spec
    built = questions.build(_make_spec(1), LOC, 0, random.Random(0))
    assert [item.family for item in built] == list(questions.FAMILIES)
    types = {item.family: item.question["type"] for item in built}
    assert types[_v0] == "score"
    assert types[_v1] == "choice"
    assert types[_v2] == "noul"
    assert types[_v3] == "noul"


#: 2 家同款——输入：两语言 `dims` 表。输出：无。
TEST_PROMPT_ONLY_COVERS_THE_TEXT_DIMENSIONS: Final[tuple[str, ...]] = ('compact', 'routing')


@pytest.mark.parametrize("name", TEST_PROMPT_ONLY_COVERS_THE_TEXT_DIMENSIONS)
def test_prompt_only_covers_the_text_dimensions(name: str) -> None:
    """输入：两语言 `dims` 表。输出：无。

    预期行为：`dims` 里只有由文本承载的两维——多一维就会让提示词把「本块位置」那一行
    管的事也写进正文，两边一冲突这道题就有歧义了（范式 §9.13）。
    """
    m = module_of(name, 'test_locales', 'locales', 'render')
    locales, render = m.locales, m.render
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["dims"]) == list(render.DIM_ORDER)


#: 3 家同款——输入：带围栏 / 多行 / 空修法的 state。输出：无。预期行为：都判不通过。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_HYGIENE_CATCHES_LEFTOVERS_AND_MULTILINE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'findtriage': ('【审查发现】', '```\n【审查发现】', 'finding', 'fix'),
    'search': ('【标题】', '```\n【标题】', 'title', 'title'),
    'sourceselect': ('【查询】', '```\n【查询】', 'query', 'query'),
}


@pytest.mark.parametrize("name", TEST_HYGIENE_CATCHES_LEFTOVERS_AND_MULTILINE_VALUES)
def test_hygiene_catches_leftovers_and_multiline(name: str) -> None:
    """输入：带围栏 / 多行 / 空修法的 state。输出：无。预期行为：都判不通过。
    """
    _v0, _v1, _v2, _v3 = TEST_HYGIENE_CATCHES_LEFTOVERS_AND_MULTILINE_VALUES[name]
    m = module_of(name, 'test_judges', 'judges', 'sample', 'state_of')
    judges, sample, state_of = m.judges, m.sample, m.state_of
    judge = judges.HygieneJudge("zh")
    assert not judge.judge(sample(state_of().replace(_v0, _v1))).ok
    assert not judge.judge(sample(state_of({_v2: "第一行\n第二行"}))).ok
    assert not judge.judge(sample(state_of({_v3: "   "}))).ok


#: 2 家同款——输入：英文 state。输出：Verdict。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_HYGIENE_JUDGE_READS_THE_MARKER_FROM_THE_LOCALE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'modelroute': ('request', 'summarize the log'),
    'verify': ('claim', 'fixed the flaky retry'),
}


@pytest.mark.parametrize("name", TEST_HYGIENE_JUDGE_READS_THE_MARKER_FROM_THE_LOCALE_VALUES)
def test_hygiene_judge_reads_the_marker_from_the_locale(name: str) -> None:
    """输入：英文 state。输出：Verdict。

    预期行为：认得出 `[Request]` 那一段，代码里没写死中文——换个语言照样认。
    """
    _v0, _v1 = TEST_HYGIENE_JUDGE_READS_THE_MARKER_FROM_THE_LOCALE_VALUES[name]
    m = module_of(name, 'test_judges', 'CASE', 'locales', 'render', 'judges_mod')
    CASE, locales, render, judges_mod = m.CASE, m.locales, m.render, m.judges_mod
    loc = locales.load("en")
    state = render.to_state({_v0: _v1}, CASE, loc)
    assert judges_mod.HygieneJudge("en").judge({"state": state}).ok is True
    assert judges_mod.HygieneJudge("en").judge({"state": state + "\n```"}).ok is False


#: 2 家同款——输入：一条规格。输出：无。预期行为：四道题、族名一一对应、题型正确。（与 `test_build_returns_one_question_per_family` 同名不同形：字面量个数不一样，拆成两份）
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_BUILD_RETURNS_ONE_QUESTION_PER_FAMILY_3_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'planstep': (2, 'remaining', 'score', 'next', 'rounds', 'unlocks'),
    'sentiment': ('review', 'tone', 'choice', 'pivot', 'drop', 'rebound'),
}


@pytest.mark.parametrize("name", TEST_BUILD_RETURNS_ONE_QUESTION_PER_FAMILY_3_VALUES)
def test_build_returns_one_question_per_family_3(name: str) -> None:
    """输入：一条规格。输出：无。预期行为：四道题、族名一一对应、题型正确。
    """
    _v0, _v1, _v2, _v3, _v4, _v5 = TEST_BUILD_RETURNS_ONE_QUESTION_PER_FAMILY_3_VALUES[name]
    m = module_of(name, 'test_questions', 'LOC', 'questions', '_make_spec')
    LOC, questions, _make_spec = m.LOC, m.questions, m._make_spec
    built = questions.build(_make_spec(3, _v0), LOC, 0, random.Random(0))
    assert [item.family for item in built] == list(questions.FAMILIES)
    types = {item.family: item.question["type"] for item in built}
    assert types[_v1] == _v2
    assert types[_v3] == "choice"
    assert types[_v4] == "score"
    assert types[_v5] == "noul"


#: 2 家同款——输入：缺右括号 / 格数不对 / 认不出的格 的局面行。输出：无。预期行为：都报人话。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_PARSE_CELLS_REPORTS_HARD_DAMAGE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'sim': ('1 前进 · 耗 1 · 收益 2 · 冷却 0', '[1] 前进 · 耗 1 · 收益 2', 'action', '[1] 不存在 · 耗 1 · 收益 2 · 冷却 0'),
    'skillpick': ('1 读代码 · 只读 · 代码', '[1] 读代码 · 只读', 'tag', '[1] 不存在 · 只读 · 代码'),
}


@pytest.mark.parametrize("name", TEST_PARSE_CELLS_REPORTS_HARD_DAMAGE_VALUES)
def test_parse_cells_reports_hard_damage(name: str) -> None:
    """输入：缺右括号 / 格数不对 / 认不出的格 的局面行。输出：无。预期行为：都报人话。
    """
    _v0, _v1, _v2, _v3 = TEST_PARSE_CELLS_REPORTS_HARD_DAMAGE_VALUES[name]
    m = module_of(name, 'test_render_parse', 'parse', 'locales')
    parse, locales = m.parse, m.locales
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(ValueError, match="右方括号"):
        parse._parse_cells(_v0, labels)
    with pytest.raises(ValueError, match="格"):
        parse._parse_cells(_v1, labels)
    with pytest.raises(ValueError, match=_v2):
        parse._parse_cells(_v3, labels)


#: 2 家同款——输入：整个 train 切分跑一遍。输出：RunResult。预期行为：零拒绝、各族的答案均衡全绿。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FULL_TRAIN_RUN_IS_BALANCED_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'adblock': ('p4-full',),
    'guardrail': ('g4-full',),
}


@pytest.mark.parametrize("name", TEST_FULL_TRAIN_RUN_IS_BALANCED_VALUES)
def test_full_train_run_is_balanced(name: str, tmp_path: Path) -> None:
    """输入：整个 train 切分跑一遍。输出：RunResult。预期行为：零拒绝、各族的答案均衡全绿。
    """
    _v0, = TEST_FULL_TRAIN_RUN_IS_BALANCED_VALUES[name]
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    fake_backend = conftest.FakeBackend
    m = module_of(name, 'test_pipeline', 'FakeBackend', 'make_config', 'cases_of')
    FakeBackend, make_config, cases_of = m.FakeBackend, m.make_config, m.cases_of
    result = pipeline.run(make_config(tmp_path, _v0, fake_backend(), limit=0))
    assert result.rejected == []
    assert result.accepted == len(cases_of("train", limit=0))
    assert [item["ok"] for item in result.stats["balance"]] == [True] * len(
        result.stats["balance"]
    )


#: 2 家同款——输入：zh / en 各一段 state。输出：无。预期行为：各自认回自己的语言，认不出的给 None。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_DETECT_LOCALE_TELLS_THE_TWO_LANGUAGES_APART_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'alertrisk': ('R1', 'R1'),
    'depcheck': ('D1', 'D1'),
}


@pytest.mark.parametrize("name", TEST_DETECT_LOCALE_TELLS_THE_TWO_LANGUAGES_APART_VALUES)
def test_detect_locale_tells_the_two_languages_apart(name: str) -> None:
    """输入：zh / en 各一段 state。输出：无。预期行为：各自认回自己的语言，认不出的给 None。
    """
    _v0, _v1 = TEST_DETECT_LOCALE_TELLS_THE_TWO_LANGUAGES_APART_VALUES[name]
    m = module_of(name, 'test_render', 'render', 'draft_of', 'packet_of', 'locales')
    render, draft_of, packet_of, locales = m.render, m.draft_of, m.packet_of, m.locales
    zh = render.to_state(draft_of(packet_of((_v0,), "zh")), locales.load("zh"))
    en = render.to_state(draft_of(packet_of((_v1,), "en")), locales.load("en"))
    assert render.detect_locale(zh) == "zh"
    assert render.detect_locale(en) == "en"
    assert render.detect_locale("这里什么抬头都没有") is None
    assert render.detect_locale(None) is None


#: 2 家同款——输入：两个判别器名 / 一个不存在的名字。输出：无。预期行为：都报人话。
TEST_MAKE_JUDGES_BUILDS_BOTH_AND_REJECTS_UNKNOWN_NAMES: Final[tuple[str, ...]] = ('bench', 'elementpick')


@pytest.mark.parametrize("name", TEST_MAKE_JUDGES_BUILDS_BOTH_AND_REJECTS_UNKNOWN_NAMES)
def test_make_judges_builds_both_and_rejects_unknown_names(name: str) -> None:
    """输入：两个判别器名 / 一个不存在的名字。输出：无。预期行为：都报人话。
    """
    m = module_of(name, 'test_api', 'WF', 'judges')
    WF, judges = m.WF, m.judges
    built = WF.make_judges(("roundtrip", "label"), "zh")
    assert [judge.name for judge in built] == ["roundtrip", "label"]
    assert all(
        isinstance(judge, (judges.RoundtripJudge, judges.LabelJudge)) for judge in built
    )
    with pytest.raises(KeyError, match="未知判别器"):
        WF.make_judges(("nope",), "zh")


#: 2 家同款——输入：两语言文案。输出：无。预期行为：每题族三种问法，`pick` 在三个编号上都取得到。
TEST_EVERY_FAMILY_HAS_THREE_PHRASINGS: Final[tuple[str, ...]] = ('browser', 'findtriage')


@pytest.mark.parametrize("name", TEST_EVERY_FAMILY_HAS_THREE_PHRASINGS)
def test_every_family_has_three_phrasings(name: str) -> None:
    """输入：两语言文案。输出：无。预期行为：每题族三种问法，`pick` 在三个编号上都取得到。
    """
    m = module_of(name, 'test_locales', 'locales', 'questions')
    locales, questions = m.locales, m.questions
    for lang in locales.available():
        loc = locales.load(lang)
        for family in questions.FAMILIES:
            assert len(loc["fam"][family]["q"]) >= 3
            for phrasing in (0, 1, 2):
                assert locales.pick(loc["fam"][family]["q"], phrasing)


#: 2 家同款——输入：choice 的答案编号越界。输出：无。预期行为：报「编号越界」，不抛 IndexError。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_CHECK_SAMPLE_CATCHES_AN_OUT_OF_RANGE_INDEX_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'clipcut': ('first_kept',),
    'quoteread': ('leader',),
}


@pytest.mark.parametrize("name", TEST_CHECK_SAMPLE_CATCHES_AN_OUT_OF_RANGE_INDEX_VALUES)
def test_check_sample_catches_an_out_of_range_index(name: str) -> None:
    """输入：choice 的答案编号越界。输出：无。预期行为：报「编号越界」，不抛 IndexError。
    """
    _v0, = TEST_CHECK_SAMPLE_CATCHES_AN_OUT_OF_RANGE_INDEX_VALUES[name]
    m = module_of(name, 'test_judges', 'sample_of', 'facts_of', 'LOC', '_make_spec', 'judges')
    sample_of, facts_of, LOC, _make_spec, judges = m.sample_of, m.facts_of, m.LOC, m._make_spec, m.judges
    sample = sample_of(_make_spec(4))
    facts = facts_of(sample)
    qid = next(name for name in sample["questions"] if name.startswith(_v0))
    sample["answers"][qid]["choice"] = 99
    problems = judges.check_sample(sample, facts, LOC)
    assert problems and "编号越界" in problems[0]


#: 2 家同款——输入：答案被改错的样本。输出：Verdict。预期行为：判不通过，理由点名那一题。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_LABEL_JUDGE_CATCHES_A_WRONG_ANSWER_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'clipcut': ('dup_any',),
    'quoteread': ('divergent',),
}


@pytest.mark.parametrize("name", TEST_LABEL_JUDGE_CATCHES_A_WRONG_ANSWER_VALUES)
def test_label_judge_catches_a_wrong_answer(name: str) -> None:
    """输入：答案被改错的样本。输出：Verdict。预期行为：判不通过，理由点名那一题。
    """
    _v0, = TEST_LABEL_JUDGE_CATCHES_A_WRONG_ANSWER_VALUES[name]
    m = module_of(name, 'test_judges', 'sample_of', '_make_spec', 'judges')
    sample_of, _make_spec, judges = m.sample_of, m._make_spec, m.judges
    sample = sample_of(_make_spec(10))
    qid = next(name for name in sample["questions"] if name.startswith(_v0))
    sample["answers"][qid]["noul"] = not bool(sample["answers"][qid]["noul"])
    verdict = judges.LabelJudge(locale="zh").judge(sample)
    assert verdict.ok is False
    assert qid in verdict.reason


#: 2 家同款——输入：带堆栈的日志（多行）。输出：无。预期行为：多行原样保住。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_MULTILINE_CONTENT_ROUND_TRIPS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'logtriage': ('\n  at handler.py:41\n  at main.py:7',),
    'tracescore': ('\n$ pytest -q\n2 failed, 8 passed\n  at handler.py:41',),
}


@pytest.mark.parametrize("name", TEST_MULTILINE_CONTENT_ROUND_TRIPS_VALUES)
def test_multiline_content_round_trips(name: str) -> None:
    """输入：带堆栈的日志（多行）。输出：无。预期行为：多行原样保住。
    """
    _v0, = TEST_MULTILINE_CONTENT_ROUND_TRIPS_VALUES[name]
    m = module_of(name, 'test_render', 'locales', 'text_of', 'CASE', 'render', 'features')
    locales, text_of, CASE, render, features = m.locales, m.text_of, m.CASE, m.render, m.features
    loc = locales.load("zh")
    text = text_of(CASE) + _v0
    state = render.to_state(render.ContentDraft(text=text), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.text == text
    assert parsed.tags == features.TAG_IDS


#: 2 家同款——问法编号越界要回绕，不许报错也不许给出空串。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_PHRASING_INDEX_WRAPS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'modelroute': ('route', 1, 1),
    'shell': ('verdict', 0, 0),
}


@pytest.mark.parametrize(("name", "lang"), _lang_pairs(TEST_PHRASING_INDEX_WRAPS_VALUES, "test_questions", "LOCALES"))
def test_phrasing_index_wraps(name: str, lang: str) -> None:
    """问法编号越界要回绕，不许报错也不许给出空串。
    """
    _v0, _v1, _v2 = TEST_PHRASING_INDEX_WRAPS_VALUES[name]
    m = module_of(name, 'test_questions', 'LOCALES', 'locales', 'CASE', 'questions')
    LOCALES, locales, CASE, questions = m.LOCALES, m.locales, m.CASE, m.questions
    loc = locales.load(lang)
    count = len(loc["fam"][_v0]["q"])
    assert questions.build(CASE, loc, phrasing=count + 1)[_v1].question["instructions"] == (
        questions.build(CASE, loc, phrasing=1)[_v2].question["instructions"]
    )


#: 2 家同款——输入：同一个 spec 出两次题。输出：无。预期行为：逐字段一致，RNG 不参与。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_BUILD_IS_PURE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'planstep': (3,),
    'sentiment': ('neighbors',),
}


@pytest.mark.parametrize("name", TEST_BUILD_IS_PURE_VALUES)
def test_build_is_pure(name: str) -> None:
    """输入：同一个 spec 出两次题。输出：无。预期行为：逐字段一致，RNG 不参与。
    """
    _v0, = TEST_BUILD_IS_PURE_VALUES[name]
    m = module_of(name, 'test_questions', '_make_spec', 'LOC', 'questions')
    _make_spec, LOC, questions = m._make_spec, m.LOC, m.questions
    spec = _make_spec(7, _v0)
    first = questions.build(spec, LOC, 1, random.Random(1))
    second = questions.build(spec, LOC, 1, random.Random(999))
    assert [item.family for item in first] == [item.family for item in second]
    assert [item.question for item in first] == [item.question for item in second]
    assert [item.answer for item in first] == [item.answer for item in second]


#: 3 家同款——输入：正文里正常提到段落名。输出：无。预期行为：不算泄漏——只认「段落名 + 冒号」开头。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_LABEL_LEAK_ONLY_READS_THE_LINE_START_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'modelroute': ('这个请求：请把日志压成摘要', '', '请求：把日志压成摘要', '请求'),
    'toolgate': ('调用意图：读一下', '调用意图', '他说的调用意图是读一下', ''),
    'verify': ('这个声明：把空指针修掉了', '', '声明：把空指针修掉了', '声明'),
}


@pytest.mark.parametrize("name", TEST_LABEL_LEAK_ONLY_READS_THE_LINE_START_VALUES)
def test_label_leak_only_reads_the_line_start(name: str) -> None:
    """输入：正文里正常提到段落名。输出：无。预期行为：不算泄漏——只认「段落名 + 冒号」开头。
    """
    _v0, _v1, _v2, _v3 = TEST_LABEL_LEAK_ONLY_READS_THE_LINE_START_VALUES[name]
    m = module_of(name, 'test_render', 'locales', 'render')
    locales, render = m.locales, m.render
    loc = locales.load("zh")
    assert render.label_leak(_v0, loc) == _v1
    assert render.label_leak(_v2, loc) == _v3


#: 2 家同款——输入：把键名写在句子中间的记录。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_MISSING_KEYS_NEEDS_THE_COLON_ON_ITS_OWN_LINE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'alertrisk': ('impact: hosts=3\nexposure: 见 rule: 那一行\nexploit: auth=required\ndetail: 一句', 'rule'),
    'depcheck': ('license: MIT\nstatus: 见 package: 那一行\nadvisories: count=0\nversion: 1.2.3\nnote: 一句', 'package'),
}


@pytest.mark.parametrize("name", TEST_MISSING_KEYS_NEEDS_THE_COLON_ON_ITS_OWN_LINE_VALUES)
def test_missing_keys_needs_the_colon_on_its_own_line(name: str) -> None:
    """输入：把键名写在句子中间的记录。输出：无。

    预期行为：算缺——判据是钉在**行首**的，键不单独成行时那三条正则连定位都定不到。
    """
    _v0, _v1 = TEST_MISSING_KEYS_NEEDS_THE_COLON_ON_ITS_OWN_LINE_VALUES[name]
    m = module_of(name, 'test_features', 'features')
    features = m.features
    text = _v0
    assert _v1 in features.missing_keys(text)


#: 2 家同款——输入：以标签开头但没有分隔符的一行。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FIELD_VALUE_NEEDS_A_SEPARATOR_RIGHT_AFTER_THE_LABEL_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'elementpick': ('元素清单', '元素', '动作'),
    'skillpick': ('技能选择单', '技能', '标签'),
}


@pytest.mark.parametrize("name", TEST_FIELD_VALUE_NEEDS_A_SEPARATOR_RIGHT_AFTER_THE_LABEL_VALUES)
def test_field_value_needs_a_separator_right_after_the_label(name: str) -> None:
    """输入：以标签开头但没有分隔符的一行。输出：无。

    预期行为：不算这一格——标题行确实以「元素」开头，收紧才不会误认。
    """
    _v0, _v1, _v2 = TEST_FIELD_VALUE_NEEDS_A_SEPARATOR_RIGHT_AFTER_THE_LABEL_VALUES[name]
    m = module_of(name, 'test_render_parse', 'read_field_value')
    read_field_value = m.read_field_value
    assert read_field_value(_v0, (_v1,)) is None
    assert read_field_value("今天天气不错。", (_v2,)) is None


#: 2 家同款——输入：改动记录被整段删掉。输出：无。预期行为：判不通过。
TEST_HYGIENE_CATCHES_A_MISSING_TABLE: Final[tuple[str, ...]] = ('findtriage', 'sourceselect')


@pytest.mark.parametrize("name", TEST_HYGIENE_CATCHES_A_MISSING_TABLE)
def test_hygiene_catches_a_missing_table(name: str) -> None:
    """输入：改动记录被整段删掉。输出：无。预期行为：判不通过。
    """
    m = module_of(name, 'test_judges', 'state_of', 'PLAN', 'locales', 'render', 'sample', 'judges')
    state_of, PLAN, locales, render, sample, judges = m.state_of, m.PLAN, m.locales, m.render, m.sample, m.judges
    loc = locales.load("zh")
    state = state_of()
    for line in render.table_lines(PLAN, loc):
        state = state.replace(f"{line}\n", "")
    assert not judges.HygieneJudge("zh").judge(sample(state)).ok


#: 2 家同款——输入：两条不同规格。输出：无。预期行为：渲染与反解的 id 各自稳定、互不相同。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_TASK_IDS_ARE_UNIQUE_AND_RECOMPUTABLE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'findtriage': ('findtriage-train-0000003-render', 'findtriage-train-0000003-trace'),
    'sourceselect': ('sourceselect-train-0000003-render', 'sourceselect-train-0000003-trace'),
}


@pytest.mark.parametrize("name", TEST_TASK_IDS_ARE_UNIQUE_AND_RECOMPUTABLE_VALUES)
def test_task_ids_are_unique_and_recomputable(name: str) -> None:
    """输入：两条不同规格。输出：无。预期行为：渲染与反解的 id 各自稳定、互不相同。
    """
    _v0, _v1 = TEST_TASK_IDS_ARE_UNIQUE_AND_RECOMPUTABLE_VALUES[name]
    m = module_of(name, 'test_spec', 'make')
    make = m.make
    one, two = make(seq=3), make(seq=4)
    assert one.render_task_id == _v0
    assert one.trace_task_id == _v1
    assert one.render_task_id != two.render_task_id
    assert one.render_task_id != one.trace_task_id


#: 2 家同款——输入：认不出正文的 state。输出：无。预期行为：不通过（条款表与正文都不对）。
TEST_FRAME_REJECTS_A_STATE_WITHOUT_A_CONTENT_BLOCK: Final[tuple[str, ...]] = ('guardrail', 'posttriage')


@pytest.mark.parametrize("name", TEST_FRAME_REJECTS_A_STATE_WITHOUT_A_CONTENT_BLOCK)
def test_frame_rejects_a_state_without_a_content_block(name: str) -> None:
    """输入：认不出正文的 state。输出：无。预期行为：不通过（条款表与正文都不对）。
    """
    m = module_of(name, 'test_judges', 'faithful', 'CASE', 'spec_for', 'judges')
    faithful, CASE, spec_for, judges = m.faithful, m.CASE, m.spec_for, m.judges
    _, sample = faithful(CASE)
    sample["state"] = "这里什么抬头都没有"
    verdict = judges.FrameJudge("zh").judge(sample, spec_for(CASE))
    assert not verdict.ok
    assert "正文" in verdict.reason


#: 2 家同款——输入：引文被反解装进 JSON 数组字符串（范式 §9.88）。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_QUOTE_HOLDS_ACCEPTS_A_QUOTE_WRAPPED_IN_A_JSON_ARRAY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'guardrail': ('P1', 'P1'),
    'posttriage': ('S1', 'S1'),
}


@pytest.mark.parametrize("name", TEST_QUOTE_HOLDS_ACCEPTS_A_QUOTE_WRAPPED_IN_A_JSON_ARRAY_VALUES)
def test_quote_holds_accepts_a_quote_wrapped_in_a_json_array(name: str) -> None:
    """输入：引文被反解装进 JSON 数组字符串（范式 §9.88）。输出：无。

    预期行为：拆成候选之后照旧算站得住——模型把「抄几段」当成了「交数组」只是打包方式不同，
    不该把一条真命中丢成误报（存量那 20 条就是这么丢的）。
    """
    _v0, _v1 = TEST_QUOTE_HOLDS_ACCEPTS_A_QUOTE_WRAPPED_IN_A_JSON_ARRAY_VALUES[name]
    m = module_of(name, 'test_judges', 'judges', 'text_of', 'CASE', 'quote_for')
    judges, text_of, CASE, quote_for = m.judges, m.text_of, m.CASE, m.quote_for
    assert judges.quote_holds(_v0, json.dumps([quote_for(_v1)]), text_of(CASE))


#: 2 家同款——state 不是文本就放过：那不是「格式残留」，是上游接线错了。
TEST_HYGIENE_JUDGE_SKIPS_A_NON_TEXT_STATE: Final[tuple[str, ...]] = ('modelroute', 'shell')


@pytest.mark.parametrize("name", TEST_HYGIENE_JUDGE_SKIPS_A_NON_TEXT_STATE)
def test_hygiene_judge_skips_a_non_text_state(name: str) -> None:
    """state 不是文本就放过：那不是「格式残留」，是上游接线错了。

    输入：state 是个字典。输出：Verdict。预期行为：通过且理由写明「不是文本」——
    本 workflow 的渲染器只产字符串，这条分支是产物侧的兜底（与 mail 的 hygiene 同口径）。
    """
    m = module_of(name, 'test_judges', 'judges_mod')
    judges_mod = m.judges_mod
    verdict = judges_mod.HygieneJudge("zh").judge({"state": {"not": "text"}})
    assert verdict.ok is True
    assert "不是文本" in verdict.reason


#: 2 家同款——输入：真样本。输出：无。预期行为：零问题——判别器不误判。
TEST_CHECK_SAMPLE_PASSES_FOR_GENUINE_SAMPLES: Final[tuple[str, ...]] = ('planstep', 'sentiment')


@pytest.mark.parametrize("name", TEST_CHECK_SAMPLE_PASSES_FOR_GENUINE_SAMPLES)
@pytest.mark.parametrize("lang", ("zh", "en"))
def test_check_sample_passes_for_genuine_samples(name: str, lang: str) -> None:
    """输入：真样本。输出：无。预期行为：零问题——判别器不误判。
    """
    m = module_of(name, 'test_judges', 'ALL_CASES', 'locales', 'sample_of', 'judges', 'parsed_of')
    ALL_CASES, locales, sample_of, judges, parsed_of = m.ALL_CASES, m.locales, m.sample_of, m.judges, m.parsed_of
    loc = locales.load(lang)
    for case in ALL_CASES[:12]:
        sample = sample_of(case, lang)
        assert judges.check_sample(sample, judges.facts_of(parsed_of(sample, lang)), loc) == []


#: 2 家同款——输入：抬头被改。输出：无。预期行为：请求号对不上就报出来。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_ROUNDTRIP_DIFF_CATCHES_A_BROKEN_REQUEST_ID_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'planstep': (1, 'P-000000'),
    'sentiment': ('neighbors', 'E-000000'),
}


@pytest.mark.parametrize("name", TEST_ROUNDTRIP_DIFF_CATCHES_A_BROKEN_REQUEST_ID_VALUES)
def test_roundtrip_diff_catches_a_broken_request_id(name: str) -> None:
    """输入：抬头被改。输出：无。预期行为：请求号对不上就报出来。
    """
    _v0, _v1 = TEST_ROUNDTRIP_DIFF_CATCHES_A_BROKEN_REQUEST_ID_VALUES[name]
    m = module_of(name, 'test_judges', '_make_spec', 'locales', 'parse', 'render', 'judges')
    _make_spec, locales, parse, render, judges = m._make_spec, m.locales, m.parse, m.render, m.judges
    loc = locales.load("zh")
    spec = _make_spec(2, _v0)
    text = render.build(spec, loc).replace(spec.query_id, _v1)
    parsed = parse.parse_state(text, parse.tables_of(loc))
    assert any("请求号" in item for item in judges.roundtrip_diff(spec, parsed, loc))


#: 2 家同款——输入：一份语言文件。输出：无。预期行为：判据原文非空、每一条都不是空串。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_RULES_LINES_ARE_NOT_EMPTY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'planstep': (4,),
    'sentiment': (5,),
}


@pytest.mark.parametrize(("name", "lang"), _lang_pairs(TEST_RULES_LINES_ARE_NOT_EMPTY_VALUES, "test_locales", "LANGS"))
def test_rules_lines_are_not_empty(name: str, lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：判据原文非空、每一条都不是空串。
    """
    _v0, = TEST_RULES_LINES_ARE_NOT_EMPTY_VALUES[name]
    m = module_of(name, 'test_locales', 'LANGS', 'locales')
    LANGS, locales = m.LANGS, m.locales
    lines = locales.load(lang)["rules"]["lines"]
    assert len(lines) >= _v0, lang
    for line in lines:
        assert str(line).strip(), lang


#: 2 家同款——输入：含表名的文本。输出：无。预期行为：命中，且大小写不敏感。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_BANNED_ECHO_FINDS_TABLE_NAMES_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'review': ('+ x = 1  # 见改动清单', '改动清单', '+ x = 1', '+ x = 1  # see Change Manifest', 'change manifest'),
    'rubric': ('line one 见评分单', '评分单', 'line one', 'see the Score Sheet', 'score sheet'),
}


@pytest.mark.parametrize("name", TEST_BANNED_ECHO_FINDS_TABLE_NAMES_VALUES)
def test_banned_echo_finds_table_names(name: str) -> None:
    """输入：含表名的文本。输出：无。预期行为：命中，且大小写不敏感。
    """
    _v0, _v1, _v2, _v3, _v4 = TEST_BANNED_ECHO_FINDS_TABLE_NAMES_VALUES[name]
    m = module_of(name, 'test_render', 'locales', 'render')
    locales, render = m.locales, m.render
    zh = locales.load("zh")
    en = locales.load("en")
    assert render.banned_echo(_v0, zh) == [_v1]
    assert render.banned_echo(_v2, zh) == []
    assert render.banned_echo(_v3, en) == [_v4]


#: 2 家同款——输入：完整记录、少两行的记录。输出：无。预期行为：只报缺掉的那几个键。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_MISSING_KEYS_FINDS_ONLY_ABSENT_ONES_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'alertrisk': ('rule', 'exploit', 'rule', 'exploit'),
    'depcheck': ('package', 'status', 'package', 'status'),
}


@pytest.mark.parametrize("name", TEST_MISSING_KEYS_FINDS_ONLY_ABSENT_ONES_VALUES)
def test_missing_keys_finds_only_absent_ones(name: str) -> None:
    """输入：完整记录、少两行的记录。输出：无。预期行为：只报缺掉的那几个键。
    """
    _v0, _v1, _v2, _v3 = TEST_MISSING_KEYS_FINDS_ONLY_ABSENT_ONES_VALUES[name]
    m = module_of(name, 'test_features', 'render_lines', 'BASE_LINES', 'features')
    render_lines, BASE_LINES, features = m.render_lines, m.BASE_LINES, m.features
    full = render_lines(BASE_LINES)
    assert features.missing_keys(full) == ()
    lines = {key: value for key, value in BASE_LINES.items() if key not in {_v0, _v1}}
    assert set(features.missing_keys(render_lines(lines))) == {_v2, _v3}


#: 3 家同款——输入：没有规格。输出：无。预期行为：记「没判成」。
TEST_TRACE_SKIPS_WITHOUT_A_SPEC: Final[tuple[str, ...]] = ('alertrisk', 'causechain', 'citation')


@pytest.mark.parametrize("name", TEST_TRACE_SKIPS_WITHOUT_A_SPEC)
def test_trace_skips_without_a_spec(name: str) -> None:
    """输入：没有规格。输出：无。预期行为：记「没判成」。
    """
    m = module_of(name, 'test_judges', 'faithful', 'CASE', 'judges', 'FakeBackend')
    faithful, CASE, judges, FakeBackend = m.faithful, m.CASE, m.judges, m.FakeBackend
    _, sample = faithful(CASE)
    assert judges.TraceJudge(FakeBackend(), "zh").judge(sample, None).skipped


#: 2 家同款——输入：两个描述一样的选项 / 答案不在选项里。输出：无。预期行为：都报人话。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_CHOICE_REJECTS_DUPLICATE_DESCRIPTIONS_AND_UNKNOWN_ANSWERS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'bench': ('第 1 个', '第 2 个'),
    'sim': ('甲', '乙'),
}


@pytest.mark.parametrize("name", TEST_CHOICE_REJECTS_DUPLICATE_DESCRIPTIONS_AND_UNKNOWN_ANSWERS_VALUES)
def test_choice_rejects_duplicate_descriptions_and_unknown_answers(name: str) -> None:
    """输入：两个描述一样的选项 / 答案不在选项里。输出：无。预期行为：都报人话。
    """
    _v0, _v1 = TEST_CHOICE_REJECTS_DUPLICATE_DESCRIPTIONS_AND_UNKNOWN_ANSWERS_VALUES[name]
    m = module_of(name, 'test_questions', 'questions')
    questions = m.questions
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("best", "题面", {"a": "同一个", "b": "同一个"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("best", "题面", {"a": _v0, "b": _v1}, "c")


#: 3 家同款——输入：没有真值规格 / 没有后端。输出：无。预期行为：都记成跳过。
TEST_TRACE_SKIPS_WITHOUT_SPEC_OR_BACKEND: Final[tuple[str, ...]] = ('browser', 'findtriage', 'sourceselect')


@pytest.mark.parametrize("name", TEST_TRACE_SKIPS_WITHOUT_SPEC_OR_BACKEND)
def test_trace_skips_without_spec_or_backend(name: str) -> None:
    """输入：没有真值规格 / 没有后端。输出：无。预期行为：都记成跳过。
    """
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    fake_backend = conftest.FakeBackend
    m = module_of(name, 'test_judges', 'FakeBackend', 'sample', 'spec', 'state_of', 'judges')
    FakeBackend, sample, spec, state_of, judges = m.FakeBackend, m.sample, m.spec, m.state_of, m.judges
    assert judges.TraceJudge(fake_backend(), "zh").judge(sample(state_of())).skipped
    assert judges.TraceJudge(None, "zh").judge(sample(state_of()), spec()).skipped


#: 3 家同款——输入：任意特征。输出：无。预期行为：七个题族对任何组合都问得出来。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FAMILIES_FOR_RETURNS_THE_FULL_TABLE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'browser': (1024,),
    'compact': (1024,),
    'routing': (2048,),
}


@pytest.mark.parametrize("name", TEST_FAMILIES_FOR_RETURNS_THE_FULL_TABLE_VALUES)
def test_families_for_returns_the_full_table(name: str) -> None:
    """输入：任意特征。输出：无。预期行为：七个题族对任何组合都问得出来。
    """
    _v0, = TEST_FAMILIES_FOR_RETURNS_THE_FULL_TABLE_VALUES[name]
    m = module_of(name, 'test_questions', 'features', 'questions')
    features, questions = m.features, m.questions
    for tags in features.build_cases(_v0):
        assert questions.families_for(tags) == questions.FAMILIES


#: 2 家同款——输入：编号越界的 choice 答案。输出：无。预期行为：报越界而不是抛 IndexError。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_LABEL_JUDGE_CATCHES_AN_OUT_OF_RANGE_INDEX_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'cachereuse': (12, 'diff_kind'),
    'match': (8, 'variant_count'),
}


@pytest.mark.parametrize("name", TEST_LABEL_JUDGE_CATCHES_AN_OUT_OF_RANGE_INDEX_VALUES)
def test_label_judge_catches_an_out_of_range_index(name: str) -> None:
    """输入：编号越界的 choice 答案。输出：无。预期行为：报越界而不是抛 IndexError。
    """
    _v0, _v1 = TEST_LABEL_JUDGE_CATCHES_AN_OUT_OF_RANGE_INDEX_VALUES[name]
    m = module_of(name, 'test_judges', 'sample_of', '_make_spec', 'judges')
    sample_of, _make_spec, judges = m.sample_of, m._make_spec, m.judges
    sample = sample_of(_make_spec(_v0))
    sample["answers"][_v1]["choice"] = 99
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "越界" in verdict.reason


#: 2 家同款——输入：题目题型被改成契约外的值。输出：无。预期行为：报未知题型。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_LABEL_JUDGE_CATCHES_AN_UNKNOWN_QUESTION_TYPE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'cachereuse': (13, 'diff_kind', 'ranking'),
    'match': (9, 'band', 'multi'),
}


@pytest.mark.parametrize("name", TEST_LABEL_JUDGE_CATCHES_AN_UNKNOWN_QUESTION_TYPE_VALUES)
def test_label_judge_catches_an_unknown_question_type(name: str) -> None:
    """输入：题目题型被改成契约外的值。输出：无。预期行为：报未知题型。
    """
    _v0, _v1, _v2 = TEST_LABEL_JUDGE_CATCHES_AN_UNKNOWN_QUESTION_TYPE_VALUES[name]
    m = module_of(name, 'test_judges', 'sample_of', '_make_spec', 'judges')
    sample_of, _make_spec, judges = m.sample_of, m._make_spec, m.judges
    sample = sample_of(_make_spec(_v0))
    sample["questions"][_v1]["type"] = _v2
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "未知题型" in verdict.reason


#: 2 家同款——输入：反解说反话的假后端。输出：无。预期行为：不通过，理由里两种说法都列出来。
TEST_TRACE_REJECTS_A_LYING_BACKEND: Final[tuple[str, ...]] = ('citation', 'rerank')


@pytest.mark.parametrize("name", TEST_TRACE_REJECTS_A_LYING_BACKEND)
def test_trace_rejects_a_lying_backend(name: str) -> None:
    """输入：反解说反话的假后端。输出：无。预期行为：不通过，理由里两种说法都列出来。
    """
    m = module_of(name, 'test_judges', 'faithful', 'CASE', 'judges', 'FakeBackend')
    faithful, CASE, judges, FakeBackend = m.faithful, m.CASE, m.judges, m.FakeBackend
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "计划" in verdict.reason and "反解" in verdict.reason


#: 3 家同款——输入：只有两行的 state。输出：无。预期行为：报「太短」，不给一个空片段表。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_PARSE_REJECTS_A_SHORT_STATE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'clipcut': ('片段筛选请求号 K-000000\n[片段表]\n',),
    'quoteread': ('行情读数请求号 Q-000000\n[读数表]\n',),
    'siftrank': ('排序请求号：S-000000\n排序提示：随便\n',),
}


@pytest.mark.parametrize("name", TEST_PARSE_REJECTS_A_SHORT_STATE_VALUES)
def test_parse_rejects_a_short_state(name: str) -> None:
    """输入：只有两行的 state。输出：无。预期行为：报「太短」，不给一个空片段表。
    """
    _v0, = TEST_PARSE_REJECTS_A_SHORT_STATE_VALUES[name]
    m = module_of(name, 'test_render_parse', 'parse')
    parse = m.parse
    with pytest.raises(ValueError, match="太短"):
        parse.parse_state(_v0, parse.merged_tables())


#: 2 家同款——输入：题号。输出：无。预期行为：去掉 `_2` 这类后缀，也认得带下划线的族名。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FAMILY_OF_STRIPS_DEDUPE_SUFFIXES_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'compact': ('rule', 'rule', 'rule_2', 'rule', 'pin_overrode', 'pin_overrode', 'pin_overrode_2', 'pin_overrode'),
    'routing': ('queue', 'queue', 'queue_2', 'queue', 'needs_human', 'needs_human', 'needs_human_2', 'needs_human'),
}


@pytest.mark.parametrize("name", TEST_FAMILY_OF_STRIPS_DEDUPE_SUFFIXES_VALUES)
def test_family_of_strips_dedupe_suffixes(name: str) -> None:
    """输入：题号。输出：无。预期行为：去掉 `_2` 这类后缀，也认得带下划线的族名。
    """
    _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7 = TEST_FAMILY_OF_STRIPS_DEDUPE_SUFFIXES_VALUES[name]
    m = module_of(name, 'test_audit', 'audit_mod')
    audit_mod = m.audit_mod
    assert audit_mod._family_of(_v0) == _v1
    assert audit_mod._family_of(_v2) == _v3
    assert audit_mod._family_of(_v4) == _v5
    assert audit_mod._family_of(_v6) == _v7


#: 2 家同款——输入：三种坏草稿。输出：无。预期行为：都抛 ValueError。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_CHECK_DRAFT_REJECTS_BLANK_MULTILINE_AND_FENCES_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'compact': ('content', 'content', 'content'),
    'routing': ('body', 'body', 'body'),
}


@pytest.mark.parametrize("name", TEST_CHECK_DRAFT_REJECTS_BLANK_MULTILINE_AND_FENCES_VALUES)
def test_check_draft_rejects_blank_multiline_and_fences(name: str) -> None:
    """输入：三种坏草稿。输出：无。预期行为：都抛 ValueError。
    """
    _v0, _v1, _v2 = TEST_CHECK_DRAFT_REJECTS_BLANK_MULTILINE_AND_FENCES_VALUES[name]
    m = module_of(name, 'test_render', 'locales', 'GOOD', 'render')
    locales, GOOD, render = m.locales, m.GOOD, m.render
    loc = locales.load("zh")
    for bad in ("   ", f"{GOOD[_v0]}\n第二行", f"```{GOOD[_v1]}```"):
        with pytest.raises(ValueError):
            render.check_draft({_v2: bad}, loc)


#: 3 家同款——输入：两行同名的会话。输出：无。预期行为：只取第一行，取不到给空串。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_LINE_VALUE_READS_THE_FIRST_MATCHING_LINE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'debugstep': ('Note: 第一行\nNote: 第二行', 'Note: 第一行', 'stack'),
    'sqlreview': ('note: 第一行\nnote: 第二行', 'note: 第一行', 'statement'),
    'subqa': ('Note: 第一行\nNote: 第二行', 'Note: 第一行', 'target'),
}


@pytest.mark.parametrize("name", TEST_LINE_VALUE_READS_THE_FIRST_MATCHING_LINE_VALUES)
def test_line_value_reads_the_first_matching_line(name: str) -> None:
    """输入：两行同名的会话。输出：无。预期行为：只取第一行，取不到给空串。
    """
    _v0, _v1, _v2 = TEST_LINE_VALUE_READS_THE_FIRST_MATCHING_LINE_VALUES[name]
    m = module_of(name, 'test_features', 'features')
    features = m.features
    assert features.line_value(_v0, "note") == "第一行"
    assert features.line_value(_v1, _v2) == ""


#: 2 家同款——输入：描述重复的选项 / 答案不在选项里。输出：无。预期行为：都报人话。（与 `test_choice_rejects_duplicate_descriptions_and_unknown_answers` 同名不同形：字面量个数不一样，拆成两份）
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_CHOICE_REJECTS_DUPLICATE_DESCRIPTIONS_AND_UNKNOWN_ANSWERS_2_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'elementpick': ('target', 'target'),
    'skillpick': ('pick', 'pick'),
}


@pytest.mark.parametrize("name", TEST_CHOICE_REJECTS_DUPLICATE_DESCRIPTIONS_AND_UNKNOWN_ANSWERS_2_VALUES)
def test_choice_rejects_duplicate_descriptions_and_unknown_answers_2(name: str) -> None:
    """输入：描述重复的选项 / 答案不在选项里。输出：无。预期行为：都报人话。
    """
    _v0, _v1 = TEST_CHOICE_REJECTS_DUPLICATE_DESCRIPTIONS_AND_UNKNOWN_ANSWERS_2_VALUES[name]
    m = module_of(name, 'test_questions', 'questions')
    questions = m.questions
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice(_v0, "题面", {"s1": "一样", "s2": "一样"}, "s1")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice(_v1, "题面", {"s1": "甲"}, "s2")


#: 3 家同款——输入：题号。输出：无。预期行为：去掉 `_2` 这类后缀。（与 `test_family_of_strips_dedupe_suffixes` 同名不同形：字面量个数不一样，拆成两份）
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FAMILY_OF_STRIPS_DEDUPE_SUFFIXES_2_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'findtriage': ('triage', 'triage', 'triage_2', 'triage'),
    'search': ('rule', 'rule', 'rule_2', 'rule'),
    'sourceselect': ('pick', 'pick', 'pick_2', 'pick'),
}


@pytest.mark.parametrize("name", TEST_FAMILY_OF_STRIPS_DEDUPE_SUFFIXES_2_VALUES)
def test_family_of_strips_dedupe_suffixes_2(name: str) -> None:
    """输入：题号。输出：无。预期行为：去掉 `_2` 这类后缀。
    """
    _v0, _v1, _v2, _v3 = TEST_FAMILY_OF_STRIPS_DEDUPE_SUFFIXES_2_VALUES[name]
    m = module_of(name, 'test_audit', 'audit_mod')
    audit_mod = m.audit_mod
    assert audit_mod._family_of(_v0) == _v1
    assert audit_mod._family_of(_v2) == _v3


#: 2 家同款——输入：序号断档 / 改动种类不认识的表。输出：无。预期行为：整张表判为解不出来。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_ROWS_OF_REJECTS_A_BROKEN_TABLE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'findtriage': ('untouched', 'maybe'),
    'sourceselect': ('official', 'vip'),
}


@pytest.mark.parametrize("name", TEST_ROWS_OF_REJECTS_A_BROKEN_TABLE_VALUES)
def test_rows_of_rejects_a_broken_table(name: str) -> None:
    """输入：序号断档 / 改动种类不认识的表。输出：无。预期行为：整张表判为解不出来。
    """
    _v0, _v1 = TEST_ROWS_OF_REJECTS_A_BROKEN_TABLE_VALUES[name]
    m = module_of(name, 'test_render_parse', 'GOOD', 'PLAN', 'locales', 'render')
    GOOD, PLAN, locales, render = m.GOOD, m.PLAN, m.locales, m.render
    loc = locales.load("zh")
    state = render.to_state(GOOD, PLAN, loc)
    assert render.rows_of(state.replace("- 2 | ", "- 7 | "), loc) == ()
    assert render.rows_of(state.replace(_v0, _v1), loc) == ()


#: 2 家同款——输入：清单行挂错了 tags。输出：无。预期行为：判不通过，理由点出清单行。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FRAME_CATCHES_A_SWAPPED_MANIFEST_ROW_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'review': ('- source | modified | small', '- config | modified | small', '清单行'),
    'rubric': ('- short | prose | none', '- long | prose | none', '材料清单'),
}


@pytest.mark.parametrize("name", TEST_FRAME_CATCHES_A_SWAPPED_MANIFEST_ROW_VALUES)
def test_frame_catches_a_swapped_manifest_row(name: str) -> None:
    """输入：清单行挂错了 tags。输出：无。预期行为：判不通过，理由点出清单行。
    """
    _v0, _v1, _v2 = TEST_FRAME_CATCHES_A_SWAPPED_MANIFEST_ROW_VALUES[name]
    m = module_of(name, 'test_judges', 'sample', 'spec', 'state_of', 'judges')
    sample, spec, state_of, judges = m.sample, m.spec, m.state_of, m.judges
    broken = state_of().replace(_v0, _v1)
    verdict = judges.FrameJudge("zh").judge(sample(broken), spec())
    assert not verdict.ok
    assert _v2 in verdict.reason


#: 2 家同款——输入：门控表少了一行。输出：无。预期行为：判不通过。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FRAME_CATCHES_A_MISSING_TABLE_ROW_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'review': ('- 0 | allow | allow | allow', '门控表'),
    'rubric': ('- mid | accept | reject\n', '处置表'),
}


@pytest.mark.parametrize("name", TEST_FRAME_CATCHES_A_MISSING_TABLE_ROW_VALUES)
def test_frame_catches_a_missing_table_row(name: str) -> None:
    """输入：门控表少了一行。输出：无。预期行为：判不通过。
    """
    _v0, _v1 = TEST_FRAME_CATCHES_A_MISSING_TABLE_ROW_VALUES[name]
    m = module_of(name, 'test_judges', 'sample', 'spec', 'state_of', 'judges')
    sample, spec, state_of, judges = m.sample, m.spec, m.state_of, m.judges
    broken = state_of().replace(_v0, "")
    verdict = judges.FrameJudge("zh").judge(sample(broken), spec())
    assert not verdict.ok
    assert _v1 in verdict.reason


#: 2 家同款——输入：反解草稿缺字段。输出：无。预期行为：记成「没判成」。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_TRACE_RECORDS_A_MISSING_FIELD_AS_SKIPPED_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'review': ('secret',),
    'rubric': ('placeholders',),
}


@pytest.mark.parametrize("name", TEST_TRACE_RECORDS_A_MISSING_FIELD_AS_SKIPPED_VALUES)
def test_trace_records_a_missing_field_as_skipped(name: str) -> None:
    """输入：反解草稿缺字段。输出：无。预期行为：记成「没判成」。
    """
    _v0, = TEST_TRACE_RECORDS_A_MISSING_FIELD_AS_SKIPPED_VALUES[name]
    m = module_of(name, 'test_judges', 'FakeBackend', 'sample', 'spec', 'state_of', 'judges')
    FakeBackend, sample, spec, state_of, judges = m.FakeBackend, m.sample, m.spec, m.state_of, m.judges
    backend = FakeBackend(trace_data={_v0: "yes"})
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert verdict.ok and verdict.skipped
    assert "缺字段" in verdict.reason


#: 2 家同款——输入：两个描述一样的选项。输出：无。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_CHOICE_REJECTS_DUPLICATE_OPTION_DESCRIPTIONS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'functions': ('选项描述重复', 'f', '题', '同一个', '同一个'),
    'receiptcat': ('重复', 'cat', 'q', '同', '同'),
}


@pytest.mark.parametrize("name", TEST_CHOICE_REJECTS_DUPLICATE_OPTION_DESCRIPTIONS_VALUES)
def test_choice_rejects_duplicate_option_descriptions(name: str) -> None:
    """输入：两个描述一样的选项。输出：无。

    预期行为：报错——选项看起来一样，考生没法分辨，是构造 bug。
    """
    _v0, _v1, _v2, _v3, _v4 = TEST_CHOICE_REJECTS_DUPLICATE_OPTION_DESCRIPTIONS_VALUES[name]
    m = module_of(name, 'test_questions', 'questions')
    questions = m.questions
    with pytest.raises(ValueError, match=_v0):
        questions._choice(_v1, _v2, {"a": _v3, "b": _v4}, "a")


#: 2 家同款——输入：全空间 8 种命中组合。输出：无。预期行为：两种表示互相还原。
TEST_HITS_OF_AND_PARSE_HITS_ROUND_TRIP: Final[tuple[str, ...]] = ('alertrisk', 'depcheck')


@pytest.mark.parametrize("name", TEST_HITS_OF_AND_PARSE_HITS_ROUND_TRIP)
def test_hits_of_and_parse_hits_round_trip(name: str) -> None:
    """输入：全空间 8 种命中组合。输出：无。预期行为：两种表示互相还原。
    """
    m = module_of(name, 'test_features', 'features', 'packet_of')
    features, packet_of = m.features, m.packet_of
    for value in features.HIT_SETS:
        assert features.hits_of(packet_of(features.parse_hits(value))) == value
    assert features.parse_hits("") == ()


#: 2 家同款——输入：含 `detail:` 的记录。输出：无。预期行为：取到冒号后面的内容，取不到给空串。（与 `test_line_value_reads_the_first_matching_line` 同名不同形：字面量个数不一样，拆成两份）
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_LINE_VALUE_READS_THE_FIRST_MATCHING_LINE_2_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'alertrisk': ('detail', '研发笔记本'),
    'depcheck': ('note', '内部工具链'),
}


@pytest.mark.parametrize("name", TEST_LINE_VALUE_READS_THE_FIRST_MATCHING_LINE_2_VALUES)
def test_line_value_reads_the_first_matching_line_2(name: str) -> None:
    """输入：含 `detail:` 的记录。输出：无。预期行为：取到冒号后面的内容，取不到给空串。
    """
    _v0, _v1 = TEST_LINE_VALUE_READS_THE_FIRST_MATCHING_LINE_2_VALUES[name]
    m = module_of(name, 'test_features', 'render_lines', 'BASE_LINES', 'features')
    render_lines, BASE_LINES, features = m.render_lines, m.BASE_LINES, m.features
    text = render_lines(BASE_LINES)
    assert features.line_value(text, _v0).startswith(_v1)
    assert features.line_value(text, "nope") == ""


#: 2 家同款——输入：任意命中组合。输出：无。预期行为：五行齐全（夹具自己不许漏行，否则测试会假绿）。
TEST_LINES_OF_KEEPS_EVERY_KEY: Final[tuple[str, ...]] = ('alertrisk', 'depcheck')


@pytest.mark.parametrize("name", TEST_LINES_OF_KEEPS_EVERY_KEY)
def test_lines_of_keeps_every_key(name: str) -> None:
    """输入：任意命中组合。输出：无。预期行为：五行齐全（夹具自己不许漏行，否则测试会假绿）。
    """
    m = module_of(name, 'test_render', 'features', 'lines_of')
    features, lines_of = m.features, m.lines_of
    for value in features.HIT_SETS:
        lines = lines_of(features.parse_hits(value))
        assert sorted(lines) == sorted(features.KEY_LINES)


#: 2 家同款——输入：四行记录。输出：无。预期行为：报人话（少一行说明构造漏了一行）。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_BUILD_REJECTS_A_WRONG_NUMBER_OF_ROWS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'bench': ('记录应该是', 4),
    'sim': ('局面应该是', 5),
}


@pytest.mark.parametrize("name", TEST_BUILD_REJECTS_A_WRONG_NUMBER_OF_ROWS_VALUES)
def test_build_rejects_a_wrong_number_of_rows(name: str) -> None:
    """输入：四行记录。输出：无。预期行为：报人话（少一行说明构造漏了一行）。
    """
    _v0, _v1 = TEST_BUILD_REJECTS_A_WRONG_NUMBER_OF_ROWS_VALUES[name]
    m = module_of(name, 'test_questions', '_make_spec', 'LOC', 'questions')
    _make_spec, LOC, questions = m._make_spec, m.LOC, m.questions
    spec = _make_spec()
    with pytest.raises(ValueError, match=_v0):
        questions.build(spec.rows[:_v1], spec, LOC, 0)


#: 2 家同款——输入：表外的题族。输出：无。预期行为：报人话，不静默给一个空答案。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_EXPECTED_OF_REPORTS_AN_UNKNOWN_FAMILY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'clipcut': ('keep_amount',),
    'quoteread': ('tier_level',),
}


@pytest.mark.parametrize("name", TEST_EXPECTED_OF_REPORTS_AN_UNKNOWN_FAMILY_VALUES)
def test_expected_of_reports_an_unknown_family(name: str) -> None:
    """输入：表外的题族。输出：无。预期行为：报人话，不静默给一个空答案。
    """
    _v0, = TEST_EXPECTED_OF_REPORTS_AN_UNKNOWN_FAMILY_VALUES[name]
    m = module_of(name, 'test_judges', 'facts_of', 'sample_of', 'LOC', '_make_spec', 'judges')
    facts_of, sample_of, LOC, _make_spec, judges = m.facts_of, m.sample_of, m.LOC, m._make_spec, m.judges
    facts = facts_of(sample_of(_make_spec(1)))
    with pytest.raises(ValueError, match="未知题族"):
        judges.expected_of(_v0, facts, LOC)


#: 2 家同款——输入：处置表被整段删掉。输出：无。预期行为：判不通过。
TEST_FRAME_CATCHES_A_MISSING_RULE_TABLE: Final[tuple[str, ...]] = ('compact', 'search')


@pytest.mark.parametrize("name", TEST_FRAME_CATCHES_A_MISSING_RULE_TABLE)
def test_frame_catches_a_missing_rule_table(name: str) -> None:
    """输入：处置表被整段删掉。输出：无。预期行为：判不通过。
    """
    m = module_of(name, 'test_judges', 'locales', 'state_of', 'sample', 'spec', 'judges')
    locales, state_of, sample, spec, judges = m.locales, m.state_of, m.sample, m.spec, m.judges
    loc = locales.load("zh")
    head = state_of().split(locales.header(loc, loc["meta"]["rules_label"]))[0]
    assert not judges.FrameJudge("zh").judge(sample(head), spec()).ok


#: 2 家同款——输入：一个没写进 locale 的取值。输出：无。预期行为：抛 KeyError，不静默给空串。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FEATURE_LINE_RAISES_ON_A_MISSING_PHRASE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'compact': ('kind',),
    'routing': ('issue',),
}


@pytest.mark.parametrize("name", TEST_FEATURE_LINE_RAISES_ON_A_MISSING_PHRASE_VALUES)
def test_feature_line_raises_on_a_missing_phrase(name: str) -> None:
    """输入：一个没写进 locale 的取值。输出：无。预期行为：抛 KeyError，不静默给空串。
    """
    _v0, = TEST_FEATURE_LINE_RAISES_ON_A_MISSING_PHRASE_VALUES[name]
    m = module_of(name, 'test_locales', 'locales')
    locales = m.locales
    loc = locales.load("zh")
    with pytest.raises(KeyError):
        locales.feature_line(loc, _v0, "bogus")


#: 2 家同款——输入：一个少填槽的模板。输出：无。预期行为：报人话（不然发出去的是读不懂的题）。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FILL_REPORTS_LEFTOVER_PLACEHOLDERS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'elementpick': ('第 {n} 个：{role}',),
    'skillpick': ('第 {n} 个：{tag}',),
}


@pytest.mark.parametrize("name", TEST_FILL_REPORTS_LEFTOVER_PLACEHOLDERS_VALUES)
def test_fill_reports_leftover_placeholders(name: str) -> None:
    """输入：一个少填槽的模板。输出：无。预期行为：报人话（不然发出去的是读不懂的题）。
    """
    _v0, = TEST_FILL_REPORTS_LEFTOVER_PLACEHOLDERS_VALUES[name]
    m = module_of(name, 'test_questions', 'questions')
    questions = m.questions
    assert questions._fill("第 {n} 个", n="1") == "第 1 个"
    with pytest.raises(ValueError, match="占位符"):
        questions._fill(_v0, n="1")


#: 2 家同款——输入：两门语言用不同的四格分隔符。输出：无。预期行为：报人话。
TEST_MERGED_REJECTS_MISMATCHED_SEPARATORS: Final[tuple[str, ...]] = ('elementpick', 'skillpick')


@pytest.mark.parametrize("name", TEST_MERGED_REJECTS_MISMATCHED_SEPARATORS)
def test_merged_rejects_mismatched_separators(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言用不同的四格分隔符。输出：无。预期行为：报人话。
    """
    m = module_of(name, 'test_render_parse', '_patched_merged', 'parse')
    _patched_merged, parse = m._patched_merged, m.parse
    _patched_merged(monkeypatch, lambda en, zh: en["meta"].update({"cell_sep": " | "}))
    with pytest.raises(ValueError, match="分隔符必须一致"):
        parse.merged()


#: 2 家同款——输入：一段带换行的正文。输出：无。预期行为：换行原样保留，不被当成结构行。（与 `test_multiline_content_round_trips` 同名不同形：字面量个数不一样，拆成两份）
TEST_MULTILINE_CONTENT_ROUND_TRIPS_2: Final[tuple[str, ...]] = ('guardrail', 'promptinject')


@pytest.mark.parametrize("name", TEST_MULTILINE_CONTENT_ROUND_TRIPS_2)
def test_multiline_content_round_trips_2(name: str) -> None:
    """输入：一段带换行的正文。输出：无。预期行为：换行原样保留，不被当成结构行。
    """
    m = module_of(name, 'test_render', 'BASE', 'render', 'locales')
    BASE, render, locales = m.BASE, m.render, m.locales
    text = BASE + "\n第二行也没有任何形状"
    state = render.to_state(render.ContentDraft(text=text), locales.load("zh"))
    assert render.parse_state(state).text == text


#: 2 家同款——输入：全空间 48 组 case 的忠实日志。输出：无。预期行为：一条都不报错。
TEST_CHECK_DRAFT_ACCEPTS_EVERY_FAITHFUL_TEXT: Final[tuple[str, ...]] = ('logtriage', 'tracescore')


@pytest.mark.parametrize("name", TEST_CHECK_DRAFT_ACCEPTS_EVERY_FAITHFUL_TEXT)
def test_check_draft_accepts_every_faithful_text(name: str) -> None:
    """输入：全空间 48 组 case 的忠实日志。输出：无。预期行为：一条都不报错。
    """
    m = module_of(name, 'test_render', 'locales', 'features', 'render', 'text_of')
    locales, features, render, text_of = m.locales, m.features, m.render, m.text_of
    loc = locales.load("zh")
    for case in features.build_cases(512):
        render.check_draft(render.ContentDraft(text=text_of(case)), case, loc)


#: 2 家同款——请求单行、无残留 → 通过。
TEST_HYGIENE_JUDGE_ACCEPTS_A_CLEAN_STATE: Final[tuple[str, ...]] = ('modelroute', 'verify')


@pytest.mark.parametrize("name", TEST_HYGIENE_JUDGE_ACCEPTS_A_CLEAN_STATE)
def test_hygiene_judge_accepts_a_clean_state(name: str) -> None:
    """请求单行、无残留 → 通过。
    """
    m = module_of(name, 'test_judges', 'state_for', 'judges_mod')
    state_for, judges_mod = m.state_for, m.judges_mod
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state_for()})
    assert verdict.ok is True, verdict.reason
    assert verdict.skipped is False


#: 2 家同款——输入：请求开头又写了一遍「请求：」。输出：Verdict。预期行为：判不通过并点名段落名。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_HYGIENE_JUDGE_FLAGS_A_LEAKED_SECTION_NAME_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'modelroute': ('请求：帮我改代码',),
    'verify': ('声明：把空指针修掉了',),
}


@pytest.mark.parametrize("name", TEST_HYGIENE_JUDGE_FLAGS_A_LEAKED_SECTION_NAME_VALUES)
def test_hygiene_judge_flags_a_leaked_section_name(name: str) -> None:
    """输入：请求开头又写了一遍「请求：」。输出：Verdict。预期行为：判不通过并点名段落名。
    """
    _v0, = TEST_HYGIENE_JUDGE_FLAGS_A_LEAKED_SECTION_NAME_VALUES[name]
    m = module_of(name, 'test_judges', 'state_for', 'judges_mod')
    state_for, judges_mod = m.state_for, m.judges_mod
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state_for(_v0)})
    assert verdict.ok is False
    assert "段落名" in verdict.reason


#: 2 家同款——输入：区块抬头之前就有一行内容。输出：无。预期行为：报错——那说明渲染漏了抬头。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_PARSE_REPORTS_CONTENT_BEFORE_ANY_SECTION_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'planstep': ('计划推进请求号 P-TEST\n  1. 需求梳理 | 待开始 | 无前置\n[推进规则]\n- 规则一\n',),
    'sentiment': ('情绪弧线请求号 E-TEST\n乱写一行\n[对话]\n  1. 客户 | 还不错\n',),
}


@pytest.mark.parametrize("name", TEST_PARSE_REPORTS_CONTENT_BEFORE_ANY_SECTION_VALUES)
def test_parse_reports_content_before_any_section(name: str) -> None:
    """输入：区块抬头之前就有一行内容。输出：无。预期行为：报错——那说明渲染漏了抬头。
    """
    _v0, = TEST_PARSE_REPORTS_CONTENT_BEFORE_ANY_SECTION_VALUES[name]
    m = module_of(name, 'test_render_parse', 'parse')
    parse = m.parse
    text = _v0
    with pytest.raises(ValueError, match="任何区块抬头之前"):
        parse.parse_state(text, parse.merged_tables())


#: 2 家同款——输入：答对的假后端。输出：无。预期行为：通过，并把反解结果记进 raw。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_TRACE_PASSES_WHEN_THE_MODEL_READS_IT_RIGHT_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'review': ('secret', 'debug'),
    'rubric': ('placeholders', 'citations'),
}


@pytest.mark.parametrize("name", TEST_TRACE_PASSES_WHEN_THE_MODEL_READS_IT_RIGHT_VALUES)
def test_trace_passes_when_the_model_reads_it_right(name: str) -> None:
    """输入：答对的假后端。输出：无。预期行为：通过，并把反解结果记进 raw。
    """
    _v0, _v1 = TEST_TRACE_PASSES_WHEN_THE_MODEL_READS_IT_RIGHT_VALUES[name]
    m = module_of(name, 'test_judges', 'sample', 'spec', 'state_of', 'judges', 'FakeBackend')
    sample, spec, state_of, judges, FakeBackend = m.sample, m.spec, m.state_of, m.judges, m.FakeBackend
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample(state_of()), spec())
    assert verdict.ok and not verdict.skipped
    assert verdict.raw["traced"] == {_v0: "yes", _v1: "no"}


#: 2 家同款——输入：词元不在取值表里的清单行。输出：无。预期行为：当解不出来。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_MANIFEST_OF_REJECTS_UNKNOWN_TOKENS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'review': ('【改动内容】\n+ x = 1\n【改动清单】（列名）\n- source | modified | huge',),
    'rubric': ('【待评材料】\nline one\n【材料清单】（列名）\n- huge | prose | none',),
}


@pytest.mark.parametrize("name", TEST_MANIFEST_OF_REJECTS_UNKNOWN_TOKENS_VALUES)
def test_manifest_of_rejects_unknown_tokens(name: str) -> None:
    """输入：词元不在取值表里的清单行。输出：无。预期行为：当解不出来。
    """
    _v0, = TEST_MANIFEST_OF_REJECTS_UNKNOWN_TOKENS_VALUES[name]
    m = module_of(name, 'test_render', 'locales', 'render')
    locales, render = m.locales, m.render
    loc = locales.load("zh")
    state = _v0
    assert render.manifest_of(state, loc) is None


#: 2 家同款——输入：只有两格的清单行。输出：无。预期行为：当解不出来。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_MANIFEST_OF_REJECTS_A_ROW_WITH_THE_WRONG_CELL_COUNT_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'review': ('【改动内容】\n+ x = 1\n【改动清单】（列名）\n- source | modified',),
    'rubric': ('【待评材料】\nline one\n【材料清单】（列名）\n- short | prose',),
}


@pytest.mark.parametrize("name", TEST_MANIFEST_OF_REJECTS_A_ROW_WITH_THE_WRONG_CELL_COUNT_VALUES)
def test_manifest_of_rejects_a_row_with_the_wrong_cell_count(name: str) -> None:
    """输入：只有两格的清单行。输出：无。预期行为：当解不出来。
    """
    _v0, = TEST_MANIFEST_OF_REJECTS_A_ROW_WITH_THE_WRONG_CELL_COUNT_VALUES[name]
    m = module_of(name, 'test_render', 'locales', 'render')
    locales, render = m.locales, m.render
    loc = locales.load("zh")
    state = _v0
    assert render.manifest_of(state, loc) is None


#: 3 家同款——输入：正文里混进 `<br>`。输出：无。预期行为：判不通过。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_HYGIENE_CATCHES_HTML_LINE_BREAKS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'compact': ('content', '上一句<br>下一句'),
    'review': ('body', '+ x = 1<br>'),
    'routing': ('body', '上一句<br>下一句'),
}


@pytest.mark.parametrize("name", TEST_HYGIENE_CATCHES_HTML_LINE_BREAKS_VALUES)
def test_hygiene_catches_html_line_breaks(name: str) -> None:
    """输入：正文里混进 `<br>`。输出：无。预期行为：判不通过。
    """
    _v0, _v1 = TEST_HYGIENE_CATCHES_HTML_LINE_BREAKS_VALUES[name]
    m = module_of(name, 'test_judges', 'sample', 'state_of', 'judges')
    sample, state_of, judges = m.sample, m.state_of, m.judges
    assert not judges.HygieneJudge("zh").judge(sample(state_of({_v0: _v1}))).ok


#: 2 家同款——输入：引文里多打了空白。输出：无。预期行为：空白折叠之后仍然算抄对了。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_QUOTE_HOLDS_IGNORES_WHITESPACE_DIFFERENCES_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'alertrisk': ('R3', 'R3', 'impact:   hosts=24  -- 二十四台生产机器'),
    'depcheck': ('D3', 'D3', 'advisories:   count=3'),
}


@pytest.mark.parametrize("name", TEST_QUOTE_HOLDS_IGNORES_WHITESPACE_DIFFERENCES_VALUES)
def test_quote_holds_ignores_whitespace_differences(name: str) -> None:
    """输入：引文里多打了空白。输出：无。预期行为：空白折叠之后仍然算抄对了。
    """
    _v0, _v1, _v2 = TEST_QUOTE_HOLDS_IGNORES_WHITESPACE_DIFFERENCES_VALUES[name]
    m = module_of(name, 'test_judges', 'packet_of', 'judges')
    packet_of, judges = m.packet_of, m.judges
    body = packet_of((_v0,))
    assert judges.quote_holds(_v1, _v2, body)


#: 2 家同款——输入：一个没填完的模板。输出：无。预期行为：报人话（题面少填一个槽比报错危险）。（与 `test_fill_reports_leftover_placeholders` 同名不同形：字面量个数不一样，拆成两份）
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FILL_REPORTS_LEFTOVER_PLACEHOLDERS_2_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'bench': ('第 {n} 题，第 1 个候选答对了吗？',),
    'sim': ('第 {n} 个动作现在能用吗？',),
}


@pytest.mark.parametrize("name", TEST_FILL_REPORTS_LEFTOVER_PLACEHOLDERS_2_VALUES)
def test_fill_reports_leftover_placeholders_2(name: str) -> None:
    """输入：一个没填完的模板。输出：无。预期行为：报人话（题面少填一个槽比报错危险）。
    """
    _v0, = TEST_FILL_REPORTS_LEFTOVER_PLACEHOLDERS_2_VALUES[name]
    m = module_of(name, 'test_questions', 'questions')
    questions = m.questions
    with pytest.raises(ValueError, match="占位符"):
        questions._fill(_v0, other="x")


#: 2 家同款——输入：干净 state。输出：无。预期行为：通过。
TEST_HYGIENE_PASSES_A_CLEAN_STATE: Final[tuple[str, ...]] = ('browser', 'review')


@pytest.mark.parametrize("name", TEST_HYGIENE_PASSES_A_CLEAN_STATE)
def test_hygiene_passes_a_clean_state(name: str) -> None:
    """输入：干净 state。输出：无。预期行为：通过。
    """
    m = module_of(name, 'test_judges', 'sample', 'state_of', 'judges')
    sample, state_of, judges = m.sample, m.state_of, m.judges
    verdict = judges.HygieneJudge("zh").judge(sample(state_of()))
    assert verdict.ok and not verdict.skipped


#: 2 家同款——输入：形状不对的反查表。输出：无。预期行为：报形状，不抛 AttributeError。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_TO_KEYS_REPORTS_A_BROKEN_LOOKUP_TABLE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'cachereuse': ('打开', '搜索框', 'action', 'target'),
    'clipcut': ('开场白', '高亮', 'topic', 'flag'),
}


@pytest.mark.parametrize("name", TEST_TO_KEYS_REPORTS_A_BROKEN_LOOKUP_TABLE_VALUES)
def test_to_keys_reports_a_broken_lookup_table(name: str) -> None:
    """输入：形状不对的反查表。输出：无。预期行为：报形状，不抛 AttributeError。
    """
    _v0, _v1, _v2, _v3 = TEST_TO_KEYS_REPORTS_A_BROKEN_LOOKUP_TABLE_VALUES[name]
    m = module_of(name, 'test_judges', 'judges')
    judges = m.judges
    with pytest.raises(ValueError, match="反查表形状不对"):
        judges.to_keys(((_v0, _v1),), {_v2: [], _v3: {}})


#: 2 家同款——输入：新样本。输出：无。预期行为：零问题（审计与判别器共用这一把尺）。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_CHECK_SAMPLE_IS_SILENT_ON_A_GOOD_SAMPLE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'cachereuse': (15,),
    'match': (22,),
}


@pytest.mark.parametrize("name", TEST_CHECK_SAMPLE_IS_SILENT_ON_A_GOOD_SAMPLE_VALUES)
def test_check_sample_is_silent_on_a_good_sample(name: str) -> None:
    """输入：新样本。输出：无。预期行为：零问题（审计与判别器共用这一把尺）。
    """
    _v0, = TEST_CHECK_SAMPLE_IS_SILENT_ON_A_GOOD_SAMPLE_VALUES[name]
    m = module_of(name, 'test_judges', 'sample_of', '_make_spec', 'LOC', 'judges', 'facts_of')
    sample_of, _make_spec, LOC, judges, facts_of = m.sample_of, m._make_spec, m.LOC, m.judges, m.facts_of
    sample = sample_of(_make_spec(_v0))
    assert judges.check_sample(sample, facts_of(sample), LOC) == []


#: 2 家同款——输入：不存在的题族。输出：无。预期行为：报错（新加题族忘了实现不能静默放过）。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_EXPECTED_OF_REJECTS_UNKNOWN_FAMILY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'cachereuse': (17,),
    'match': (19,),
}


@pytest.mark.parametrize("name", TEST_EXPECTED_OF_REJECTS_UNKNOWN_FAMILY_VALUES)
def test_expected_of_rejects_unknown_family(name: str) -> None:
    """输入：不存在的题族。输出：无。预期行为：报错（新加题族忘了实现不能静默放过）。
    """
    _v0, = TEST_EXPECTED_OF_REJECTS_UNKNOWN_FAMILY_VALUES[name]
    m = module_of(name, 'test_judges', 'LOC', 'judges', 'facts_of', 'sample_of', '_make_spec')
    LOC, judges, facts_of, sample_of, _make_spec = m.LOC, m.judges, m.facts_of, m.sample_of, m._make_spec
    with pytest.raises(ValueError, match="未知题族"):
        judges.expected_of("nope", facts_of(sample_of(_make_spec(_v0))), LOC)


#: 2 家同款——输入：不存在的题族。输出：无。预期行为：报错（不然会静默返回另一族的选项）。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_OPTIONS_OF_REJECTS_UNKNOWN_FAMILY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'cachereuse': ('reuse_half', 5),
    'match': ('nope', 3),
}


@pytest.mark.parametrize("name", TEST_OPTIONS_OF_REJECTS_UNKNOWN_FAMILY_VALUES)
def test_options_of_rejects_unknown_family(name: str) -> None:
    """输入：不存在的题族。输出：无。预期行为：报错（不然会静默返回另一族的选项）。
    """
    _v0, _v1 = TEST_OPTIONS_OF_REJECTS_UNKNOWN_FAMILY_VALUES[name]
    m = module_of(name, 'test_questions', 'LOC', 'questions', '_make_spec')
    LOC, questions, _make_spec = m.LOC, m.questions, m._make_spec
    with pytest.raises(ValueError, match="未知题族"):
        questions.options_of(_v0, _make_spec(_v1), LOC)


#: 2 家同款——输入：没有规格。输出：无。预期行为：记「没判成」，不硬判。
TEST_FRAME_SKIPS_WITHOUT_A_SPEC: Final[tuple[str, ...]] = ('causechain', 'citation')


@pytest.mark.parametrize("name", TEST_FRAME_SKIPS_WITHOUT_A_SPEC)
def test_frame_skips_without_a_spec(name: str) -> None:
    """输入：没有规格。输出：无。预期行为：记「没判成」，不硬判。
    """
    m = module_of(name, 'test_judges', 'faithful', 'CASE', 'judges')
    faithful, CASE, judges = m.faithful, m.CASE, m.judges
    _, sample = faithful(CASE)
    assert judges.FrameJudge("zh").judge(sample, None).skipped


#: 2 家同款——输入：两个条目同名。输出：无。预期行为：报错——反解会把两个条目认成一个。
TEST_REVERSE_REJECTS_A_DUPLICATE_NAME: Final[tuple[str, ...]] = ('clipcut', 'planstep')


@pytest.mark.parametrize("name", TEST_REVERSE_REJECTS_A_DUPLICATE_NAME)
def test_reverse_rejects_a_duplicate_name(name: str) -> None:
    """输入：两个条目同名。输出：无。预期行为：报错——反解会把两个条目认成一个。
    """
    m = module_of(name, 'test_locales', 'locales')
    locales = m.locales
    with pytest.raises(ValueError, match="同时属于"):
        locales.reverse({"a": "同名", "b": "同名"})


#: 2 家同款——输入：表外的题族。输出：无。预期行为：报人话，不返回空选项表。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_UNKNOWN_FAMILY_HAS_NO_OPTIONS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'clipcut': ('keep_count', 4),
    'quoteread': ('tier', 5),
}


@pytest.mark.parametrize("name", TEST_UNKNOWN_FAMILY_HAS_NO_OPTIONS_VALUES)
def test_unknown_family_has_no_options(name: str) -> None:
    """输入：表外的题族。输出：无。预期行为：报人话，不返回空选项表。
    """
    _v0, _v1 = TEST_UNKNOWN_FAMILY_HAS_NO_OPTIONS_VALUES[name]
    m = module_of(name, 'test_questions', 'LOC', 'questions', '_make_spec')
    LOC, questions, _make_spec = m.LOC, m.questions, m._make_spec
    with pytest.raises(ValueError, match="未知题族"):
        questions.options_of(_v0, _make_spec(_v1), LOC)


#: 2 家同款——输入：正文里又写了一遍段落名。输出：无。预期行为：判不通过。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_HYGIENE_CATCHES_A_REPEATED_SECTION_NAME_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'compact': ('content', '本块内容：刚才那句话'),
    'routing': ('body', '工单正文：刚才那句话'),
}


@pytest.mark.parametrize("name", TEST_HYGIENE_CATCHES_A_REPEATED_SECTION_NAME_VALUES)
def test_hygiene_catches_a_repeated_section_name(name: str) -> None:
    """输入：正文里又写了一遍段落名。输出：无。预期行为：判不通过。
    """
    _v0, _v1 = TEST_HYGIENE_CATCHES_A_REPEATED_SECTION_NAME_VALUES[name]
    m = module_of(name, 'test_judges', 'state_of', 'sample', 'judges')
    state_of, sample, judges = m.state_of, m.sample, m.judges
    broken = state_of({_v0: _v1})
    assert not judges.HygieneJudge("zh").judge(sample(broken)).ok


#: 2 家同款——输入：每种语言。输出：无。预期行为：四格分隔符一致（合并表只按一个分隔符切行）。
TEST_CELL_SEPARATOR_IS_SHARED_ACROSS_LANGUAGES: Final[tuple[str, ...]] = ('elementpick', 'skillpick')


@pytest.mark.parametrize(("name", "lang"), _lang_pairs(TEST_CELL_SEPARATOR_IS_SHARED_ACROSS_LANGUAGES, "test_locales", "LANGS"))
def test_cell_separator_is_shared_across_languages(name: str, lang: str) -> None:
    """输入：每种语言。输出：无。预期行为：四格分隔符一致（合并表只按一个分隔符切行）。
    """
    m = module_of(name, 'test_locales', 'LANGS', 'locales')
    LANGS, locales = m.LANGS, m.locales
    assert locales.load(lang)["meta"]["cell_sep"] == locales.load("zh")["meta"]["cell_sep"]


#: 2 家同款——输入：某一行的改动种类被改掉。输出：无。预期行为：判不通过。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_FRAME_CATCHES_A_WRONG_ROW_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'findtriage': ('added', 'untouched'),
    'sourceselect': ('official', 'community'),
}


@pytest.mark.parametrize("name", TEST_FRAME_CATCHES_A_WRONG_ROW_VALUES)
def test_frame_catches_a_wrong_row(name: str) -> None:
    """输入：某一行的改动种类被改掉。输出：无。预期行为：判不通过。
    """
    _v0, _v1 = TEST_FRAME_CATCHES_A_WRONG_ROW_VALUES[name]
    m = module_of(name, 'test_judges', 'state_of', 'sample', 'spec', 'judges')
    state_of, sample, spec, judges = m.state_of, m.sample, m.spec, m.judges
    broken = state_of().replace(_v0, _v1)
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


#: 2 家同款——输入：金答案不在选项里。输出：无。预期行为：报错。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_CHOICE_REJECTS_AN_ANSWER_OUTSIDE_THE_OPTIONS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'functions': ('f', '题'),
    'receiptcat': ('cat', 'q'),
}


@pytest.mark.parametrize("name", TEST_CHOICE_REJECTS_AN_ANSWER_OUTSIDE_THE_OPTIONS_VALUES)
def test_choice_rejects_an_answer_outside_the_options(name: str) -> None:
    """输入：金答案不在选项里。输出：无。预期行为：报错。
    """
    _v0, _v1 = TEST_CHOICE_REJECTS_AN_ANSWER_OUTSIDE_THE_OPTIONS_VALUES[name]
    m = module_of(name, 'test_questions', 'questions')
    questions = m.questions
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice(_v0, _v1, {"a": "甲", "b": "乙"}, "c")


#: 2 家同款——输入：全空间 32 组 case 的忠实正文。输出：无。预期行为：一条都不报错。（与 `test_check_draft_accepts_every_faithful_text` 同名不同形：字面量个数不一样，拆成两份）
TEST_CHECK_DRAFT_ACCEPTS_EVERY_FAITHFUL_TEXT_2: Final[tuple[str, ...]] = ('guardrail', 'promptinject')


@pytest.mark.parametrize("name", TEST_CHECK_DRAFT_ACCEPTS_EVERY_FAITHFUL_TEXT_2)
def test_check_draft_accepts_every_faithful_text_2(name: str) -> None:
    """输入：全空间 32 组 case 的忠实正文。输出：无。预期行为：一条都不报错。
    """
    m = module_of(name, 'test_render', 'CASES', 'render', 'text_of')
    CASES, render, text_of = m.CASES, m.render, m.text_of
    for case in CASES:
        render.check_draft(render.ContentDraft(text=text_of(case)), case)


#: 2 家同款——输入：缺通道的语言字典。输出：无。预期行为：抛 ValueError。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_SOURCE_COPY_REPORTS_A_MISSING_SHAPE_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'logtriage': ('通道 infra', 'app', '应用日志', 'infra'),
    'tracescore': ('缺入口 editor', 'cli', '命令行代理的轨迹', 'editor'),
}


@pytest.mark.parametrize("name", TEST_SOURCE_COPY_REPORTS_A_MISSING_SHAPE_VALUES)
def test_source_copy_reports_a_missing_shape(name: str) -> None:
    """输入：缺通道的语言字典。输出：无。预期行为：抛 ValueError。
    """
    _v0, _v1, _v2, _v3 = TEST_SOURCE_COPY_REPORTS_A_MISSING_SHAPE_VALUES[name]
    m = module_of(name, 'test_render', 'render')
    render = m.render
    with pytest.raises(ValueError, match=_v0):
        render.source_copy({"source": {_v1: {"desc": _v2}}}, _v3)


#: 2 家同款——输入：state 被写坏。输出：无。预期行为：判不通过并说明反解失败，不抛异常。
TEST_LABEL_JUDGE_REPORTS_AN_UNPARSABLE_STATE: Final[tuple[str, ...]] = ('planstep', 'sentiment')


@pytest.mark.parametrize("name", TEST_LABEL_JUDGE_REPORTS_AN_UNPARSABLE_STATE)
def test_label_judge_reports_an_unparsable_state(name: str) -> None:
    """输入：state 被写坏。输出：无。预期行为：判不通过并说明反解失败，不抛异常。
    """
    m = module_of(name, 'test_judges', 'judges')
    judges = m.judges
    verdict = judges.LabelJudge(locale="zh").judge({"state": "乱写", "questions": {}, "answers": {}})
    assert verdict.ok is False and "重算失败" in verdict.reason


#: 2 家同款——输入：不存在的题族。输出：无。预期行为：报错——新加了题族却忘了在这儿实现，不能静默放过。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_EXPECTED_OF_REJECTS_AN_UNKNOWN_FAMILY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'planstep': ('next', 't1'),
    'sentiment': ('tone', 'pos'),
}


@pytest.mark.parametrize("name", TEST_EXPECTED_OF_REJECTS_AN_UNKNOWN_FAMILY_VALUES)
def test_expected_of_rejects_an_unknown_family(name: str) -> None:
    """输入：不存在的题族。输出：无。预期行为：报错——新加了题族却忘了在这儿实现，不能静默放过。
    """
    _v0, _v1 = TEST_EXPECTED_OF_REJECTS_AN_UNKNOWN_FAMILY_VALUES[name]
    m = module_of(name, 'test_judges', 'LOC', 'judges')
    LOC, judges = m.LOC, m.judges
    with pytest.raises(ValueError, match="未知题族"):
        judges.expected_of("nope", {_v0: _v1}, LOC)


#: 2 家同款——输入：重算出来的选项名不在选项表里。输出：无。预期行为：报人话，不静默取空串。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_OPTION_TEXT_REPORTS_A_MISSING_OPTION_KEY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'planstep': ('next', 'next'),
    'sentiment': ('pivot', 'pivot'),
}


@pytest.mark.parametrize("name", TEST_OPTION_TEXT_REPORTS_A_MISSING_OPTION_KEY_VALUES)
def test_option_text_reports_a_missing_option_key(name: str) -> None:
    """输入：重算出来的选项名不在选项表里。输出：无。预期行为：报人话，不静默取空串。
    """
    _v0, _v1 = TEST_OPTION_TEXT_REPORTS_A_MISSING_OPTION_KEY_VALUES[name]
    m = module_of(name, 'test_judges', 'LOC', 'judges')
    LOC, judges = m.LOC, m.judges
    with pytest.raises(ValueError, match="不在选项表里"):
        judges.option_text(_v0, {_v1: "t9"}, LOC)


#: 2 家同款——输入：score 题族。输出：无。预期行为：报错——带选项的只有 `next` 一族。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_OPTION_TEXT_REPORTS_A_FAMILY_WITHOUT_OPTIONS_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'planstep': ('remaining', 'remaining'),
    'sentiment': ('drop', 'drop'),
}


@pytest.mark.parametrize("name", TEST_OPTION_TEXT_REPORTS_A_FAMILY_WITHOUT_OPTIONS_VALUES)
def test_option_text_reports_a_family_without_options(name: str) -> None:
    """输入：score 题族。输出：无。预期行为：报错——带选项的只有 `next` 一族。
    """
    _v0, _v1 = TEST_OPTION_TEXT_REPORTS_A_FAMILY_WITHOUT_OPTIONS_VALUES[name]
    m = module_of(name, 'test_judges', 'LOC', 'judges')
    LOC, judges = m.LOC, m.judges
    with pytest.raises(ValueError, match="带选项的只有"):
        judges.option_text(_v0, {_v1: 1}, LOC)


#: 2 家同款——输入：五行骨架。输出：无。预期行为：一条判据都不取到（它是 1 级那一档要的东西）。
TEST_THE_PLAIN_RECORD_TRIPS_NOTHING: Final[tuple[str, ...]] = ('alertrisk', 'depcheck')


@pytest.mark.parametrize("name", TEST_THE_PLAIN_RECORD_TRIPS_NOTHING)
def test_the_plain_record_trips_nothing(name: str) -> None:
    """输入：五行骨架。输出：无。预期行为：一条判据都不取到（它是 1 级那一档要的东西）。
    """
    m = module_of(name, 'test_features', 'features', 'render_lines', 'BASE_LINES')
    features, render_lines, BASE_LINES = m.features, m.render_lines, m.BASE_LINES
    assert features.hit_set(render_lines(BASE_LINES)) == ()


#: 2 家同款——输入：无。输出：无。预期行为：全空间够大，否则下面的分布尺子量不到东西。
#: 家 -> 各家不同的字面量（按正文里出现的顺序）。
TEST_CASE_SPACE_IS_NOT_EMPTY_VALUES: Final[dict[str, tuple[object, ...]]] = {
    'cachereuse': (100,),
    'siftrank': (60,),
}


@pytest.mark.parametrize("name", TEST_CASE_SPACE_IS_NOT_EMPTY_VALUES)
def test_case_space_is_not_empty(name: str) -> None:
    """输入：无。输出：无。预期行为：全空间够大，否则下面的分布尺子量不到东西。
    """
    _v0, = TEST_CASE_SPACE_IS_NOT_EMPTY_VALUES[name]
    m = module_of(name, 'test_spec', 'FULL')
    FULL = m.FULL
    assert len(FULL) >= _v0, f"全空间只有 {len(FULL)} 个 case"
