"""多家逐字抄同一段断言的用例：收成表 + 参数化用例。

输入：各家测试包里那段逐字相同的断言（只有域名 / 题号 / 字段名不同）。输出：无。
预期行为：与原来各家各写一遍的那些用例等价——脚手架与数据都从各家原处现场取
（`importlib` 按包名导入），这里只留「哪几家 x 哪几个字段」这一份事实。原来散在 42 个
workflow 测试包里的 275 个逐字重复的测试函数收成 72 条用例（312 个参数组合）；各家的
`_run` / `_rows` / `BAD_LABELS` / `CASE` 仍留在原处，不搬家。
"""

from __future__ import annotations

import importlib
import random
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
