"""假后端作弊时必须被拦：两种「故意说反话」各收成一张表。

输入：各家假后端「写歪 / 反解说反话」的开关，以及它该踩出来的拒绝码。输出：无。
预期行为：表里每一格各跑一遍小批量管线——要么一条都留不下、拒绝码就是表里那一个，要么
只有「本来就没东西可说」的那几条能活下来。写歪的正文是渲染期的 `E_ASSEMBLE`（重试耗尽后
仍被拒，不静默放行），反解说反话是 `E_JUDGE_TRACE`。

这两批原来是逐字抄在各家 `test_pipeline.py` 里的（`BROKEN_DRAFTS` 24 家 x 101 个开关，
`LYING_TRACES` 13 家），除了前缀与开关名逐字不同。开关名与拒绝码都是「假后端长什么样」的
事实，收在这两张表里；各家的 `FakeBackend` 与 `make_config` 仍从原处取：前者是开关的定义处，
后者带着各家自己的差异（如 `language_gate`）。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from jc import pipeline

#: 每家：运行前缀（原来用例里的 `p3` / `c3` / `g3`）、开关名 -> 拒绝码。
#: 全部 24 家 x 101 个开关，逐格与原来那条参数化用例等价。
BROKEN_DRAFTS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "adblock": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("no_html", "E_ASSEMBLE"),)),
    "alertrisk": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("missing_key", "E_ASSEMBLE"), ("empty_detail", "E_ASSEMBLE"),)),
    "causechain": ("p3", (("flip_event", "E_ASSEMBLE"), ("drop_service", "E_ASSEMBLE"), ("extra_service", "E_ASSEMBLE"), ("pasted_rule", "E_ASSEMBLE"), ("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"),)),
    "chatgate": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("clean_carries_flaw", "E_ASSEMBLE"),)),
    "citation": ("c3", (("duplicate", "E_ASSEMBLE"), ("drop_sentence", "E_ASSEMBLE"), ("no_digit", "E_ASSEMBLE"), ("short_sentence", "E_ASSEMBLE"), ("fence_line", "E_ASSEMBLE"), ("title_fence", "E_ASSEMBLE"), ("trace_lies", "E_JUDGE_TRACE"),)),
    "debugstep": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("missing_key", "E_ASSEMBLE"),)),
    "depcheck": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("missing_key", "E_ASSEMBLE"), ("empty_note", "E_ASSEMBLE"),)),
    "guardrail": ("g3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("decoy_extra", "E_ASSEMBLE"), ("trace_lies", "E_JUDGE_TRACE"),)),
    "hirecheck": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("missing_key", "E_ASSEMBLE"), ("empty_note", "E_ASSEMBLE"), ("verdict_note", "E_ASSEMBLE"), ("leak_skill", "E_ASSEMBLE"),)),
    "lint": ("c3", (("unrealized", "E_ASSEMBLE"), ("extra_hit", "E_ASSEMBLE"), ("drop_line", "E_ASSEMBLE"), ("fence_line", "E_ASSEMBLE"), ("trace_lies_hits", "E_JUDGE_TRACE"),)),
    "logtriage": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"),)),
    "memorygate": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"),)),
    "posttriage": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("decoy_extra", "E_ASSEMBLE"),)),
    "promptinject": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"),)),
    "receiptcat": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("name_leak", "E_ASSEMBLE"), ("wrong_category", "E_ASSEMBLE"), ("wrong_amount", "E_ASSEMBLE"), ("decoy_extra", "E_ASSEMBLE"),)),
    "rerank": ("c3", (("unrealized", "E_ASSEMBLE"), ("extra_hit", "E_ASSEMBLE"), ("duplicate", "E_ASSEMBLE"), ("drop_candidate", "E_ASSEMBLE"), ("bad_terms", "E_ASSEMBLE"), ("term_missing_from_query", "E_ASSEMBLE"), ("fence_line", "E_ASSEMBLE"), ("trace_lies", "E_JUDGE_TRACE"),)),
    "review": ("c3", (("trace_lies_secret", "E_JUDGE_TRACE"), ("trace_lies_debug", "E_JUDGE_TRACE"), ("table_body", "E_ASSEMBLE"), ("no_prefix_body", "E_ASSEMBLE"), ("fence_body", "E_ASSEMBLE"), ("label_body", "E_ASSEMBLE"), ("empty_body", "E_ASSEMBLE"),)),
    "rubric": ("c3", (("trace_lies_placeholders", "E_JUDGE_TRACE"), ("trace_lies_citations", "E_JUDGE_TRACE"), ("table_body", "E_ASSEMBLE"), ("fence_body", "E_ASSEMBLE"), ("label_body", "E_ASSEMBLE"), ("empty_body", "E_ASSEMBLE"), ("short_body", "E_ASSEMBLE"),)),
    "segmentation": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("drop_marker", "E_ASSEMBLE"), ("extra_marker", "E_ASSEMBLE"), ("no_trace", "E_ASSEMBLE"), ("wrong_pages", "E_ASSEMBLE"),)),
    "sqlreview": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("missing_key", "E_ASSEMBLE"),)),
    "subqa": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("missing_key", "E_ASSEMBLE"),)),
    "testimpact": ("p3", (("prose", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"), ("short_title", "E_ASSEMBLE"),)),
    "tracescore": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"),)),
    "turntake": ("p3", (("too_short", "E_ASSEMBLE"), ("fence", "E_ASSEMBLE"),)),
}


@pytest.mark.parametrize("name", sorted(BROKEN_DRAFTS))
def test_broken_drafts_are_rejected(tmp_path: Path, name: str) -> None:
    """输入：该家每一个「作弊开关」。输出：无。

    预期行为：每个开关各跑一遍，一条都留不下，拒绝码与该开关该踩的那一个完全相同。
    假后端与运行配置从各家原处取，所以这里只表「开关 -> 拒绝码」这一份事实。
    """
    prefix, switches = BROKEN_DRAFTS[name]
    test_pipeline = importlib.import_module(f"tests.workflows.{name}.test_pipeline")
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    for switch, code in switches:
        result = pipeline.run(
            test_pipeline.make_config(
                tmp_path, f"{prefix}-{switch}", conftest.FakeBackend(**{switch: True})
            )
        )
        assert result.accepted == 0, f"{name} 的 {switch} 竟有草稿被收下"
        assert {item["code"] for item in result.rejected} == {code}, f"{name} 的 {switch}"


#: 13 家逐字相同的「反解说反话」：只有本来就没东西可漏报的那几条能活下来，其余全被
#: `E_JUDGE_TRACE` 拦下。运行前缀都是 `p3-trace`，`cases_of("train")` 与 `case.tags()`
#: 在这 13 家里同形（4 家的函数名不同，函数体逐字相同）。
LYING_TRACES: tuple[str, ...] = (
    "adblock",
    "alertrisk",
    "chatgate",
    "debugstep",
    "depcheck",
    "hirecheck",
    "logtriage",
    "memorygate",
    "promptinject",
    "sqlreview",
    "subqa",
    "tracescore",
    "turntake",
)


@pytest.mark.parametrize("name", LYING_TRACES)
def test_a_lying_trace_only_spares_the_cases_with_nothing_to_lie_about(
    tmp_path: Path, name: str
) -> None:
    """输入：该家「反解时漏报最后一条」的假后端。输出：无。

    预期行为：命中组合为空的那几条 case 照样合格（没有可漏报的），其余全被 `E_JUDGE_TRACE`
    拦下——「反解说反话」与「片段写歪」是两条独立的闸，各自拦各自那一类。
    """
    test_pipeline = importlib.import_module(f"tests.workflows.{name}.test_pipeline")
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    unaffected = sum(1 for case in test_pipeline.cases_of("train") if not case.tags())
    result = pipeline.run(
        test_pipeline.make_config(tmp_path, "p3-trace", conftest.FakeBackend(trace_lies=True))
    )
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}
