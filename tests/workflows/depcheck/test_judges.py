"""判别器测试：三道闸各自拦什么、端点故障为什么只算「没判成」。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue

from tests.workflows.depcheck.conftest import (
    BASE_LINES,
    HIT_LINES,
    FakeBackend,
    faithful,
    packet_of,
    quote_for,
    render_lines,
    sample_of,
    spec_for,
)
from workflows.depcheck import features, judges, locales, render
from workflows.depcheck.features import DepCheckCase

CASE: Final[DepCheckCase] = DepCheckCase("D1+D3", features.ECO_NPM)


def test_quote_holds_needs_the_quote_and_the_whole_record() -> None:
    """输入：四种引文。输出：无。

    预期行为：引文原样在记录里、引文自己满足判据、整屏记录也满足判据——三件事同时成立才算数；
    只把形状塞进引文（记录里没有那一行）是误报。
    """
    body = packet_of(("D1",))
    assert judges.quote_holds("D1", quote_for("D1"), body)
    assert not judges.quote_holds("D1", "license: MIT", body)
    assert not judges.quote_holds("D2", quote_for("D2"), body)
    assert not judges.quote_holds("D9", quote_for("D1"), body)
    assert not judges.quote_holds("D1", "AGPL-3.0-only", body)  # 只抄取值、不带键名，不算


def test_quote_holds_ignores_whitespace_differences() -> None:
    """输入：引文里多打了空白。输出：无。预期行为：空白折叠之后仍然算抄对了。"""
    body = packet_of(("D3",))
    assert judges.quote_holds("D3", "advisories:   count=3", body)


@pytest.mark.parametrize(
    ("switch", "needle"),
    [
        ("drop_line", "命中组合"),
        ("extra_line", "命中组合"),
        ("missing_key", "缺这几行"),
        ("empty_note", "那一行是空的"),
        ("fence", "格式残留"),
    ],
)
def test_hygiene_blocks_each_broken_record(switch: str, needle: str) -> None:
    """输入：五种「写歪」的记录。输出：无。预期行为：各自被拦下，理由说得清。"""
    backend = FakeBackend(**{switch: True})
    spec = spec_for(CASE)
    packet = backend.packet_for(CASE)
    verdict = judges.HygieneJudge("zh").judge(sample_of(packet), spec)
    assert not verdict.ok and not verdict.skipped
    assert needle in verdict.reason


def test_hygiene_blocks_a_record_that_pastes_the_rule() -> None:
    """输入：把标记原文整段贴进 `package:` 那一行（命中组合仍然是对的）。输出：无。

    预期行为：被**抄规格**那道闸拦下——命中组合、长度、结构键全都合格，只有「整段贴判据原文」
    这一条不成立，所以拦下它的只能是 `pasted_rules`。
    """
    loc = locales.load("zh")
    rule = str(render.fact_copy(loc, "D1")["rule"])
    packet = render_lines(
        {**BASE_LINES, "license": HIT_LINES["D1"], "package": f"package: {rule}"}
    )
    spec = spec_for(DepCheckCase("D1", features.ECO_NPM))
    # 除了「整段贴判据原文」这一条，别的闸全都合格——所以拦下它的只能是 `pasted_rules`。
    assert features.missing_keys(packet) == ()
    assert features.hits_of(packet) == "D1"
    assert render.MIN_CONTENT_CHARS <= len(packet) <= render.MAX_CONTENT_CHARS
    assert render.pasted_rules(packet, loc)
    verdict = judges.HygieneJudge("zh").judge(sample_of(packet), spec)
    assert not verdict.ok
    assert "标记原文" in verdict.reason


def test_hygiene_blocks_a_note_with_a_risk_verdict() -> None:
    """输入：`note:` 那一行写进了风险结论（命中组合仍然是对的）。输出：无。

    预期行为：被**结论闸**拦下——命中组合、长度、结构键、抄规格全都合格，只有「这一行不许下结论」
    这一条不成立（范式 §9.59：结论写进自由文字，考生就能绕过结构键去读备注）。
    """
    loc = locales.load("zh")
    packet = render_lines(
        {
            **BASE_LINES,
            "status": HIT_LINES["D2"],
            "note": "note: 这个包用来把结构化日志写进收集端，上游已经归档不再维护，但功能稳定所以引入喵",
        }
    )
    spec = spec_for(DepCheckCase("D2", features.ECO_NPM))
    assert features.missing_keys(packet) == ()
    assert features.hits_of(packet) == "D2"
    assert render.MIN_CONTENT_CHARS <= len(packet) <= render.MAX_CONTENT_CHARS
    assert render.pasted_rules(packet, loc) == []
    verdict = judges.HygieneJudge("zh").judge(sample_of(packet), spec)
    assert not verdict.ok and not verdict.skipped
    assert "风险结论" in verdict.reason


def _drop_line(state: str, needle: str) -> str:
    """输入：state、要抹掉的那一行里含的字样。输出：抹掉之后的第一处 state。"""
    out: list[str] = []
    removed = False
    for line in state.splitlines():
        if not removed and needle in line:
            removed = True
            continue
        out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize(
    ("needle", "expected"),
    [
        ("D2", "标记表列出的编号"),
        ("D1=2", "定级规则与规范行不一致"),
        ("AGPL-3.0-only", "依赖记录缺这几行"),
    ],
)
def test_frame_blocks_a_broken_skeleton(needle: str, expected: str) -> None:
    """输入：标记表少一行、权重行被抹掉、正文少一个结构键。输出：无。

    预期行为：三种骨架损坏各自被点名——标记表、定级规则与依赖记录的骨架都必须与规范形态一致，
    任何一处对不上都说明渲染挂错了 case，标签会跟着一起错。
    """
    spec, sample = faithful(CASE)
    broken = _drop_line(str(sample["state"]), needle)
    verdict = judges.FrameJudge("zh").judge({"state": broken}, spec)
    assert not verdict.ok
    assert expected in verdict.reason


def test_frame_skips_when_it_cannot_judge() -> None:
    """输入：没有规格、没有 state。输出：无。预期行为：都记「没判成」。"""
    spec, sample = faithful(CASE)
    judge = judges.FrameJudge("zh")
    assert judge.judge(sample, None).skipped
    assert judge.judge({"state": 42}, spec).skipped


def test_trace_passes_a_faithful_sample() -> None:
    """输入：完全读对的反解。输出：无。预期行为：通过，理由里带上读到的命中组合与等级。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw["traced_hits"] == ["D1", "D3"]
    assert verdict.raw["traced_level"] == spec.labels.level


def test_trace_blocks_a_missing_tag() -> None:
    """输入：反解漏报一条。输出：无。预期行为：拦下——读者读出来的东西与计划对不上。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok
    assert "D3" in verdict.reason


def test_trace_blocks_a_misread_level() -> None:
    """输入：命中读对了、等级读错一档。输出：无。

    预期行为：拦下——`grade` 是这一族的中心题型，权重没算对就等于那一道题答不出来。
    """
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_level_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok
    assert "权重" in verdict.reason


@dataclass(frozen=True)
class StubBackend:
    """只会吐固定草稿的后端：用来验反解草稿形态不对时的那几条「没判成」路径。"""

    data: dict[str, JsonValue]
    name: str = "stub"

    def generate(self, task: object) -> Draft:
        """输入：任务（忽略）。输出：固定草稿。预期行为：不发网络。"""
        del task
        return Draft(
            data=self.data, raw_text="", backend="stub", provider="stub", model="stub", mode="stub"
        )

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：逐个走 `generate`。"""
        return [self.generate(task) for task in tasks]


@pytest.mark.parametrize(
    ("data", "needle"),
    [
        ({}, "缺 tags"),
        ({"tags": "不是列表", "level": 3}, "缺 tags"),
        ({"tags": ["不是对象"], "level": 3}, "不是「编号 + 引文」对象"),
        ({"tags": [{"tag": "D9", "quote": "x"}], "level": 3}, "不认识的标记编号"),
        ({"tags": [], "level": 9}, "1~5 的整数"),
        ({"tags": [], "level": "三"}, "1~5 的整数"),
    ],
)
def test_trace_skips_drafts_it_cannot_read(data: dict[str, JsonValue], needle: str) -> None:
    """输入：六种形态不对的反解草稿。输出：无。

    预期行为：都记「没判成」（`skipped=True`）——那是反解格式没对齐，不是数据本身错。
    """
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(StubBackend(data), "zh").judge(sample, spec)
    assert verdict.skipped and verdict.ok
    assert needle in verdict.reason


def test_trace_drops_quotes_that_do_not_stand_up() -> None:
    """输入：反解多报一条，但引文在记录里不存在。输出：无。

    预期行为：那条误报被丢掉，剩下的与计划一致——所以整条通过，理由里点名被丢掉的那条。
    """
    spec, sample = faithful(CASE)
    data: dict[str, JsonValue] = {
        "tags": [
            {"tag": "D1", "quote": quote_for("D1")},
            {"tag": "D3", "quote": quote_for("D3")},
            {"tag": "D2", "quote": "status: archived -- 记录里根本没这一行"},
        ],
        "level": spec.labels.level,
    }
    verdict = judges.TraceJudge(StubBackend(data), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw["traced_hits"] == ["D1", "D3"]
    assert "已按没命中处理" in verdict.reason
