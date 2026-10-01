"""判别器测试：三道闸各自拦什么、端点故障为什么只算「没判成」。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

import pytest

from jc.backend.types import AllProvidersDown, Draft, ProviderUnavailable
from jc.jsontypes import JsonValue

from tests.workflows.alertrisk.conftest import (
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
from workflows.alertrisk import features, judges, locales, render
from workflows.alertrisk.features import AlertRiskCase

CASE: Final[AlertRiskCase] = AlertRiskCase("R1+R3", features.SOURCE_EDR)


def test_quote_holds_needs_the_quote_and_the_whole_record() -> None:
    """输入：四种引文。输出：无。

    预期行为：引文原样在记录里、引文自己满足判据、整屏记录也满足判据——三件事同时成立才算数；
    只把形状塞进引文（记录里没有那一行）是误报。
    """
    body = packet_of(("R1",))
    assert judges.quote_holds("R1", quote_for("R1"), body)
    assert judges.quote_holds("R1", json.dumps([quote_for("R1")]), body)  # 引文被装进 JSON 数组字符串（范式 §9.88）
    assert not judges.quote_holds("R1", "exposure: reachable=internal", body)
    assert not judges.quote_holds("R2", quote_for("R2"), body)
    assert not judges.quote_holds("R9", quote_for("R1"), body)
    assert not judges.quote_holds("R1", "reachable=public", body)  # 只抄取值、不带键名，不算


def test_quote_holds_ignores_whitespace_differences() -> None:
    """输入：引文里多打了空白。输出：无。预期行为：空白折叠之后仍然算抄对了。"""
    body = packet_of(("R3",))
    assert judges.quote_holds("R3", "impact:   hosts=24  -- 二十四台生产机器", body)


def test_hygiene_passes_a_faithful_sample() -> None:
    """输入：完全兑现计划的一条样本。输出：无。预期行为：通过。"""
    spec, sample = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped


@pytest.mark.parametrize(
    ("switch", "needle"),
    [
        ("drop_line", "命中组合"),
        ("extra_line", "命中组合"),
        ("missing_key", "缺这几行"),
        ("empty_detail", "那一行是空的"),
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


def test_hygiene_skips_when_it_cannot_judge() -> None:
    """输入：没有规格、规格串了别家、样本里没有 state。输出：无。

    预期行为：三种都记「没判成」——它们不是数据不合格，是这一遍判不了。
    """
    judge = judges.HygieneJudge("zh")
    spec, sample = faithful(CASE)
    assert judge.judge(sample, None).skipped
    assert judge.judge(sample, "别的 workflow 的 spec").skipped
    assert judge.judge({"state": 42}, spec).skipped


def test_hygiene_blocks_a_record_that_pastes_the_rule() -> None:
    """输入：把因子原文整段贴进 `rule:` 那一行（命中组合仍然是对的）。输出：无。

    预期行为：被**抄规格**那道闸拦下——命中组合、长度、结构键全都合格，只有「整段贴判据原文」
    这一条不成立，所以拦下它的只能是 `pasted_rules`。
    """
    loc = locales.load("zh")
    rule = str(render.fact_copy(loc, "R1")["rule"])
    packet = render_lines(
        {**BASE_LINES, "exposure": HIT_LINES["R1"], "rule": f"rule: {rule}"}
    )
    spec = spec_for(AlertRiskCase("R1", features.SOURCE_EDR))
    # 除了「整段贴判据原文」这一条，别的闸全都合格——所以拦下它的只能是 `pasted_rules`。
    assert features.missing_keys(packet) == ()
    assert features.hits_of(packet) == "R1"
    assert render.MIN_CONTENT_CHARS <= len(packet) <= render.MAX_CONTENT_CHARS
    assert render.pasted_rules(packet, loc)
    verdict = judges.HygieneJudge("zh").judge(sample_of(packet), spec)
    assert not verdict.ok
    assert "因子原文" in verdict.reason


def test_frame_passes_a_faithful_sample() -> None:
    """输入：完全兑现计划的一条样本。输出：无。预期行为：通过。"""
    spec, sample = faithful(CASE)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped


def test_frame_blocks_a_broken_skeleton() -> None:
    """输入：因子表少一行、矩阵少一行、正文少一个键。输出：无。

    预期行为：三种骨架损坏各自被点名——矩阵少一行等于有组合没定义，那几格就答不出来。
    """
    spec, sample = faithful(CASE)
    judge = judges.FrameJudge("zh")
    state = str(sample["state"])
    without_fact = "\n".join(
        line for line in state.splitlines() if not line.strip().startswith("- R2")
    )
    assert not judge.judge({"state": without_fact}, spec).ok
    without_matrix_row = "\n".join(
        line for line in state.splitlines() if "R1=high R2=high R3=high" not in line
    )
    verdict = judge.judge({"state": without_matrix_row}, spec)
    assert not verdict.ok and "定级矩阵" in verdict.reason
    without_detail = "\n".join(
        line for line in state.splitlines() if not line.strip().startswith("detail:")
    )
    assert not judge.judge({"state": without_detail}, spec).ok


def test_frame_skips_when_it_cannot_judge() -> None:
    """输入：没有规格、样本里没有 state。输出：无。预期行为：都记「没判成」。"""
    judge = judges.FrameJudge("zh")
    spec, sample = faithful(CASE)
    assert judge.judge(sample, None).skipped
    assert judge.judge({"state": None}, spec).skipped


def test_trace_passes_when_the_reader_agrees() -> None:
    """输入：反解说真话。输出：无。预期行为：通过，并把读到的因子与等级记在 raw 里。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw["traced_hits"] == ["R1", "R3"]
    assert verdict.raw["traced_level"] == spec.labels.level


def test_trace_blocks_a_missing_factor() -> None:
    """输入：反解漏报一条。输出：无。预期行为：拦下——读者读出来的东西与计划对不上。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok
    assert "R3" in verdict.reason


def test_trace_blocks_a_misread_matrix() -> None:
    """输入：因子读对了、等级读错一档。输出：无。

    预期行为：拦下——`grade` 是这一族的中心题型，矩阵读不对就等于那一道题答不出来。
    """
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_level_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok
    assert "矩阵" in verdict.reason


def test_trace_treats_endpoint_failure_as_unjudged() -> None:
    """输入：反解时端点故障。输出：无。预期行为：记「没判成」，不写成数据质量问题。"""
    spec, sample = faithful(CASE)
    backend = FakeBackend(trace_raises=ProviderUnavailable("端点挂了"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.skipped and verdict.ok
    assert "端点故障" in verdict.reason


def test_trace_reraises_a_fatal_failure() -> None:
    """输入：反解时所有 provider 都挂了。输出：无。预期行为：抛出去让整批停下。"""
    spec, sample = faithful(CASE)
    backend = FakeBackend(trace_raises=AllProvidersDown("全挂了"))
    with pytest.raises(AllProvidersDown):
        judges.TraceJudge(backend, "zh").judge(sample, spec)


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
        ({"tags": [{"tag": "R9", "quote": "x"}], "level": 3}, "不认识的因子编号"),
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
            {"tag": "R1", "quote": quote_for("R1")},
            {"tag": "R3", "quote": quote_for("R3")},
            {"tag": "R2", "quote": "exploit: auth=none -- 记录里根本没这一行"},
        ],
        "level": spec.labels.level,
    }
    verdict = judges.TraceJudge(StubBackend(data), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw["traced_hits"] == ["R1", "R3"]
    assert "已按没命中处理" in verdict.reason


def test_trace_skips_without_a_spec() -> None:
    """输入：没有规格。输出：无。预期行为：记「没判成」。"""
    _, sample = faithful(CASE)
    assert judges.TraceJudge(FakeBackend(), "zh").judge(sample, None).skipped
