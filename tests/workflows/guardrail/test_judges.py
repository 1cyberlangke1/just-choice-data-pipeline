"""内容审核的判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations

import json

from jc.backend.types import Draft
from jc.jsontypes import JsonValue

from tests.workflows.guardrail.conftest import (
    BASE,
    FakeBackend,
    faithful,
    quote_for,
    sample_of,
    spec_for,
    text_of,
    trace_report,
)
from workflows.guardrail import features, judges, locales, render
from workflows.guardrail.features import GuardrailCase

#: 全空间 32 组 case。
CASES = features.build_cases(512)

#: 一个普通 case：命中两条、没有诱饵。
CASE = GuardrailCase("P1+P3", features.DECOY_NONE)


def _drop_clause_line(state: str, loc: dict[str, JsonValue], clause: str) -> str:
    """输入：state、语言字典、条款编号。输出：抹掉条款表里那一行之后的 state。"""
    prefix = f"{render.LINE_PREFIX}{clause}"
    lines = [line for line in state.splitlines() if not line.strip().startswith(prefix)]
    return "\n".join(lines)


def test_hygiene_accepts_every_faithful_sample() -> None:
    """输入：全空间 32 组 case 的忠实样本。输出：无。预期行为：一条都不报。"""
    judge = judges.HygieneJudge("zh")
    for case in CASES:
        spec, sample = faithful(case)
        verdict = judge.judge(sample, spec)
        assert verdict.ok and not verdict.skipped, f"{case.describe()}: {verdict.reason}"
    assert "兑现" in judge.judge(*reversed(faithful(CASE))).reason


def test_hygiene_rejects_a_missing_decoy() -> None:
    """输入：计划要诱饵、正文里没有的样本。输出：无。预期行为：不通过，理由点名诱饵。"""
    case = GuardrailCase("P1", features.DECOY_DASHED)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(BASE + " 联系电话 13812345678"), spec_for(case)
    )
    assert not verdict.ok and not verdict.skipped
    assert "诱饵" in verdict.reason


def test_hygiene_rejects_an_extra_hit() -> None:
    """输入：正文多踩了一条条款的样本。输出：无。预期行为：不通过，理由是命中组合对不上。"""
    case = GuardrailCase("P1", features.DECOY_NONE)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(BASE + " 联系电话 13812345678 ！！！"), spec_for(case)
    )
    assert not verdict.ok
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_a_state_without_a_content_block() -> None:
    """输入：一段没有正文抬头的 state。输出：无。预期行为：不通过，理由是认不出正文。"""
    spec, _ = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge({"state": "这里什么抬头都没有"}, spec)
    assert not verdict.ok
    assert "正文" in verdict.reason


def test_hygiene_skips_when_it_cannot_judge() -> None:
    """输入：没有 spec / state 不是文本。输出：无。预期行为：记成「没判成」，不误判成坏数据。"""
    _, sample = faithful(CASE)
    judge = judges.HygieneJudge("zh")
    assert judge.judge(sample, None).skipped
    assert judge.judge({"state": 1}, spec_for(CASE)).skipped


def test_frame_accepts_a_faithful_sample() -> None:
    """输入：忠实样本。输出：无。预期行为：条款表与正文骨架都对得上。"""
    spec, sample = faithful(CASE)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert "骨架" in verdict.reason


def test_frame_rejects_a_state_missing_a_clause_line() -> None:
    """输入：条款表里少一行的 state。输出：无。预期行为：不通过，理由是条款表不齐。"""
    spec, sample = faithful(CASE)
    loc = locales.load("zh")
    sample["state"] = _drop_clause_line(str(sample["state"]), loc, "P2")
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "条款表" in verdict.reason


def test_frame_rejects_a_state_without_a_content_block() -> None:
    """输入：认不出正文的 state。输出：无。预期行为：不通过（条款表与正文都不对）。"""
    _, sample = faithful(CASE)
    sample["state"] = "这里什么抬头都没有"
    verdict = judges.FrameJudge("zh").judge(sample, spec_for(CASE))
    assert not verdict.ok
    assert "正文" in verdict.reason


def test_frame_skips_without_a_spec_or_text() -> None:
    """输入：没有 spec / state 不是文本。输出：无。预期行为：都记成「没判成」。"""
    _, sample = faithful(CASE)
    judge = judges.FrameJudge("zh")
    assert judge.judge(sample, None).skipped
    assert judge.judge({"state": None}, spec_for(CASE)).skipped


def test_trace_accepts_the_truth() -> None:
    """输入：从不撒谎的假后端。输出：无。预期行为：通过，而且真的走的是反解那个任务 id。"""
    spec, sample = faithful(CASE)
    backend = FakeBackend()
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert backend.tasks == [spec.trace_task_id]
    assert "反解与计划一致" in verdict.reason


def test_trace_rejects_a_lying_reader() -> None:
    """输入：把命中组合说反的假后端。输出：无。预期行为：不通过，理由里列出两边。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "反解读成" in verdict.reason


def test_quote_holds_accepts_a_quote_wrapped_in_a_json_array() -> None:
    """输入：引文被反解装进 JSON 数组字符串（范式 §9.88）。输出：无。

    预期行为：拆成候选之后照旧算站得住——模型把「抄几段」当成了「交数组」只是打包方式不同，
    不该把一条真命中丢成误报（存量那 20 条就是这么丢的）。
    """
    assert judges.quote_holds("P1", json.dumps([quote_for("P1")]), text_of(CASE))


def test_trace_drops_an_overclaim_whose_quote_is_real_but_off_criterion() -> None:
    """输入：多报一条 P1、引文确实在正文里但里面没有 7 位数字的反解。输出：无。

    预期行为：这条误报被丢掉，判定仍然通过——「抄不出原文的那一条不算踩中」必须由代码把关，
    只写在提示词里挡不住（真机里 en 侧就因为模型多报一条 P1 白拒过一条好数据）。
    """
    case = GuardrailCase("P3+P4", features.DECOY_NONE)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("P3"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["P3", "P4"]}
    assert "另有 1 条" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_is_fabricated() -> None:
    """输入：多报一条 P1、引文是正文里根本没有的一串数字的反解。输出：无。

    预期行为：引文抄不出原文，同样按误报丢掉，判定通过。
    """
    case = GuardrailCase("P3+P4", features.DECOY_NONE)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote="13812345678")
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["P3", "P4"]}


def test_trace_skips_an_unknown_clause_id() -> None:
    """输入：反解答了一个不存在的条款编号。输出：无。

    预期行为：记成「没判成」——那是反解格式没对齐，不是这条数据错。
    """
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(_UnknownIdBackend(), "zh").judge(sample, spec)
    assert verdict.skipped and verdict.ok
    assert "不认识的条款编号" in verdict.reason


class _UnknownIdBackend:
    """只会答一个不存在的条款编号的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`hits` 里带 P9 的草稿。预期行为：不联网。"""
        del task
        return Draft(
            data={"hits": [{"clause": "P9", "quote": "P9"}]},
            raw_text="P9",
            backend="fake",
            provider="fake",
            model="fake",
            mode="fake",
        )


class _OverclaimBackend:
    """真话之外**多报一条 P1** 的假后端（真机里出现过这种误报）。

    字段：`report` 真话（由 `trace_report` 推出来）、`quote` 给那条多报的 P1 配的引文——
    分别验「引文在正文里但按判据不算命中」与「引文在正文里根本不存在」两条路径。
    """

    def __init__(self, case: GuardrailCase, quote: str) -> None:
        """输入：注入特征、误报的引文。输出：实例。"""
        self.report = trace_report(case)
        self.quote = quote

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：真话 + 一条 P1 误报。预期行为：不联网。"""
        del task
        hits = [*self.report, {"clause": "P1", "quote": self.quote}]
        return Draft(
            data={"hits": hits},
            raw_text="hits=overclaim",
            backend="fake",
            provider="fake",
            model="fake",
            mode="fake",
        )
