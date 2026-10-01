"""社交帖子分诊的判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations

import json

from jc.backend.types import Draft
from jc.jsontypes import JsonValue

from tests.workflows.posttriage.conftest import (
    BASE,
    faithful,
    quote_for,
    sample_of,
    spec_for,
    text_of,
    trace_report,
)
from workflows.posttriage import features, judges, locales, render
from workflows.posttriage.features import PostTriageCase

#: 全空间 32 组 case。
CASES = features.build_cases(512)

#: 一个普通 case：命中两条、没有诱饵。
CASE = PostTriageCase("S1+S3", features.DECOY_NONE)


def _drop_signal_line(state: str, loc: dict[str, JsonValue], signal: str) -> str:
    """输入：state、语言字典、判据编号。输出：抹掉判据表里那一行之后的 state。"""
    prefix = f"{render.LINE_PREFIX}{signal}"
    lines = [line for line in state.splitlines() if not line.strip().startswith(prefix)]
    return "\n".join(lines)


def test_hygiene_rejects_a_missing_decoy() -> None:
    """输入：计划要诱饵、正文里没有的样本。输出：无。预期行为：不通过，理由点名诱饵。"""
    case = PostTriageCase("S1", features.DECOY_LOWER)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(BASE + " 关注 $ABCDE"), spec_for(case)
    )
    assert not verdict.ok and not verdict.skipped
    assert "诱饵" in verdict.reason


def test_hygiene_rejects_an_extra_hit() -> None:
    """输入：正文多命中一条判据的样本。输出：无。预期行为：不通过，理由是命中组合对不上。"""
    case = PostTriageCase("S1", features.DECOY_NONE)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(BASE + " 关注 $ABCDE 涨了 7.5%"), spec_for(case)
    )
    assert not verdict.ok
    assert "命中组合" in verdict.reason


def test_frame_accepts_a_faithful_sample() -> None:
    """输入：忠实样本。输出：无。预期行为：判据表与正文骨架都对得上。"""
    spec, sample = faithful(CASE)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert "骨架" in verdict.reason


def test_frame_rejects_a_state_missing_a_signal_line() -> None:
    """输入：判据表里少一行的 state。输出：无。预期行为：不通过，理由是判据表不齐。"""
    spec, sample = faithful(CASE)
    loc = locales.load("zh")
    sample["state"] = _drop_signal_line(str(sample["state"]), loc, "S2")
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "判据表" in verdict.reason


def test_frame_rejects_a_state_without_a_content_block() -> None:
    """输入：认不出正文的 state。输出：无。预期行为：不通过（判据表与正文都不对）。"""
    _, sample = faithful(CASE)
    sample["state"] = "这里什么抬头都没有"
    verdict = judges.FrameJudge("zh").judge(sample, spec_for(CASE))
    assert not verdict.ok
    assert "正文" in verdict.reason


def test_quote_holds_accepts_a_quote_wrapped_in_a_json_array() -> None:
    """输入：引文被反解装进 JSON 数组字符串（范式 §9.88）。输出：无。

    预期行为：拆成候选之后照旧算站得住——模型把「抄几段」当成了「交数组」只是打包方式不同，
    不该把一条真命中丢成误报（存量那 20 条就是这么丢的）。
    """
    assert judges.quote_holds("S1", json.dumps([quote_for("S1")]), text_of(CASE))


def test_trace_drops_an_overclaim_whose_quote_is_real_but_off_criterion() -> None:
    """输入：多报一条 S1、引文确实在正文里但里面没有大写 cashtag 的反解。输出：无。

    预期行为：这条误报被丢掉，判定仍然通过——「抄不出原文的那一条不算命中」必须由代码把关，
    只写在提示词里挡不住（真机里 en 侧就因为模型多报一条白拒过好数据）。
    """
    case = PostTriageCase("S3+S4", features.DECOY_NONE)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("S3"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_signals": ["S3", "S4"]}
    assert "另有 1 条" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_is_fabricated() -> None:
    """输入：多报一条 S1、引文是正文里根本没有的一段的反解。输出：无。

    预期行为：引文抄不出原文，同样按误报丢掉，判定通过。
    """
    case = PostTriageCase("S3+S4", features.DECOY_NONE)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote="$ZZZZZ")
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_signals": ["S3", "S4"]}


def test_trace_skips_an_unknown_signal_id() -> None:
    """输入：反解答了一个不存在的判据编号。输出：无。

    预期行为：记成「没判成」——那是反解格式没对齐，不是这条数据错。
    """
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(_UnknownIdBackend(), "zh").judge(sample, spec)
    assert verdict.skipped and verdict.ok
    assert "不认识的判据编号" in verdict.reason


class _UnknownIdBackend:
    """只会答一个不存在的判据编号的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`signals` 里带 S9 的草稿。预期行为：不联网。"""
        del task
        return Draft(
            data={"signals": [{"signal": "S9", "quote": "S9"}]},
            raw_text="S9",
            backend="fake",
            provider="fake",
            model="fake",
            mode="fake",
        )


class _OverclaimBackend:
    """真话之外**多报一条 S1** 的假后端（真机里出现过这种误报）。

    字段：`report` 真话（由 `trace_report` 推出来）、`quote` 给那条多报的 S1 配的引文——
    分别验「引文在正文里但按判据不算命中」与「引文在正文里根本不存在」两条路径。
    """

    def __init__(self, case: PostTriageCase, quote: str) -> None:
        """输入：注入特征、误报的引文。输出：实例。"""
        self.report = trace_report(case)
        self.quote = quote

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：真话 + 一条 S1 误报。预期行为：不联网。"""
        del task
        signals = [*self.report, {"signal": "S1", "quote": self.quote}]
        return Draft(
            data={"signals": signals},
            raw_text="signals=overclaim",
            backend="fake",
            provider="fake",
            model="fake",
            mode="fake",
        )
