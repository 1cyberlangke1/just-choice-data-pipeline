"""调试器下一步动作的判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations

import json
import pytest

from jc.backend.types import Draft

from tests.workflows._fakes import draft as _draft
from tests.workflows.debugstep.conftest import (
    BASE_LINES,
    FakeBackend,
    faithful,
    quote_for,
    render_lines,
    sample_of,
    session_of,
    spec_for,
    trace_report,
)
from workflows.debugstep import features, judges, locales, render
from workflows.debugstep.features import DebugStepCase

#: 全空间 48 组 case。
CASES = features.build_cases(512)

#: 一个普通 case：命中两条，会话来自 Python 调试器。
CASE = DebugStepCase("D1+D3", features.RUNTIME_PYTHON)


def _drop_tag_line(state: str, tag: str) -> str:
    """输入：state、判据编号。输出：抹掉判据表里那一行之后的 state。"""
    prefix = f"{render.LINE_PREFIX}{tag}"
    lines = [line for line in state.splitlines() if not line.strip().startswith(prefix)]
    return "\n".join(lines)


def test_hygiene_rejects_a_missing_hit() -> None:
    """输入：计划要 D1+D2、会话里只写了 D1 的样本。输出：无。

    预期行为：不通过，理由是命中组合对不上——少写一条和多重一条一样，都是标签兑现不了。
    """
    case = DebugStepCase("D1+D2", features.RUNTIME_PYTHON)
    verdict = judges.HygieneJudge("zh").judge(sample_of(session_of(("D1",))), spec_for(case))
    assert not verdict.ok and not verdict.skipped
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_an_extra_hit() -> None:
    """输入：会话多带了一条判据的样本。输出：无。预期行为：不通过，理由是命中组合对不上。"""
    case = DebugStepCase("D1", features.RUNTIME_PYTHON)
    verdict = judges.HygieneJudge("zh").judge(
        sample_of(session_of(("D1", "D4"))), spec_for(case)
    )
    assert not verdict.ok
    assert "命中组合" in verdict.reason


def test_hygiene_rejects_a_session_missing_a_structure_key() -> None:
    """输入：少了 `Note:` 那一行的会话。输出：无。

    预期行为：不通过，理由点名缺了哪几行——五个结构键是 state 格式的一部分，少一行那几条钉在
    行上的判据就定位不到，标签会跟着一起错。
    """
    spec, sample = faithful(CASE)
    without_note = render_lines(
        {key: value for key, value in BASE_LINES.items() if key != "note"}
    )
    verdict = judges.HygieneJudge("zh").judge(sample_of(without_note), spec)
    assert not verdict.ok and not verdict.skipped
    assert "缺这几行" in verdict.reason


def test_hygiene_rejects_fence_lines_anywhere() -> None:
    """输入：整段被围栏包起来的 state。输出：无。预期行为：不通过，理由是残留。

    四条判据都是**子串**性质，加围栏不会改变命中组合，所以这条单验的是残留那道闸。
    """
    spec, sample = faithful(DebugStepCase("D1", features.RUNTIME_PYTHON))
    sample["state"] = f"```text\n{sample['state']}\n```"
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "残留" in verdict.reason


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_hygiene_rejects_a_pasted_rule(lang: str) -> None:
    """输入：会话里整段贴了判据原文。输出：无。

    预期行为：不通过，理由是贴判据原文——那是「抄规格」的捷径，不是「写一屏调试会话」
    （范式 §9.45）。用 D3 那条判据：它的原文里带帧分隔符，所以贴进去之后按字面算仍然命中 D3，
    正好落在贴判据那道闸上（而不是先被命中组合那道闸拦下）。
    """
    loc = locales.load(lang)
    rule = str(render.tag_copy(loc, "D3")["rule"])
    case = DebugStepCase("D3", features.RUNTIME_PYTHON)
    verdict = judges.HygieneJudge(lang).judge(
        sample_of(f"{session_of(('D3',))} {rule}", lang), spec_for(case)
    )
    assert not verdict.ok and not verdict.skipped
    assert "判据原文" in verdict.reason


def test_frame_rejects_a_session_missing_a_structure_key() -> None:
    """输入：会话少了 `Note:` 那一行。输出：无。预期行为：不通过，理由点名缺了哪几行。"""
    spec, sample = faithful(CASE)
    without_note = render_lines(
        {key: value for key, value in BASE_LINES.items() if key != "note"}
    )
    verdict = judges.FrameJudge("zh").judge(sample_of(without_note), spec)
    assert not verdict.ok
    assert "缺这几行" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_does_not_match_the_criterion() -> None:
    """输入：多报一条 D1、引文却是 D3 那一段的反解。输出：无。

    预期行为：这条误报被丢掉，判定仍然通过——「抄出来的那段按判据根本不算命中」必须由代码把关。
    """
    case = DebugStepCase("D3+D4", features.RUNTIME_PYTHON)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("D3"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["D3", "D4"]}
    assert "另有 1 条" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_is_fabricated() -> None:
    """输入：多报一条 D1、引文是会话里根本没有的一段。输出：无。

    预期行为：引文抄不出原文，同样按误报丢掉，判定通过。
    """
    case = DebugStepCase("D3+D4", features.RUNTIME_PYTHON)
    spec, sample = faithful(case)
    backend = _OverclaimBackend(case, quote=quote_for("D1"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_hits": ["D3", "D4"]}


def test_quote_holds_needs_both_the_quote_and_the_criterion() -> None:
    """输入：几种引文。输出：无。

    预期行为：引文得原样在会话里、且自己满足那条判据；空白差异不算抄错。
    """
    body = session_of(("D1", "D4"))
    assert judges.quote_holds("D1", quote_for("D1"), body)
    # 引文首尾多带了换行：折叠之后一致，照样算抄对了（模型常这么答）。
    assert judges.quote_holds("D1", f"\n{quote_for('D1')}\n", body)
    assert judges.quote_holds("D1", json.dumps([quote_for("D1")]), body)  # 引文被装进 JSON 数组字符串（范式 §9.88）
    assert not judges.quote_holds("D1", quote_for("D4"), body)  # 抄出来了但不合判据
    assert not judges.quote_holds("D1", "count=3", body)  # 抄出来了但不合判据
    assert not judges.quote_holds("D9", "count=3", body)  # 不认识的编号


def test_quote_holds_tracks_the_plan_on_every_faithful_session() -> None:
    """输入：全空间 48 组 case 的忠实会话。输出：无。

    预期行为：本 workflow 的四条判据都是**纯子串**性质（不像 `turntake` 的「标点断在半路」看的是
    整段结尾），所以「引文命中」与「整段命中」在这四族里总是同进同退：`quote_holds` 的第二、三条
    条件不会互相打架。谁把某条判据改成看整段结尾，这条会当场变红。
    """
    for case in CASES:
        session = session_of(case.tags())
        for tag in features.TAG_IDS:
            assert judges.quote_holds(tag, quote_for(tag), session) is (tag in case.tags())




class _OverclaimBackend:
    """真话之外**多报一条 D1** 的假后端。

    字段：`report` 真话（由 `trace_report` 推出来）、`quote` 给那条多报的 D1 配的引文——分别验
    「引文在会话里但按判据不算命中」与「引文在会话里根本不存在」两条路径。
    """

    def __init__(self, case: DebugStepCase, quote: str) -> None:
        """输入：注入特征、误报的引文。输出：实例。"""
        self.report = trace_report(case)
        self.quote = quote

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：真话 + 一条 D1 误报。预期行为：不联网。"""
        del task
        tags = [*self.report, {"tag": "D1", "quote": self.quote}]
        return _draft({"tags": tags})
