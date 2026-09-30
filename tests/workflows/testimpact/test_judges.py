"""testimpact 的判别器测试：零 token 的两把闸与一次后端反解。"""

from __future__ import annotations

import pytest

from jc.backend.types import BackendError, Draft, TransportError
from jc.jsontypes import JsonValue
from tests.workflows.testimpact.conftest import (
    TITLE_ZH,
    FakeBackend,
    diff_for,
    faithful,
    file_block,
    quote_for,
    sample_of,
    spec_for,
    title_for,
)
from workflows.testimpact import features, judges, locales, render
from workflows.testimpact.features import ImpactCase
from workflows.testimpact.workflow import ImpactWorkflow

LANGS = ("zh", "en")


def _state(touch: str, diff: str, lang: str = "zh", title: str | None = None) -> str:
    """输入：触及组合、改动、语言、可选标题。输出：一份 state（**不做形状校验**）。

    预期行为：判别器要能拦住「不是本 workflow 渲染出来的」state，所以这里故意绕过 `check_draft`。
    """
    draft = render.ContentDraft(title=title or title_for(lang), diff=diff)
    return render.to_state(draft, locales.load(lang))


def _draft(data: dict[str, JsonValue], raw: str = "scripted") -> Draft:
    """输入：反解草稿的数据。输出：`Draft`。预期行为：记账字段填假值。"""
    return Draft(
        data=data, raw_text=raw, backend="fake", provider="fake", model="fake", mode="fake"
    )


class _ScriptedTrace:
    """测试用：反解返回脚本里那一份结果（造「引文站不住」这类真后端不容易复现的形状）。"""

    def __init__(self, report: list[JsonValue]) -> None:
        """输入：要伪造的 files 列表。输出：实例。"""
        self.report = report

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：反解草稿。预期行为：只服务反解任务，渲染任务不该走到这里。"""
        del task
        return _draft({"files": self.report})


class _Fatal(BackendError):
    """测试用：全部 provider 都挂了的错误（`fatal_for_run` 是通用层唯一的判据）。"""

    fatal_for_run = True


@pytest.mark.parametrize("lang", LANGS)
def test_hygiene_accepts_every_faithful_sample(lang: str) -> None:
    """输入：16 组触及组合各一份完全兑现计划的样本。输出：无。预期行为：一条都不报。"""
    judge = judges.HygieneJudge(lang)
    for touch in features.TOUCH_SETS:
        spec, sample = faithful(ImpactCase(touch=touch, flavor="fix"), lang)
        verdict = judge.judge(sample, spec)
        assert verdict.ok, f"{lang}/{touch}: {verdict.reason}"


def test_hygiene_rejects_a_missing_and_an_extra_file() -> None:
    """输入：少改一个文件、多改一个文件。输出：无。预期行为：两种都报「落点对不上」。"""
    judge = judges.HygieneJudge("zh")
    only_p1 = "\n".join(file_block(features.PATH_FILES["P1"]))
    verdict = judge.judge(
        {"state": _state("P1+P2", only_p1)}, spec_for(ImpactCase(touch="P1+P2", flavor="fix"))
    )
    assert not verdict.ok and "落点" in verdict.reason
    two_files = only_p1 + "\n" + "\n".join(file_block(features.PATH_FILES["P2"]))
    verdict = judge.judge(
        {"state": _state("P1", two_files)}, spec_for(ImpactCase(touch="P1", flavor="fix"))
    )
    assert not verdict.ok and "落点" in verdict.reason


def test_hygiene_rejects_prose_fences_and_a_pasted_table() -> None:
    """输入：散文、围栏、贴了表行的 state。输出：无。预期行为：三种都被拦下。"""
    judge = judges.HygieneJudge("zh")
    case = ImpactCase(touch="P1", flavor="fix")
    spec = spec_for(case)
    prose = "这次改动调整了请求解析里的边界判断。" * 12
    assert not judge.judge({"state": _state("P1", prose)}, spec).ok
    fenced = f"```\n{diff_for(case)}\n```"
    assert not judge.judge({"state": _state("P1", fenced)}, spec).ok
    pasted = "\n".join(file_block(features.PATH_FILES["P1"])) + "\n+ " + render.coverage_line(
        locales.load("zh"), "P1"
    )
    assert not judge.judge({"state": _state("P1", pasted)}, spec).ok


def test_hygiene_skips_without_a_spec_or_a_text_state() -> None:
    """输入：没有 spec、state 不是字符串、spec 串了别的 workflow。输出：无。

    预期行为：三种都记「没判成」而不是「不通过」——那是接线问题，不是数据问题。
    """
    judge = judges.HygieneJudge("zh")
    spec = spec_for(ImpactCase(touch="P1", flavor="fix"))
    assert judge.judge({"state": "x"}, None).skipped
    assert judge.judge({"state": 42}, spec).skipped
    assert judge.judge({"state": "x"}, object()).skipped


def test_frame_accepts_faithful_and_rejects_a_broken_table() -> None:
    """输入：合规 state、覆盖表少一行、档位表顺序被打乱。输出：无。

    预期行为：合规放行；两种坏表各报各的——覆盖表与档位表是标签的依据，顺序错了标签就跟着错。
    """
    judge = judges.FrameJudge("zh")
    loc = locales.load("zh")
    case = ImpactCase(touch="P1", flavor="fix")
    spec, sample = faithful(case)
    assert judge.judge(sample, spec).ok
    state = _state("P1", diff_for(case))
    missing = state.replace(f"- {render.coverage_line(loc, 'P3')}\n", "")
    verdict = judge.judge({"state": missing}, spec)
    assert not verdict.ok and "覆盖表" in verdict.reason
    first = f"- {render.ladder_line(loc, features.LEVELS[0])}"
    last = f"- {render.ladder_line(loc, features.LEVELS[-1])}"
    swapped = state.replace(first, "@@TMP@@").replace(last, first).replace("@@TMP@@", last)
    verdict = judge.judge({"state": swapped}, spec)
    assert not verdict.ok and "档位表" in verdict.reason


def test_frame_skips_without_a_spec_or_a_text_state() -> None:
    """输入：没有 spec、state 不是字符串。输出：无。预期行为：两种都记「没判成」。"""
    judge = judges.FrameJudge("zh")
    spec = spec_for(ImpactCase(touch="P1", flavor="fix"))
    assert judge.judge({"state": "x"}, None).skipped
    assert judge.judge({"state": None}, spec).skipped


@pytest.mark.parametrize("lang", LANGS)
def test_trace_agrees_with_the_plan(lang: str) -> None:
    """输入：忠实假后端。输出：无。预期行为：反解与计划一致（两种语言各一遍）。"""
    spec, sample = faithful(ImpactCase(touch="P1+P3", flavor="feat"), lang)
    verdict = judges.TraceJudge(FakeBackend(locale=lang), lang).judge(sample, spec)
    assert verdict.ok, verdict.reason


def test_trace_catches_a_lie() -> None:
    """输入：反解漏报最后一个文件。输出：无。预期行为：判不通过，理由里列出两边。"""
    spec, sample = faithful(ImpactCase(touch="P1+P2", flavor="fix"))
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and "计划里改到" in verdict.reason


def test_trace_drops_an_overclaim_whose_quote_does_not_hold() -> None:
    """输入：反解报了两个文件，其中一条引文是编的。输出：无。

    预期行为：抄不出原文的那一条按「没改到」处理，剩下的落点与计划对不上 → 判不通过，理由里
    带上「误报」那句说明。
    """
    spec, sample = faithful(ImpactCase(touch="P1+P2", flavor="fix"))
    report: list[JsonValue] = [
        {"file": features.PATH_FILES["P1"], "quote": "这一行根本不在改动里"},
        {"file": features.PATH_FILES["P2"], "quote": quote_for("P2")},
    ]
    verdict = judges.TraceJudge(_ScriptedTrace(report), "zh").judge(sample, spec)
    assert not verdict.ok and "误报" in verdict.reason


def test_trace_marks_transport_errors_as_skipped() -> None:
    """输入：反解时端点故障。输出：无。预期行为：记「没判成」，绝不写成数据不合格。"""
    spec, sample = faithful(ImpactCase(touch="P1", flavor="fix"))
    verdict = judges.TraceJudge(FakeBackend(trace_raises=TransportError("炸了")), "zh").judge(
        sample, spec
    )
    assert verdict.ok and verdict.skipped


def test_trace_reraises_when_every_provider_is_down() -> None:
    """输入：反解时抛「整批该停下」的错误。输出：无。预期行为：原样抛出去，不吞。"""
    spec, sample = faithful(ImpactCase(touch="P1", flavor="fix"))
    with pytest.raises(_Fatal):
        judges.TraceJudge(FakeBackend(trace_raises=_Fatal("全挂了")), "zh").judge(sample, spec)


def test_trace_skips_an_unknown_file_and_a_malformed_draft() -> None:
    """输入：反解报了覆盖表之外的文件、草稿项不是对象。输出：无。预期行为：两种都记「没判成」。"""
    spec, sample = faithful(ImpactCase(touch="P1", flavor="fix"))
    unknown = judges.TraceJudge(
        _ScriptedTrace([{"file": "src/other.py", "quote": "x"}]), "zh"
    ).judge(sample, spec)
    assert unknown.skipped and "覆盖表之外" in unknown.reason
    malformed = judges.TraceJudge(_ScriptedTrace(["src/api/handlers.py"]), "zh").judge(sample, spec)
    assert malformed.skipped and "判不了" in malformed.reason


def test_trace_skips_without_a_spec() -> None:
    """输入：没有 spec。输出：无。预期行为：记「没判成」，不去调后端。"""
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge({"state": "x"}, None)
    assert verdict.skipped


def test_trace_prompt_carries_the_whole_state() -> None:
    """输入：一份 state。输出：无。预期行为：反解提示词里带着整段 state（含两张表与改动）。"""
    case = ImpactCase(touch="P2", flavor="fix")
    state = _state("P2", diff_for(case))
    prompt = judges.build_trace_prompt(state, locales.load("zh"))
    assert state in prompt
    assert features.PATH_FILES["P2"] in prompt


@pytest.mark.parametrize(
    ("pid", "quote", "expect"),
    [
        ("P1", "diff --git a/src/api/handlers.py b/src/api/handlers.py", True),
        ("P1", "+++ b/src/api/handlers.py", True),
        ("P1", "diff --git a/src/db/schema.py b/src/db/schema.py", False),
        ("P1", "", False),
        ("P1", "编的一段话", False),
    ],
)
def test_quote_holds_needs_both_the_quote_and_the_file_header(
    pid: str, quote: str, expect: bool
) -> None:
    """输入：几种引文。输出：无。

    预期行为：引文要**原样在改动里**、且**带着这个文件的头**——只抄一段别的文件里的代码不算。
    """
    diff = diff_for(ImpactCase(touch="P1+P2", flavor="fix"))
    assert judges.quote_holds(pid, quote, diff) is expect


def test_language_texts_reports_only_the_commit_title() -> None:
    """输入：一份样本。输出：无。

    预期行为：`language_texts` 只报模型写的提交标题——diff 是代码，报进去会把好数据当坏数据。
    """
    sample = sample_of(diff_for(ImpactCase(touch="P1", flavor="fix")), TITLE_ZH)
    assert ImpactWorkflow().language_texts(sample) == {"提交标题": TITLE_ZH}
    assert ImpactWorkflow().language_texts({"state": "没有结构"}) == {}
