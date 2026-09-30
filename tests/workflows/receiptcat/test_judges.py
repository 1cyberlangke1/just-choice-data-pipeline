"""receiptcat 三个判别器的单元测试：该拦的拦得住，端点故障只算「没判成」。

输入：`workflows.receiptcat.judges` 与 `conftest` 里的假后端。输出：无（断言）。
预期行为：`hygiene` / `frame` 是零 token 的口径权威，`trace` 是读者视角的复核；把「模型
写歪」「渲染挂错 case」「端点抖动」三种情况分开判，别混成一个结论。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import TransportError
from jc.jsontypes import JsonValue
from tests.workflows.receiptcat.conftest import FakeBackend, faithful, sample_of, spec_for, text_of
from workflows.receiptcat import features, judges, locales, render
from workflows.receiptcat.features import ReceiptCase

#: 两种语言，参数化用。
LANGS: Final[tuple[str, ...]] = ("zh", "en")

#: 覆盖三档与两个诱饵取值的 case 表。
CASES: Final[tuple[ReceiptCase, ...]] = (
    ReceiptCase("C1", "under", "none"),
    ReceiptCase("C2", "mild", "none"),
    ReceiptCase("C4", "heavy", "near"),
)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("case", CASES)
def test_hygiene_passes_a_faithful_sample(case: ReceiptCase, lang: str) -> None:
    """输入：一条兑现计划的收据行。输出：无。预期行为：`hygiene` 通过。"""
    spec, sample = faithful(case, lang)
    verdict = judges.HygieneJudge(lang).judge(sample, spec)
    assert verdict.ok and not verdict.skipped


def test_hygiene_without_a_spec_is_skipped() -> None:
    """输入：没有真值规格。输出：无。

    预期行为：记成「没判成」而不是「不通过」——没有 spec 就无从兑现，判成不通过会把
    「这条没被检查」写成「这条是坏数据」。
    """
    _spec, sample = faithful(ReceiptCase("C1", "under", "none"))
    verdict = judges.HygieneJudge("zh").judge(sample, None)
    assert verdict.skipped and verdict.ok


@pytest.mark.parametrize(
    "text",
    [
        "咖啡 报销明细 168.00",
        "打车 报销明细 254.00",
        "打车 顺风车 报销明细 168.00",
        "打车 交通出行 报销明细 168.00",
    ],
)
def test_hygiene_rejects_broken_text(text: str) -> None:
    """输入：几种写歪的收据行。输出：无。预期行为：一律不通过，理由非空。"""
    case = ReceiptCase("C1", "under", "none")
    verdict = judges.HygieneJudge("zh").judge(sample_of(text), spec_for(case))
    assert not verdict.ok and not verdict.skipped
    assert verdict.reason


def test_hygiene_rejects_format_leftovers_in_the_state() -> None:
    """输入：state 里带了围栏。输出：无。

    预期行为：不通过——判别器兜的是「不是本 workflow 渲染出来的 state」，格式残留是最常见
    的那一种（模型把整段围起来交回来）。
    """
    case = ReceiptCase("C1", "under", "none")
    state = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    verdict = judges.HygieneJudge("zh").judge({"state": f"```text\n{state}\n```"}, spec_for(case))
    assert not verdict.ok
    assert "残留" in verdict.reason


def test_hygiene_rejects_a_state_without_a_receipt_line() -> None:
    """输入：没有收据行抬头的 state。输出：无。预期行为：不通过（反解不出正文）。"""
    verdict = judges.HygieneJudge("zh").judge({"state": "空空的"}, spec_for(ReceiptCase("C1", "under", "none")))
    assert not verdict.ok


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("case", CASES)
def test_frame_passes_a_faithful_sample(case: ReceiptCase, lang: str) -> None:
    """输入：一条规范渲染的 state。输出：无。预期行为：`frame` 通过。"""
    _spec, sample = faithful(case, lang)
    verdict = judges.FrameJudge(lang).judge(sample, None)
    assert verdict.ok and not verdict.skipped


@pytest.mark.parametrize(
    ("broken", "expect"),
    [
        ("", "骨架"),
        ("【归类表】\n- 交通出行：打车\n【收据行】\n打车 168.00", "归类表"),
    ],
)
def test_frame_rejects_a_broken_skeleton(broken: str, expect: str) -> None:
    """输入：骨架不全的 state。输出：无。预期行为：不通过，理由点出缺的是哪一块。"""
    verdict = judges.FrameJudge("zh").judge({"state": broken}, None)
    assert not verdict.ok
    assert expect in verdict.reason or "应为" in verdict.reason


def test_frame_rejects_a_tampered_cap_table() -> None:
    """输入：把上限表里的一行删掉。输出：无。预期行为：不通过（上限表不再是 C1~C4）。"""
    case = ReceiptCase("C1", "under", "none")
    state = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    lines = [line for line in state.splitlines() if "餐饮招待：150.00" not in line]
    verdict = judges.FrameJudge("zh").judge({"state": "\n".join(lines)}, None)
    assert not verdict.ok
    assert "上限表" in verdict.reason


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("case", CASES)
def test_trace_passes_a_faithful_sample(case: ReceiptCase, lang: str) -> None:
    """输入：假后端的忠实反解。输出：无。预期行为：`trace` 通过。

    假后端按 `task.spec` 里的注入特征答，等于「照着计划答」——它验的是接线（提示词、期望值、
    语言选取），不是模型能力。
    """
    spec, sample = faithful(case, lang)
    verdict = judges.TraceJudge(FakeBackend(locale=lang), lang).judge(sample, spec)
    assert verdict.ok and not verdict.skipped


def test_trace_catches_a_wrong_answer() -> None:
    """输入：反解说反话。输出：无。预期行为：不通过，理由里带上计划与反解两边的值。"""
    spec, sample = faithful(ReceiptCase("C2", "mild", "none"))
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "计划是" in verdict.reason


def test_trace_treats_endpoint_failure_as_not_judged() -> None:
    """输入：反解时抛传输错误。输出：无。

    预期行为：`skipped=True`、`ok=True`——端点抖动不是数据质量问题；但**全部 provider 都挂**
    那种致命错误必须往上抛（整批该停下）。
    """
    spec, sample = faithful(ReceiptCase("C1", "under", "none"))
    soft = judges.TraceJudge(FakeBackend(trace_raises=TransportError("超时")), "zh")
    verdict = soft.judge(sample, spec)
    assert verdict.skipped and verdict.ok
    assert "端点故障" in verdict.reason

    fatal = TransportError("全挂了")
    fatal.fatal_for_run = True
    with pytest.raises(TransportError):
        judges.TraceJudge(FakeBackend(trace_raises=fatal), "zh").judge(sample, spec)


def test_trace_without_a_spec_is_skipped() -> None:
    """输入：没有真值规格。输出：无。预期行为：记成「没判成」。"""
    _spec, sample = faithful(ReceiptCase("C1", "under", "none"))
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, None)
    assert verdict.skipped


#: 反解草稿写歪的几种形态（值, 期望是否算「没判成」）。
TRACE_GARBAGE: Final[tuple[dict[str, JsonValue], ...]] = (
    {"category": "不存在的类", "tier": 0},
    {"category": "交通出行", "tier": 9},
    {"category": 3, "tier": 0},
    {"category": "交通出行"},
)


@pytest.mark.parametrize("payload", TRACE_GARBAGE)
def test_trace_skips_a_malformed_answer(payload: dict[str, JsonValue]) -> None:
    """输入：几种写歪的反解草稿。输出：无。

    预期行为：一律记成「没判成」——认不出的类别名、越界的下标、缺字段都是**反解格式没对齐**，
    判成「不通过」会把格式漂移写成数据质量问题（而数据本身可能是对的）。
    """
    class Garbage:
        """只回一份固定草稿的假后端。"""

        def generate(self, task: object) -> object:
            """输入：任务。输出：固定草稿。预期行为：忽略任务内容。"""
            del task
            return type(
                "Draft",
                (),
                {"data": payload, "raw_text": "{}", "backend": "g", "provider": "g", "model": "g", "mode": "g"},
            )()

    spec, sample = faithful(ReceiptCase("C1", "under", "none"))
    verdict = judges.TraceJudge(Garbage(), "zh").judge(sample, spec)  # type: ignore[arg-type]
    assert verdict.skipped and verdict.ok


def test_trace_uses_the_state_own_language() -> None:
    """输入：en 的 state，但判别器配的是 zh。输出：无。

    预期行为：按 **state 自己的语言**取类别名——一次 run 里混了 zh / en 时，拿调用方给的那
    一种去比会把对的判成错的（review / browser / routing / rubric 四家都踩过这个坑）。
    """
    case = ReceiptCase("C3", "heavy", "near")
    _spec, sample = faithful(case, "en")
    verdict = judges.TraceJudge(FakeBackend(locale="en"), "zh").judge(sample, _spec)
    assert verdict.ok and not verdict.skipped


def test_hygiene_reports_the_field_when_the_draft_lacks_text() -> None:
    """输入：state 里收据行为空。输出：无。预期行为：不通过（空正文不算兑现计划）。"""
    case = ReceiptCase("C1", "under", "none")
    state = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    empty = state.replace(text_of(case), "")
    verdict = judges.HygieneJudge("zh").judge({"state": empty}, spec_for(case))
    assert not verdict.ok
    assert "收据行" in verdict.reason


def test_features_label_is_used_by_the_judge() -> None:
    """输入：一个 case。输出：无。预期行为：判别器用的标签与 `features.label_for` 同一份。"""
    case = ReceiptCase("C4", "heavy", "near")
    labels = features.label_for(case)
    assert (labels.over, labels.tier, labels.near) == (True, "finance", True)
