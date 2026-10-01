"""判别器测试：零 token 的两个各拦各的，模型那一个只管反解。"""

from __future__ import annotations

import json
import pytest

from jc.backend.types import AllProvidersDown, TransportError
from jc.jsontypes import JsonValue

from tests.workflows.segmentation.conftest import (
    FakeBackend,
    MARKERS,
    TRACES,
    faithful,
    pages_of,
    page_text,
)
from workflows.segmentation import features, judges, locales, render
from workflows.segmentation.features import SegmentationCase

CASE = SegmentationCase("form+receipt", "3", features.SOURCE_SCAN)


def _sample(pages: list[str], lang: str = "zh") -> dict[str, JsonValue]:
    """输入：逐页正文、语言。输出：只带 state 的样本。"""
    return {"state": render.to_state(render.PacketDraft(pages=pages), locales.load(lang))}


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_hygiene_passes_on_a_faithful_packet(lang: str) -> None:
    """输入：完全兑现计划的一包。输出：无。预期行为：通过，且不是「没判成」。"""
    spec, sample = faithful(CASE, lang)
    verdict = judges.HygieneJudge(lang).judge(sample, spec)
    assert verdict.ok and not verdict.skipped


def test_hygiene_needs_a_spec_and_a_state() -> None:
    """输入：没有 spec / 没有 state。输出：无。预期行为：两种都记「没判成」，不记不通过。"""
    judge = judges.HygieneJudge("zh")
    assert judge.judge(_sample([page_text("form", features.SOURCE_SCAN)]), None).skipped
    assert judge.judge({}, None).skipped
    assert judge.judge({}, "别的 workflow 的 spec").skipped


def test_hygiene_rejects_a_page_of_the_wrong_class() -> None:
    """输入：把最后一页的抬头换成别的类。输出：无。预期行为：不通过，理由里列两边的逐页种类。"""
    spec, _ = faithful(CASE)
    pages = pages_of(spec.plan, CASE.source)
    pages[-1] = page_text("letter", CASE.source)
    verdict = judges.HygieneJudge("zh").judge(_sample(pages), spec)
    assert not verdict.ok and not verdict.skipped
    assert "计划要的是" in verdict.reason


def test_hygiene_rejects_a_page_without_the_source_trace() -> None:
    """输入：把某一页的来源痕迹删掉。输出：无。预期行为：不通过，理由里点名是哪一页。"""
    spec, _ = faithful(CASE)
    pages = pages_of(spec.plan, CASE.source)
    pages[-1] = pages[-1].replace(TRACES[CASE.source], "")
    verdict = judges.HygieneJudge("zh").judge(_sample(pages), spec)
    assert not verdict.ok
    assert "来源痕迹" in verdict.reason


def test_hygiene_rejects_a_pasted_rule() -> None:
    """输入：正文里整段贴了判据原文。输出：无。预期行为：不通过，理由里点名判据原文。"""
    spec, _ = faithful(CASE)
    loc = locales.load("zh")
    pages = pages_of(spec.plan, CASE.source)
    pages[0] = f"{pages[0]}{loc['class']['form']['rule']}"
    verdict = judges.HygieneJudge("zh").judge(_sample(pages), spec)
    assert not verdict.ok
    assert "判据原文" in verdict.reason


@pytest.mark.parametrize("leftover", ["```", "<br>"])
def test_hygiene_rejects_format_leftovers(leftover: str) -> None:
    """输入：state 里混进围栏 / HTML 换行标签。输出：无。预期行为：不通过，理由点名残留。"""
    spec, sample = faithful(CASE)
    sample["state"] = f"{sample['state']}\n{leftover}"
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "格式残留" in verdict.reason or "HTML" in verdict.reason


def test_frame_passes_and_reports_broken_skeletons() -> None:
    """输入：好骨架 / 缺一行的种类表 / 页数对不上的 state。输出：无。

    预期行为：好骨架通过；缺一行时点名种类表；页数对不上时点名页数——两种都说明渲染挂错了 case。
    """
    spec, sample = faithful(CASE)
    judge = judges.FrameJudge("zh")
    assert judge.judge(sample, spec).ok

    state = str(sample["state"])
    broken = "\n".join(line for line in state.splitlines() if not line.startswith("- contract"))
    verdict = judge.judge({"state": broken}, spec)
    assert not verdict.ok
    assert "种类表" in verdict.reason

    short = state.replace(render.page_mark(locales.load("zh"), 3), "")
    verdict = judge.judge({"state": short}, spec)
    assert not verdict.ok
    assert "页数" in verdict.reason


def test_frame_skips_without_a_spec() -> None:
    """输入：没有 spec。输出：无。预期行为：记「没判成」。"""
    assert judges.FrameJudge("zh").judge(_sample([page_text("form", features.SOURCE_SCAN)]), None).skipped


def test_read_judge_accepts_a_faithful_read(backend: FakeBackend) -> None:
    """输入：假后端照计划反解。输出：无。预期行为：通过，`raw` 里留档读出来的种类。"""
    spec, sample = faithful(CASE)
    verdict = judges.ReadJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw["read_classes"] == ["form", "receipt"]


def test_read_judge_catches_a_lie() -> None:
    """输入：反解漏报一类。输出：无。预期行为：不通过，理由里列「计划的」与「反解出来的」。"""
    spec, sample = faithful(CASE)
    verdict = judges.ReadJudge(FakeBackend(read_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "反解读成" in verdict.reason


def test_read_judge_drops_a_quote_that_does_not_hold() -> None:
    """输入：反解报了一类，但引文抄错了。输出：无。

    预期行为：**误报不算命中**（引文站不住就按没出现处理），所以结论与计划对不上 → 不通过，
    理由里带上「已按没出现处理」的说明。
    """
    spec, sample = faithful(CASE)

    class LyingQuote(FakeBackend):
        """把引文改坏：报出来的种类还在，但抄的那段话不在正文里。"""

        def read_report(self, classes: list[str]) -> list[dict[str, str]]:
            """输入：计划里出现过的种类。输出：引文被改坏的草稿。"""
            return [{"class_id": cls, "quote": f"改过的引文 {cls}"} for cls in classes]

    verdict = judges.ReadJudge(LyingQuote(), "zh").judge(sample, spec)
    assert not verdict.ok
    assert "已按没出现处理" in verdict.reason


def test_read_judge_skips_on_endpoint_and_format_trouble() -> None:
    """输入：端点故障 / 不认识的编号 / 缺字段。输出：无。

    预期行为：全部记「没判成」而不是「不通过」——那是端点抖动或格式漂移，不是数据质量问题。
    """
    spec, sample = faithful(CASE)
    assert judges.ReadJudge(FakeBackend(read_raises=TransportError("抖了")), "zh").judge(
        sample, spec
    ).skipped

    class BadId(FakeBackend):
        """反解报一个不认识的编号。"""

        def read_report(self, classes: list[str]) -> list[dict[str, str]]:
            """输入：计划里出现过的种类。输出：编号被改坏的草稿。"""
            return [{"class_id": "unknown", "quote": "x"}]

    assert judges.ReadJudge(BadId(), "zh").judge(sample, spec).skipped

    class BadShape(FakeBackend):
        """反解草稿的 classes 字段不是列表。"""

        def generate(self, task: object) -> object:
            """输入：任务。输出：字段形状不对的草稿。"""
            draft = super().generate(task)
            if str(getattr(task, "task_id")).endswith("-read"):
                return type(draft)(
                    data={"classes": "form"},
                    raw_text=draft.raw_text,
                    backend=draft.backend,
                    provider=draft.provider,
                    model=draft.model,
                    mode=draft.mode,
                )
            return draft

    assert judges.ReadJudge(BadShape(), "zh").judge(sample, spec).skipped


def test_read_judge_lets_all_providers_down_through() -> None:
    """输入：反解时全部 provider 都挂了。输出：无。

    预期行为：`AllProvidersDown` 带着 `fatal_for_run` 直接抛出去（整批停下），不被吞成一条
    「没判成」的账。
    """
    spec, sample = faithful(CASE)
    judge = judges.ReadJudge(FakeBackend(read_raises=AllProvidersDown("全挂了")), "zh")
    with pytest.raises(AllProvidersDown):
        judge.judge(sample, spec)


def test_read_judge_skips_without_a_spec() -> None:
    """输入：没有 spec。输出：无。预期行为：记「没判成」，一次后端都不调。"""
    backend = FakeBackend()
    verdict = judges.ReadJudge(backend, "zh").judge(_sample([page_text("form", features.SOURCE_SCAN)]), None)
    assert verdict.skipped
    assert backend.tasks == []


@pytest.mark.parametrize(
    ("cls", "quote", "ok"),
    [
        ("form", MARKERS["form"], True),
        # 空白重排不算抄错（模型复述时常常把空格换成换行）。
        ("form", MARKERS["form"].replace(" ", "\n"), True),
        # 引文被装进 JSON 数组字符串（范式 §9.88）：拆成候选之后照样算数。
        ("form", json.dumps([MARKERS["form"]]), True),
        # 引文不在正文里 / 引文按正则不算这一类，都不算数。
        ("form", "正文里没有这句话", False),
        ("form", MARKERS["receipt"], False),
        ("form", "", False),
    ],
)
def test_quote_holds(cls: str, quote: str, ok: bool) -> None:
    """输入：种类、引文、正文。输出：无。预期行为：原样出现 **且** 引文合判据才算站得住。"""
    body = page_text("form", features.SOURCE_SCAN)
    assert judges.quote_holds(cls, quote, body) is ok


def test_read_prompt_carries_the_whole_state() -> None:
    """输入：一段 state。输出：无。预期行为：提示词里有整段 state 与那句「抄不出原文就不算」。"""
    spec, sample = faithful(CASE)
    loc = locales.load("zh")
    prompt = judges.build_read_prompt(str(sample["state"]), loc)
    assert str(sample["state"]) in prompt
    assert str(loc["verify"]["q"]) in prompt


def test_a_foreign_spec_is_never_judged_as_bad_data() -> None:
    """输入：没有 spec（或串了别的 workflow 的 spec）。输出：无。

    预期行为：三个判别器一律记「没判成」——把「判不了」写成「不通过」会把接线错误记成数据
    质量问题。
    """
    sample = _sample([page_text("form", features.SOURCE_SCAN)])
    foreign: object = "别的 workflow 的 spec"
    assert judges.HygieneJudge("zh").judge(sample, foreign).skipped
    assert judges.FrameJudge("zh").judge(sample, foreign).skipped
    assert judges.ReadJudge(FakeBackend(), "zh").judge(sample, foreign).skipped
