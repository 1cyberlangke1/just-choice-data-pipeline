"""文档包切分的渲染层测试：state 骨架、提示词、渲染期校验与反解。"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from tests.workflows.segmentation.conftest import (
    MARKERS,
    PAD,
    TRACES,
    pages_of,
    page_text,
    spec_for,
)
from workflows.segmentation import features, locales, render
from workflows.segmentation.features import SegmentationCase

CASE = SegmentationCase("form+receipt", "3", features.SOURCE_SCAN)


def test_class_line_carries_the_id_and_the_rule() -> None:
    """输入：一种语言。输出：无。预期行为：编号排在最前（解析要靠它切回来），判据原文也在行里。"""
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        line = render.class_line(loc, "receipt")
        assert line.startswith("receipt")
        assert str(loc["class"]["receipt"]["rule"]) in line
        assert str(loc["class"]["receipt"]["name"]) in line


def test_rule_lines_split_the_classes_by_mergeability() -> None:
    """输入：一种语言。输出：无。

    预期行为：三条规则（跨类一定切、可并的并、不可并的一页一份），两条分组规则里的类名与
    `features.MERGEABLE` 现算出来的分组一致——不在语言文件里另抄一份，两边不可能各说各的。
    """
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        lines = render.rule_lines(loc)
        assert len(lines) == 3
        mergeable = [cls for cls in features.CLASS_IDS if cls in features.MERGEABLE]
        each = [cls for cls in features.CLASS_IDS if cls not in features.MERGEABLE]
        for group, line in ((mergeable, lines[1]), (each, lines[2])):
            for cls in group:
                assert str(loc["class"][cls]["name"]) in line


def test_page_marker_round_trips_in_every_language() -> None:
    """输入：一种语言 × 若干页号。输出：无。

    预期行为：`page_mark` 写出来的那一行，`page_marker_re` 必须认得出、且认出的页号对得上；
    两种语言的抬头互不误认（zh 的正则不该认 en 的抬头）。
    """
    marks: dict[str, str] = {}
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        pattern = render.page_marker_re(loc)
        for index in (1, 2, 10):
            mark = render.page_mark(loc, index)
            match = pattern.match(mark)
            assert match and int(match.group(1)) == index
        marks[lang] = render.page_mark(loc, 1)
    assert marks["zh"] != marks["en"]
    assert not render.page_marker_re(locales.load("zh")).match(marks["en"])
    assert not render.page_marker_re(locales.load("en")).match(marks["zh"])


def test_page_bounds_are_calibrated_per_language() -> None:
    """输入：两种语言。输出：无。

    预期行为：两边的上下限都是正数且下界小于上界，而且**英文的上限比中文宽**——同样一页纸，
    英文的字符数天然更多（真机实测 zh 中位数 131、en 中位数 212），一份共用上限会把英文顶在
    上限上（首轮 en 侧 1 条 295 字符被拒，而 zh 侧最大才 186）。
    """
    zh_low, zh_high = render.page_bounds(locales.load("zh"))
    en_low, en_high = render.page_bounds(locales.load("en"))
    for low, high in ((zh_low, zh_high), (en_low, en_high)):
        assert 0 < low < high
    assert en_high > zh_high


def test_page_bounds_reject_broken_locales() -> None:
    """输入：缺字段 / 不是整数 / 下界不小于上界的语言字典。输出：无。预期行为：都当场报错。"""
    with pytest.raises(ValueError, match="不是整数"):
        render.page_bounds({"state": {"page_min": "60", "page_max": 260}})
    with pytest.raises(ValueError, match="不是整数"):
        render.page_bounds({"state": {"page_min": 60}})
    with pytest.raises(ValueError, match="下界必须小于上界"):
        render.page_bounds({"state": {"page_min": 260, "page_max": 260}})


def test_build_prompt_pins_language_type_and_source() -> None:
    """输入：一种语言 × 一个计划。输出：无。

    预期行为：提示词里有语言约束、每一页属于哪一类与判据原文、不许出现的类抬头、来源痕迹、
    长度区间；**一个可抄的成品页都没有**（范式 §9.45）。
    """
    plan = features.plan_for(CASE, 0)
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        prompt = render.build_prompt(loc, plan, CASE.source)
        assert str(loc["language_name"]) in prompt
        assert str(loc["prompt"]["lang_rule"]).format(language=loc["language_name"]) in prompt
        assert str(render.source_copy(loc, CASE.source)["trace"]) in prompt
        for cls in plan.classes:
            assert str(render.class_copy(loc, cls)["rule"]) in prompt
        for cls in features.CLASS_IDS:
            if cls not in plan.classes:
                assert str(render.class_copy(loc, cls)["avoid"]) in prompt
        assert str(plan.shape.count("form")) in prompt


def test_check_draft_accepts_every_faithful_packet() -> None:
    """输入：全空间 126 组 case 的忠实页面。输出：无。预期行为：一条都不报错。"""
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        for case in features.build_cases(512):
            plan = features.plan_for(case, 0)
            render.check_draft(render.PacketDraft(pages=pages_of(plan, case.source)), plan, case.source, loc)


def test_check_draft_rejects_a_wrong_page_count() -> None:
    """输入：少一页 / 多一页。输出：无。预期行为：两种都抛错，理由里说清页数。"""
    plan = features.plan_for(CASE, 0)
    loc = locales.load("zh")
    pages = pages_of(plan, CASE.source)
    with pytest.raises(DraftError, match="页数不对"):
        render.check_draft(render.PacketDraft(pages=pages[:-1]), plan, CASE.source, loc)
    with pytest.raises(DraftError, match="页数不对"):
        render.check_draft(render.PacketDraft(pages=[*pages, pages[0]]), plan, CASE.source, loc)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda pages: ["", *pages[1:]], "空"),
        (lambda pages: ["短", *pages[1:]], "太短"),
        # 400 个字符比两种语言的上限都长（zh 260 / en 380），这条尺子对两边的口径都成立。
        (lambda pages: ["会" * 400, *pages[1:]], "太长"),
        (lambda pages: [f"```{pages[0]}```", *pages[1:]], "围栏"),
        # 把抬头删掉：这一页一条判据都不命中，按字面算出来「哪一类都不属于」。
        (lambda pages: [pages[0].replace(MARKERS["form"], ""), *pages[1:]], "哪一类都不属于"),
        # 添上别的类的抬头：这一页同时命中两类，读者也分不清。
        (
            lambda pages: [pages[0] + MARKERS["contract"], *pages[1:]],
            "按字面算出来属于",
        ),
        # 把来源痕迹删掉：页面本身还是那一类，但这一包读起来不是从那个来源来的。
        (
            lambda pages: [pages[0].replace(TRACES[CASE.source], ""), *pages[1:]],
            "没有带来源",
        ),
    ],
)
def test_check_draft_rejects_broken_pages(
    mutate: Callable[[list[str]], list[str]], reason: str
) -> None:
    """输入：各种写歪的页面。输出：无。预期行为：全部当场抛错，理由里点名是哪一条。"""
    plan = features.plan_for(CASE, 0)
    loc = locales.load("zh")
    pages = pages_of(plan, CASE.source)
    broken = mutate(pages)
    with pytest.raises(DraftError, match=reason):
        render.check_draft(render.PacketDraft(pages=broken), plan, CASE.source, loc)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_check_draft_rejects_a_pasted_rule(lang: str) -> None:
    """输入：把判据原文**整段贴**进正文。输出：无。

    预期行为：抛 `DraftError`，理由是贴判据原文——提示词里唯一可抄的具体文字就是它，贴进去
    标签照样算命中（正则只看子串），但那段文本读起来不是材料（范式 §9.45）。
    """
    loc = locales.load(lang)
    plan = features.plan_for(CASE, 0)
    rule = str(render.class_copy(loc, "form")["rule"])
    pages = pages_of(plan, CASE.source)
    with pytest.raises(DraftError, match="判据原文"):
        render.check_draft(
            render.PacketDraft(pages=[f"{pages[0]}{rule}", *pages[1:]]), plan, CASE.source, loc
        )


def test_pasted_rules_only_flags_verbatim_text() -> None:
    """输入：整段贴 / 改了一个词的两种正文。输出：无。

    预期行为：只认**逐字相同**的整段判据原文——改一个词就不再算，否则正常产物会被误伤。
    """
    loc = locales.load("zh")
    rule = str(render.class_copy(loc, "form")["rule"])
    assert render.pasted_rules(rule, loc) == [rule]
    # 换掉的是**夹在中间**的那个词：把末尾的词加长反而仍包含原串（子串匹配不要求到结尾），
    # 那样这条断言就测了个寂寞。
    assert render.pasted_rules(rule.replace("填报人", "填报人员"), loc) == []
    assert render.pasted_rules(PAD, loc) == []


def test_to_state_and_parse_state_round_trip() -> None:
    """输入：一种语言 × 一个计划。输出：无。

    预期行为：state 里种类表、切分规则、页序抬头齐备；反解回来的逐页正文与入参逐字相同，
    种类表就是 `CLASS_IDS`。
    """
    plan = features.plan_for(CASE, 0)
    pages = pages_of(plan, CASE.source)
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        state = render.to_state(render.PacketDraft(pages=pages), loc)
        for mark in (loc["state"]["table_mark"], loc["state"]["rule_mark"], loc["state"]["order_mark"]):
            assert str(mark) in state
        parsed = render.parse_with(state, loc)
        assert parsed.pages == tuple(pages)
        assert parsed.classes == features.CLASS_IDS
        assert parsed.found


def test_parse_state_ignores_the_callers_language() -> None:
    """输入：zh 与 en 的 state。输出：无。

    预期行为：`parse_state` 自己认语言——拿 zh 的 state 去问 en 的解析器也要得到同一份结果，
    否则一次混语言的 run 里一半产物会解析成空。
    """
    plan = features.plan_for(CASE, 0)
    pages = pages_of(plan, CASE.source)
    for lang in ("zh", "en"):
        state = render.to_state(render.PacketDraft(pages=pages), locales.load(lang))
        assert render.detect_locale(state) == lang
        assert render.parse_state(state).pages == tuple(pages)


def test_parse_state_on_garbage_returns_nothing() -> None:
    """输入：不是 state 的东西。输出：无。预期行为：空结果，不抛异常。"""
    for value in (None, 42, "一段随便的文字"):
        parsed = render.parse_state(value)
        assert parsed.pages == ()
        assert parsed.classes == ()
        assert not parsed.found


def test_build_task_carries_the_injected_features() -> None:
    """输入：真值规格。输出：无。

    预期行为：任务 id 可重算、`spec` 里带上注入特征（假后端与真后端都能照着它答）、响应模型是
    草稿模型。
    """
    spec = spec_for(CASE)
    task = render.build_task(spec, locales.load("zh"))
    assert task.task_id == "segmentation-train-0000001-render"
    assert task.domain == "segmentation"
    assert task.response_model is render.PacketDraft
    assert task.spec["case"] == {"classes": "form+receipt", "pages": "3", "source": "scan"}
    assert task.spec["shape"] == ["form", "form", "receipt"]
    assert task.spec["source"] == "scan"


def test_class_copy_reports_missing_fields() -> None:
    """输入：缺字段的语言字典。输出：无。预期行为：抛 ValueError，点名缺了哪个字段。"""
    with pytest.raises(ValueError, match="缺种类 form"):
        render.class_copy({"class": {}}, "form")
    broken: dict[str, JsonValue] = {"class": {"form": {"name": "表单"}}}
    with pytest.raises(ValueError, match="缺字段 rule"):
        render.class_copy(broken, "form")


def test_source_copy_reports_a_missing_shape() -> None:
    """输入：缺来源的语言字典。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="缺来源 fax"):
        render.source_copy({"source": {"scan": {"desc": "扫描件", "trace": "扫描页码"}}}, "fax")


def test_a_single_page_packet_renders_and_parses() -> None:
    """输入：一页的包。输出：无。

    预期行为：最短的那一档（1 页 1 类）也要能渲染、反解、判通过——边界档位不能只在采样器里
    存在、渲染期却挂掉。
    """
    case = SegmentationCase("letter", "1", features.SOURCE_PHOTO)
    plan = features.plan_for(case, 0)
    pages = [page_text("letter", case.source)]
    render.check_draft(render.PacketDraft(pages=pages), plan, case.source, locales.load("zh"))
    state = render.to_state(render.PacketDraft(pages=pages), locales.load("zh"))
    assert render.parse_state(state).pages == tuple(pages)
