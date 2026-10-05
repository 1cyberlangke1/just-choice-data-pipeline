"""文本外壳测试：提示词该说的话都说了、state 能来回转、写歪的片段在渲染期就被拦下。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from tests.workflows.adblock.conftest import BASE, spec_for, text_of
from workflows.adblock import features, locales, render
from workflows.adblock.features import AdBlockCase

#: 一个普通 case：命中两条，元素长在文章正文页上。
CASE = AdBlockCase("A1+A3", features.SURFACE_ARTICLE)


def test_prompt_says_none_may_be_tripped_when_the_plan_is_empty() -> None:
    """输入：空命中的 case。输出：无。预期行为：改成「一条都不许踩」的说法，且不出现要求段落。"""
    loc = locales.load("zh")
    case = AdBlockCase("", features.SURFACE_ARTICLE)
    prompt = render.build_prompt(loc, case)
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt


def test_prompt_states_the_surface_shape() -> None:
    """输入：一种语言 × 一种页面类型。输出：无。预期行为：提示词里点明这个元素长在哪种页面上。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["en", "zh"]:
        for surface in ["article", "feed", "video"]:
            try:
                loc = locales.load(lang)
                case = AdBlockCase("A2", surface)
                prompt = render.build_prompt(loc, case)
                assert render.surface_copy(loc, surface) in prompt
            except AssertionError as exc:
                bad.append(f"{(lang, surface)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_prompt_has_no_unfilled_slots() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：渲染完一个花括号都不剩。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case)
        assert "{" not in prompt and "}" not in prompt, case.describe()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：写出去再读回来，判据表与片段一字不差。"""
    loc = locales.load(lang)
    snippet = text_of(CASE.tags())
    state = render.to_state(render.ContentDraft(snippet=snippet), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.tags == features.TAG_IDS
    assert parsed.text == snippet
    assert render.detect_locale(state) == lang


def test_state_keeps_the_tag_table_and_the_thresholds() -> None:
    """输入：一段 state。输出：无。

    预期行为：判据表四行与档位表四行都印在 state 里，档位表那几行用的是与题面**同一份**处置文案。
    """
    loc = locales.load("zh")
    state = render.to_state(render.ContentDraft(snippet=text_of(CASE.tags())), loc)
    for tag in features.TAG_IDS:
        assert render.tag_line(loc, tag) in state
    for level in features.ACTS:
        assert render.threshold_line(loc, level) in state


def test_state_prints_each_criterion_rule() -> None:
    """输入：两种语言 × 一段 state。输出：无。

    预期行为：判据表每行都带上该判据的**原文**——只印判据名时，反解模型认不出「广告位命名」
    指的是什么形状（范式 §9.44，在 `memorygate` 上量过 1/8 → 8/8）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        state = render.to_state(render.ContentDraft(snippet=text_of(CASE.tags())), loc)
        for tag in features.TAG_IDS:
            assert str(render.tag_copy(loc, tag)["rule"]) in state


def test_multiline_snippet_round_trips() -> None:
    """输入：带换行的片段。输出：无。预期行为：多行原样保住。"""
    loc = locales.load("zh")
    snippet = text_of(CASE.tags()) + '\n<aside class="ad-banner">再来一条</aside>'
    state = render.to_state(render.ContentDraft(snippet=snippet), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.text == snippet
    assert parsed.tags == features.TAG_IDS


def test_snippet_mentioning_a_rule_does_not_break_the_parse() -> None:
    """输入：片段里出现一行「像判据表」的内容。输出：无。

    预期行为：片段抬头之后的一切都当正文——否则片段里出现一行 `- A9：假的` 就会把判据表数歪。
    """
    loc = locales.load("zh")
    snippet = text_of(CASE.tags()) + "\n- A9：这一行是正文，不是判据表"
    parsed = render.parse_with(render.to_state(render.ContentDraft(snippet=snippet), loc), loc)
    assert parsed.tags == features.TAG_IDS
    assert "- A9：这一行是正文，不是判据表" in str(parsed.text)


@pytest.mark.parametrize("garbage", [None, 123, "随便一段话", ["a"], {"state": "x"}])
def test_parse_state_gives_up_on_unknown_input(garbage: object) -> None:
    """输入：认不出抬头的各种输入。输出：无。预期行为：返回空结果，不抛异常。"""
    parsed = render.parse_state(garbage)
    assert parsed.text is None
    assert parsed.tags == ()


def test_check_draft_accepts_every_faithful_snippet() -> None:
    """输入：全空间 48 组 case 的忠实片段。输出：无。预期行为：一条都不报错。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        render.check_draft(render.ContentDraft(snippet=text_of(case.tags())), case, loc)


@pytest.mark.parametrize("tags", [("A1",), ("A1", "A2", "A4")])
def test_check_draft_rejects_broken_snippet(tags: tuple[str, ...]) -> None:
    """输入：少写一条 / 多写一条的片段。输出：无。

    预期行为：抛 `DraftError`，理由里说清「算出来的组合」与「计划要的」各是什么。
    """
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(render.ContentDraft(snippet=text_of(tags)), CASE, locales.load("zh"))


def test_check_draft_rejects_empty_overlong_fence_and_too_short() -> None:
    """输入：空 / 超长 / 带围栏 / 太短 / 没有标签的五种片段。输出：无。

    预期行为：五种都当场抛错，且各自命中该有的那道闸（顺序：空 → 围栏 → 不是 HTML → 太短 →
    太长）——闸门顺序搞反的话，报出来的理由会指错方向。
    """
    case = AdBlockCase("", features.SURFACE_ARTICLE)
    loc = locales.load("zh")
    with pytest.raises(DraftError, match="空"):
        render.check_draft(render.ContentDraft(snippet="   "), case, loc)
    with pytest.raises(DraftError, match="围栏"):
        render.check_draft(render.ContentDraft(snippet=f"```{BASE}```"), case, loc)
    with pytest.raises(DraftError, match="不是一段 HTML"):
        render.check_draft(render.ContentDraft(snippet="x" * (render.MIN_CONTENT_CHARS + 5)), case, loc)
    with pytest.raises(DraftError, match="太短"):
        render.check_draft(render.ContentDraft(snippet="<p>短</p>"), case, loc)
    with pytest.raises(DraftError, match="太长"):
        overlong = "<p>" + "会" * render.MAX_CONTENT_CHARS + "</p>"
        render.check_draft(render.ContentDraft(snippet=overlong), case, loc)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_check_draft_rejects_a_pasted_rule(lang: str) -> None:
    """输入：把判据原文**整段贴**进片段。输出：无。

    预期行为：抛 `DraftError`，理由是贴判据原文——提示词里唯一可抄的具体文字就是它，贴进去
    标签照样算命中（正则只看子串），但那段片段读起来不是 HTML（范式 §9.45）。
    """
    loc = locales.load(lang)
    rule = str(render.tag_copy(loc, "A4")["rule"])
    case = AdBlockCase("A4", features.SURFACE_ARTICLE)
    with pytest.raises(DraftError, match="判据原文"):
        render.check_draft(render.ContentDraft(snippet=f"{BASE} {rule}"), case, loc)


def test_threshold_line_reuses_the_act_copy() -> None:
    """输入：一种语言 × 四个档位。输出：无。预期行为：档位表与 `act` 那一族的选项文案同源。"""
    loc = locales.load("zh")
    for level in features.ACTS:
        assert str(loc["fam"]["act"]["opt"][level]) in render.threshold_line(loc, level)


def test_surface_copy_reports_a_missing_shape() -> None:
    """输入：缺页面类型的语言字典。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="页面类型 feed"):
        render.surface_copy({"surface": {"article": {"desc": "正文"}}}, "feed")


@pytest.mark.parametrize(
    ("snippet", "expect"),
    [
        ('<div class="post-body"><p>第一段</p><p>第二段</p></div>', "第一段 第二段"),
        ('<span class="note">赞助内容</span>', "赞助内容"),
        ('<div style="display:none" aria-hidden="true"></div>', ""),
        ('<div class="ad-slot">\n  推荐阅读\n</div>', "推荐阅读"),
        ('<div>a</div> 尾巴', "a 尾巴"),
    ],
)
def test_visible_text_keeps_only_the_text_between_tags(snippet: str, expect: str) -> None:
    """输入：几种片段。输出：无。

    预期行为：只留标签之间那一块，属性值 / 标签名 / 属性名一律不算——A2 那条判据对「可见文字」
    的口径与它一致，两处说的是同一件事。
    """
    assert render.visible_text(snippet) == expect
