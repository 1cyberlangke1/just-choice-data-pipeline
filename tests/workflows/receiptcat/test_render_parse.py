"""receiptcat 的渲染与反解：state 必须**往返一致**，草稿必须兑现注入特征。

输入：`workflows.receiptcat.render` 与 `conftest` 里的草稿构造器。
输出：无（断言）。预期行为：`to_state` 写出来的东西，`parse_state` 必须一字不差地读回来
（含语言判定）；`check_draft` 的每一条拒绝路径都要真的拦得住。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.workflow import DraftError
from tests.workflows.receiptcat.conftest import faithful, sample_of, spec_for, text_of
from workflows.receiptcat import features, locales, render
from workflows.receiptcat.features import ReceiptCase

#: 两种语言，参数化用。
LANGS: Final[tuple[str, ...]] = ("zh", "en")

#: 一个覆盖三种档位的 case 表（含诱饵那一档）。
CASES: Final[tuple[ReceiptCase, ...]] = (
    ReceiptCase("C1", "under", "none"),
    ReceiptCase("C2", "mild", "none"),
    ReceiptCase("C4", "heavy", "near"),
)


def test_state_round_trips() -> None:
    """输入：一个 case、一种语言。输出：无。

    预期行为：三张表与收据行原样读回来——归类表 / 上限表按 `CATEGORY_IDS` 顺序、档位表四行、
    收据行一字不差。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for case in CASES:
            try:
                text = text_of(case, lang)
                state = render.to_state(render.ContentDraft(text=text), locales.load(lang))
                parsed = render.parse_state(state)
                assert parsed.categories == features.CATEGORY_IDS
                assert parsed.caps == features.CATEGORY_IDS
                assert parsed.tiers == len(features.TIERS)
                assert parsed.text == text
                assert parsed.found
            except AssertionError as exc:
                bad.append(f"{(lang, case)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("lang", LANGS)
def test_detect_locale_picks_the_state_own_language(lang: str) -> None:
    """输入：一种语言的 state。输出：无。

    预期行为：认出这一种语言——审计就是靠它按 state 自己的语言取规范文案的（混语言 run 里用
    调用方语言去比会平白报错）。
    """
    state = render.to_state(render.ContentDraft(text=text_of(CASES[0], lang)), locales.load(lang))
    assert render.detect_locale(state) == lang


@pytest.mark.parametrize("state", [None, 42, "随便一段没有抬头的文本"])
def test_parse_state_gives_up_cleanly(state: object) -> None:
    """输入：不是本 workflow 的 state。输出：无。预期行为：不抛异常，收据行是 None。"""
    parsed = render.parse_state(state)
    assert parsed.text is None
    assert not parsed.found
    assert parsed.categories == ()


def test_cap_text_formats_cents() -> None:
    """输入：四个类别。输出：无。预期行为：上限按「整数 + 两位小数」写出来。"""
    assert render.cap_text("C1") == "200.00"
    assert render.cap_text("C2") == "150.00"
    assert render.cap_text("C4") == "2000.00"


@pytest.mark.parametrize("lang", LANGS)
def test_decoy_word_comes_from_another_category(lang: str) -> None:
    """输入：一种语言、四个类别。输出：无。

    预期行为：诱饵词取自**下一个类别**（环形），一定不是本条真值那一类的词——否则读者分不清
    「真命中」与「诱饵」，那一族题就没意义了。
    """
    loc = locales.load(lang)
    for index, cid in enumerate(features.CATEGORY_IDS):
        other = features.CATEGORY_IDS[(index + 1) % len(features.CATEGORY_IDS)]
        word = render.decoy_word_for(ReceiptCase(cid, "mild", "near"), loc)
        assert word == features.decoy_table(loc)[other][0]
        assert word not in features.keyword_table(loc)[cid]


def test_faithful_draft_passes() -> None:
    """输入：一条完全兑现计划的收据行。输出：无。预期行为：`check_draft` 不报错。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for case in CASES:
            try:
                render.check_draft(render.ContentDraft(text=text_of(case, lang)), case, locales.load(lang))
            except AssertionError as exc:
                bad.append(f"{(lang, case)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize(
    ("case", "broken", "expect"),
    [
        (ReceiptCase("C1", "under", "none"), "", "是空的"),
        (ReceiptCase("C1", "under", "none"), "```text\n打车 报销明细 168.00\n```", "围栏"),
        (ReceiptCase("C1", "under", "none"), "打车", "太短"),
        (ReceiptCase("C1", "under", "none"), "打车 报销明细 168.00 " + "很" * 120, "太长"),
        (ReceiptCase("C1", "under", "none"), "咖啡 报销明细 168.00", "计划要的是"),
        (ReceiptCase("C1", "under", "none"), "打车 咖啡 报销明细 168.00", "计划要的是"),
        (ReceiptCase("C1", "under", "none"), "打车 报销明细 254.00", "金额"),
        (ReceiptCase("C1", "under", "none"), "打车 报销明细 168.00 254.00", "金额"),
        (ReceiptCase("C1", "under", "none"), "打车 顺风车 报销明细 168.00", "诱饵"),
        (ReceiptCase("C1", "under", "near"), "打车 报销明细 168.00", "诱饵"),
        (ReceiptCase("C1", "under", "none"), "打车 交通出行 报销明细 168.00", "类别名"),
    ],
)
def test_check_draft_rejects(case: ReceiptCase, broken: str, expect: str) -> None:
    """输入：一条写歪的收据行与期望的理由片段。输出：无。

    预期行为：每一条拒绝路径都真的拦得住，而且理由里说得清差在哪——渲染期靠它决定重渲染，
    含糊的理由会让人分不清「模型写歪」还是「判据写错」。
    """
    with pytest.raises(DraftError, match=expect):
        render.check_draft(render.ContentDraft(text=broken), case, locales.load("zh"))


def test_broken_draft_is_retryable_for_the_case() -> None:
    """输入：一次拒绝。输出：无。

    预期行为：抛的是 `DraftError`，它带 `retryable_for_case`——管线看到这个标记才会重渲染，
    没这个标记的异常一律当生成器 bug，重渲染只会把 bug 藏起来。
    """
    case = ReceiptCase("C1", "under", "none")
    with pytest.raises(DraftError) as info:
        render.check_draft(render.ContentDraft(text=""), case, locales.load("zh"))
    assert type(info.value).retryable_for_case is True


def test_prompt_slots_are_all_filled() -> None:
    """输入：一个 case、一种语言。输出：无。

    预期行为：提示词里没有残留的占位符（花括号一个都不剩），关键词与金额都原样出现，而且
    **明说了语言约束**——提示词通篇是目标语言并不足以让模型不换语言（范式 §9.32）。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for case in CASES:
            try:
                loc = locales.load(lang)
                prompt = render.build_prompt(loc, case)
                assert "{" not in prompt and "}" not in prompt
                assert case.amount in prompt
                assert features.keyword_table(loc)[case.category][0] in prompt
                assert str(loc["language_name"]) in prompt
                if case.decoy == features.DECOY_NEAR:
                    assert render.decoy_word_for(case, loc) in prompt
                else:
                    assert all(
                        word not in prompt
                        for cid in features.CATEGORY_IDS
                        for word in features.decoy_table(loc)[cid]
                    )
            except AssertionError as exc:
                bad.append(f"{(lang, case)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("lang", LANGS)
def test_the_decoy_reminder_is_repeated_at_the_end(lang: str) -> None:
    """输入：一个要埋诱饵的 case、一种语言。输出：无。

    预期行为：诱饵词在提示词里出现**两次**（中段那句要求 + 末尾那句复查），而且倒数第二行就是
    复查那句。实测 en 侧漏抄诱饵是这一族最常见的渲染期拒稿原因，靠末尾重述压下去；不埋诱饵的
    case 上一次都不许出现。
    """
    loc = locales.load(lang)
    near = ReceiptCase("C2", "mild", features.DECOY_NEAR)
    word = render.decoy_word_for(near, loc)
    prompt = render.build_prompt(loc, near)
    assert prompt.count(word) == 2
    assert word in prompt.splitlines()[-2]
    assert word not in render.build_prompt(loc, ReceiptCase("C2", "mild", "none"))


@pytest.mark.parametrize("lang", LANGS)
def test_prompt_forbids_other_categories(lang: str) -> None:
    """输入：一个 case、一种语言。输出：无。

    预期行为：提示词把**别的类别**的关键词逐条列为「不许出现」，并点名所有类别名不许出现——
    只写「必须出现什么」时模型会顺手带上别的形状，命中类别当场多出来。
    """
    case = ReceiptCase("C2", "mild", "none")
    loc = locales.load(lang)
    prompt = render.build_prompt(loc, case)
    for cid in features.CATEGORY_IDS:
        if cid == case.category:
            continue
        for word in features.keyword_table(loc)[cid]:
            assert word in prompt
        assert render.category_name(loc, cid) in prompt


def test_build_task_carries_the_plan() -> None:
    """输入：一个规格。输出：无。

    预期行为：任务 id 由规格算得出来（断点续跑靠它）、`spec` 里带上注入特征——后端与假后端
    想「照着计划答」不必解析提示词。
    """
    case = ReceiptCase("C3", "heavy", "near")
    task = render.build_task(spec_for(case, seq=7), locales.load("zh"))
    assert task.task_id == "receiptcat-train-0000007-render"
    assert task.domain == "receiptcat"
    assert task.spec["category"] == "C3"
    assert task.spec["amount"] == case.amount
    assert task.spec["decoy"] == "near"


def test_to_state_is_stable() -> None:
    """输入：同一份草稿渲染两次。输出：无。预期行为：state 逐字节相同（可重放）。"""
    case = ReceiptCase("C1", "under", "none")
    first = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    second = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    assert first == second
    assert first.endswith(text_of(case))


def test_sample_of_helper_matches_the_renderer() -> None:
    """输入：一个忠实样本。输出：无。预期行为：夹具造的 state 与直接渲染的一致。"""
    case = ReceiptCase("C2", "mild", "none")
    _spec, sample = faithful(case)
    assert sample == sample_of(text_of(case))
