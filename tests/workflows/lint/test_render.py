"""行级规则检查的渲染测试：提示词、草稿校验、state 往返与反解。"""

from __future__ import annotations

import pytest

from workflows.lint import features, locales, render
from workflows.lint.spec import LintSpec
from jc.workflow import DraftError

from tests.workflows.lint.conftest import CLEAN_LINES, HIT_LINES, draft_lines

#: 一个「恰好一行命中」的 case（`spot` 族会出）与一个「五行命中」的 case。
SINGLE = features.LintCase(features.RULE_TODO, "code", 1)
MANY = features.LintCase(features.RULE_DEBUG, "log", 5)


def spec_for(case: features.LintCase, phrasing: int = 0) -> LintSpec:
    """输入：case、问法编号。输出：真值规格。预期行为：行计划由 case 指纹算出来。"""
    return LintSpec(
        case=case,
        phrasing=phrasing,
        split="train",
        seq=1,
        plan=features.plan_for(case, phrasing),
    )


def test_gate_rows_match_the_code_bands() -> None:
    """输入：两个语言。输出：无。预期行为：表的三行与 `features.gate_of` 的分段一一对应。"""
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        rows = render.gate_rows(loc)
        assert len(rows) == 3
        for row, (low, _high) in zip(rows, render.GATE_BOUNDS, strict=True):
            assert str(loc["gate"][features.gate_of(low)]) in row


def test_body_header_names_the_genre_and_line_count() -> None:
    """输入：三条体裁。输出：无。预期行为：抬头里带上行数，且三条互不相同。"""
    loc = locales.load("zh")
    headers = {genre: render.body_header(loc, genre, features.SIZE) for genre in features.GENRES}
    assert len(set(headers.values())) == len(features.GENRES)
    assert all(str(features.SIZE) in header for header in headers.values())
    assert render.genre_of_lead(loc, headers["doc"][len(loc["state"]["body_mark"]) :]) == "doc"
    assert render.genre_of_lead(loc, "看不出来") is None


def test_rule_names_resolve_both_ways() -> None:
    """输入：两个语言。输出：无。预期行为：规则名 → 规则名键、再回得来。"""
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        for rule in features.RULES:
            assert render.rule_of_name(loc, render.rule_name(loc, rule)) == rule
        assert render.rule_of_name(loc, "没有这条规则") is None


@pytest.mark.parametrize("hits", [0, 1, 5, 8])
def test_prompt_states_the_required_rows(hits: int) -> None:
    """输入：每个命中行数。输出：无。

    预期行为：提示词里写明哪几行必须命中（1 起、顿号分隔）；命中 0 / 8 时换用另外两句，
    而且整段提示词填完槽之后一个花括号都不剩。
    """
    plan = features.plan_for(features.LintCase(features.RULE_PATH, "doc", hits), 0)
    prompt = render.build_prompt(plan, locales.load("zh"))
    assert "{" not in prompt and "}" not in prompt
    if hits == 0:
        assert "一行都不许命中" in prompt
    elif hits == features.SIZE:
        assert "全部命中" in prompt
    else:
        assert "、".join(str(row + 1) for row in plan.hit_rows) in prompt


def test_prompt_lists_every_rule_criteria() -> None:
    """输入：一个计划。输出：无。预期行为：五条规则的判据都在提示词里（模型才知道要躲谁）。"""
    loc = locales.load("zh")
    prompt = render.build_prompt(features.plan_for(SINGLE, 0), loc)
    for rule in features.RULES:
        assert render.rule_criteria(loc, rule) in prompt


def test_check_draft_accepts_a_faithful_draft() -> None:
    """输入：完全兑现计划的草稿。输出：无。预期行为：不抛异常。"""
    plan = features.plan_for(MANY, 0)
    render.check_draft(draft_lines(MANY.rule, list(plan.hit_rows)), plan)


@pytest.mark.parametrize(
    ("switch", "match"),
    [
        ("unrealized", "该命中"),
        ("extra_hit", "该是干净行"),
        ("drop_line", "正好"),
        ("fence_line", "围栏"),
    ],
)
def test_check_draft_rejects_broken_drafts(switch: str, match: str) -> None:
    """输入：四种写歪的草稿。输出：无。预期行为：各自抛出带人话的 ValueError。"""
    plan = features.plan_for(MANY, 0)
    lines = draft_lines(MANY.rule, list(plan.hit_rows), **{switch: True})
    with pytest.raises(ValueError, match=match):
        render.check_draft(lines, plan)


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        (["", *CLEAN_LINES[:7]], "空"),
        (["a\nb", *CLEAN_LINES[:7]], "换行"),
        (["3. import os", *CLEAN_LINES[:7]], "行号前缀"),
    ],
)
def test_check_draft_rejects_bad_line_shapes(lines: list[str], match: str) -> None:
    """输入：空行 / 行内换行 / 自带行号。输出：无。预期行为：各自抛 ValueError。"""
    plan = features.plan_for(features.LintCase(features.RULE_SECRET, "code", 0), 0)
    with pytest.raises(ValueError, match=match):
        render.check_draft(lines, plan)


def test_broken_drafts_are_marked_retryable() -> None:
    """输入：一条漏了一行命中的草稿。输出：无。

    预期行为：抛的是 `jc.workflow.DraftError`（带 `retryable_for_case`），管线据此重渲染一次——
    真机上这个 workflow 的接受率只有 6/10，拒稿全是「渲染期写歪」（PLAN 10.79），单发等于白丢。
    """
    plan = features.plan_for(MANY, 0)
    lines = draft_lines(MANY.rule, list(plan.hit_rows), unrealized=True)
    with pytest.raises(DraftError) as caught:
        render.check_draft(lines, plan)
    assert getattr(caught.value, "retryable_for_case", False) is True


def test_state_roundtrip_in_both_locales() -> None:
    """输入：两个语言、两条 case。输出：无。

    预期行为：`to_state` 写出来的 state，`parse_state` 能原样读回片段行、规则、体裁与
    处置办法表——而且**不需要知道是哪种语言**。
    """
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        for case in (SINGLE, MANY):
            plan = features.plan_for(case, 1)
            lines = draft_lines(case.rule, list(plan.hit_rows))
            state = render.to_state(lines, plan, loc)
            parsed = render.parse_state(state)
            assert parsed.lines == tuple(lines)
            assert parsed.rule == case.rule
            assert parsed.genre == case.genre
            assert parsed.gate == render.gate_rows(loc)
            assert render.detect_locale(state) == lang


def test_parse_state_ignores_non_text_and_unknown_layout() -> None:
    """输入：非文本、以及一段没有片段抬头的文本。输出：无。预期行为：都返回空结果。"""
    empty = render.parse_state({"not": "text"})
    assert empty.lines == () and empty.genre is None
    assert render.detect_locale(123) is None
    other = render.parse_state("1. import os\n2. import sys\n")
    assert other.lines == () and other.genre is None


def test_hit_lines_really_hit_and_clean_lines_really_dont() -> None:
    """输入：夹具的两组样板行。输出：无。

    预期行为：`HIT_LINES` 每条都命中它对应的规则，`CLEAN_LINES` 每条一条规则都不命中——
    夹具写歪了，整个测试套件都会变成「用错的口径验错的口径」。
    """
    for rule, line in HIT_LINES.items():
        assert features.matches(rule, line), (rule, line)
    for line in CLEAN_LINES:
        assert features.match_any(line) is None, line
