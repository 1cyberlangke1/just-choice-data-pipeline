"""渲染与反解：往返必须逐格一致，认不出的行必须忽略而不是错认。"""

from __future__ import annotations

import pytest

from tests.workflows.routing.conftest import _make_spec
from workflows.routing import judges, locales, parse, render, spec as sp

LANGS = ("zh", "en")


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"area": "identity", "issue": "feature"},
        {"tier": "free", "wait": "short"},
        {"compliance": "security", "impact": "single_user", "urgency": "can_wait"},
        {"issue": "data_loss", "compliance": "gdpr", "wait": "medium"},
    ],
)
def test_roundtrip_is_exact(lang: str, overrides: dict) -> None:
    """输入：几组 case × 两种语言。输出：无。

    预期行为：渲染 → 反解 → 与真值逐格比对，零差异。
    """
    loc = locales.load(lang)
    spec = _make_spec(**overrides)
    state = render.build(spec, loc)
    ticket = parse.parse_ticket(state, parse.label_index(loc))
    assert judges.roundtrip_diff(spec, ticket, loc) == []


def test_render_is_pure_and_case_sensitive() -> None:
    """输入：同一份规格渲染两次、两份不同规格各渲染一次。输出：无。

    预期行为：同规格逐字节一致；不同规格（哪怕只差一维）文本必须不同——
    否则反解出来的事实与素材对不上。
    """
    loc = locales.load("zh")
    spec = _make_spec()
    assert render.build(spec, loc) == render.build(spec, loc)
    other = _make_spec(area="storage")
    assert render.build(spec, loc) != render.build(other, loc)


def test_title_line_is_not_mistaken_for_a_field() -> None:
    """输入：整屏工单。输出：无。

    预期行为：标题行以「工单」开头，但**不是**工单号那一格——反解要求标签后面紧跟分隔符，
    否则标题会被当成工单号，`ticket_id` 就变成「分诊台」了。
    """
    loc = locales.load("zh")
    spec = _make_spec()
    state = render.build(spec, loc)
    ticket = parse.parse_ticket(state, parse.label_index(loc))
    assert ticket.ticket_id == spec.ticket_id
    assert loc["meta"]["title"] not in ticket.ticket_id


def test_unknown_lines_are_ignored() -> None:
    """输入：在工单前面插几行无关文本。输出：无。预期行为：无关行不影响反解。"""
    loc = locales.load("zh")
    spec = _make_spec()
    state = "自动转接自邮件网关\n\n" + render.build(spec, loc) + "\n\n—— 完 ——\n"
    ticket = parse.parse_ticket(state, parse.label_index(loc))
    assert judges.roundtrip_diff(spec, ticket, loc) == []


def test_missing_field_speaks_human() -> None:
    """输入：被挖掉「合规标记」那一行的工单。输出：无。

    预期行为：报人话并点出缺的是哪一格——缺格是渲染漏写字段的硬伤，不能当成空值蒙过去。
    """
    loc = locales.load("zh")
    state = render.build(_make_spec(), loc)
    label = loc["meta"]["compliance_label"]
    stripped = "\n".join(line for line in state.splitlines() if not line.startswith(label))
    with pytest.raises(ValueError, match="compliance_label"):
        parse.parse_ticket(stripped, parse.label_index(loc))


def test_body_mentions_the_area_and_the_pressure() -> None:
    """输入：几组 case。输出：无。

    预期行为：正文里出现产品域名、影响面句与紧急度句——正文不是装饰，
    问题类型只出现在正文里（判别器要靠它反推队列）。
    """
    loc = locales.load("zh")
    for overrides in ({}, {"area": "analytics", "impact": "team", "urgency": "deadline"}):
        spec = _make_spec(**overrides)
        body = render.body_text(spec, loc)
        assert loc["area"][spec.case.area] in body
        assert loc["body"]["impact"][spec.case.impact] in body
        assert loc["body"]["urgency"][spec.case.urgency] in body


def test_reparse_works_without_a_locale_argument() -> None:
    """输入：两语渲染出来的工单。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，用合并标签表照样能反解——
    这是 `reparse` 这条协议能被管线直接调用的前提。
    """
    for lang in LANGS:
        spec = _make_spec(tier="enterprise", compliance="gdpr")
        state = render.build(spec, locales.load(lang))
        ticket = parse.parse_ticket(state, parse.merged_labels())
        assert ticket.ticket_id == spec.ticket_id
        assert ticket.compliance == locales.load(lang)["compliance"][spec.case.compliance]


def test_wait_is_not_written_anywhere_as_a_number() -> None:
    """输入：整屏工单。输出：无。

    预期行为：state 里**不出现**「已等待 N 分钟」这种直接给答案的格子——
    已等待必须由读者从两个时间相减算出来，否则 `sla_breach` 那一族就退化成读数了。
    """
    spec = _make_spec(wait="long")
    for lang in LANGS:
        state = render.build(spec, locales.load(lang))
        assert str(sp.elapsed_of("long")) not in state

