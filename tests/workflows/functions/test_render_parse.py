"""渲染与反解：一句话的生成与读回必须互为逆运算。

这是本 workflow 最要紧的一条不变量——判别器 `roundtrip` 就是靠它把答案从 state 独立
重算出来的。构造写歪与渲染写歪会变成同一类错误，不会两边各错一半还互相印证。
"""

from __future__ import annotations

import pytest

from workflows.functions import locales, parse, render, spec as sp

LANGS = tuple(locales.available())


def _plan_of(case: sp.CallCase, lang: str) -> tuple[sp.CallPlan, str]:
    """输入：case、语言。输出：(渲染出来的 state, 从 state 读回来的计划)。"""
    loc = locales.load(lang)
    state = render.render_state(sp.sample_spec(case, "train", 0), loc)
    return parse.read_request(state, loc), state


def test_render_parse_round_trips_the_whole_matrix() -> None:
    """输入：全部 case × 两种语言。输出：无。

    预期行为：从 state 读回来的计划与金计划**逐字段一致**（工具、每个参数的在场与取值 /
    成员）。这是 `roundtrip` 判别器的地基。
    """
    for case in sp.build_cases(sp.CallConfig()):
        for lang in LANGS:
            plan, state = _plan_of(case, lang)
            assert plan.tool == case.tool, (lang, case.describe(), state)
            got = {b.name: b for b in plan.args}
            want = {b.name: b for b in sp.build_plan(case).args}
            assert got == want, (lang, case.describe(), state)


def test_unstated_optional_args_leave_no_trace(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：最简语气的请求。输出：无。

    预期行为：没被提到的可选参数的取值短语与 flag 短语在文本里**一个都不出现**——
    出现一个，`stated_*` 题的答案就反了。
    """
    case = make_case(tool="plot_price", voice="terse")
    for lang in LANGS:
        loc = locales.load(lang)
        state = render.render_state(sp.sample_spec(case, "train", 0), loc)
        for name in ("style", "resolution"):
            for phrase in render.value_phrases(loc, sp.phrase_key_of(name), _any_value(name)):
                assert phrase not in state, (lang, name, phrase, state)
        for phrase in render.flag_phrases(loc, "include_volume"):
            assert phrase not in state, (lang, state)


def _any_value(name: str) -> str:
    """输入：参数名。输出：它取值表里的第一个取值（只为取「全部写法」用）。"""
    return sp.ARG_BY_NAME[name].values[0]


def test_alias_changes_the_spelling_not_the_answer(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：只差写法的两个 case。输出：无。

    预期行为：写代码与写名字渲染出来的文本不同，而读回来的取值各自等于自己那份金计划里的
    取值——写法只影响措辞，不影响答案。
    （两份 case 的素材指纹不同，计划里的随机取值本来就可以不一样；所以这里比的是
    「读回来 == 自己的金计划」，不是「两份计划相等」。）
    """
    ticker, name = make_case(alias="ticker"), make_case(alias="name")
    for lang in LANGS:
        plan_t, state_t = _plan_of(ticker, lang)
        plan_n, state_n = _plan_of(name, lang)
        assert state_t != state_n
        assert plan_t.binding("symbol").value == sp.build_plan(ticker).binding("symbol").value
        assert plan_n.binding("symbol").value == sp.build_plan(name).binding("symbol").value


def test_aliased_value_phrase_comes_from_the_right_subtable(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：两种写法。输出：无。预期行为：文本里出现的是对应别名子表里的那条短语。"""
    loc = locales.load("zh")
    value = sp.build_plan(make_case(alias="ticker")).binding("symbol").value
    assert render.value_phrase(loc, "symbol", value, "ticker") == value
    assert render.value_phrase(loc, "symbol", value, "name") == loc["val"]["symbol"]["name"][value]


def test_set_argument_is_listed_with_the_join_marks(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：语气越啰嗦的 compare_returns。输出：无。

    预期行为：成员越多文本越长，且每个成员都能读回来（列举用 `[mark]` 里的连接词）。
    """
    loc = locales.load("zh")
    lengths = []
    for voice in sp.VOICES:
        plan, state = _plan_of(make_case(tool="compare_returns", voice=voice), "zh")
        lengths.append(len(state))
        members = sp.build_plan(make_case(tool="compare_returns", voice=voice)).binding("symbols")
        for member in members.members:
            assert render.value_phrase(loc, "symbol", member, "ticker") in state
        assert plan.binding("symbols").members == members.members
    assert lengths == sorted(lengths)


def test_top_movers_has_no_subject_slot(make_spec) -> None:  # type: ignore[no-untyped-def]
    """输入：top_movers 的规格。输出：无。预期行为：`{subject}` 槽填成空串，不留占位符。"""
    loc = locales.load("zh")
    spec = make_spec(tool="top_movers", voice="terse")
    assert render.subject_of(spec.plan, loc) == ""
    assert "{" not in render.render_state(spec, loc)


def test_join_items_handles_zero_one_and_many() -> None:
    """输入：0 / 1 / 3 个短语。输出：无。预期行为：中文用顿号 + 「和」，英文用逗号 + and。"""
    zh = locales.load("zh")
    en = locales.load("en")
    assert render.join_items([], zh) == ""
    assert render.join_items(["A"], zh) == "A"
    assert render.join_items(["A", "B", "C"], zh) == "A、B和C"
    assert render.join_items(["A", "B", "C"], en) == "A, B and C"


def test_tool_of_requires_exactly_one_intent() -> None:
    """输入：说不清意图 / 意图短语撞车的文本。输出：无。

    预期行为：命中 0 个或 2 个以上都直接报错，不猜一个。
    """
    loc = locales.load("zh")
    with pytest.raises(ValueError, match="命中 0 个"):
        parse.tool_of("随便说点什么。", loc)
    both = f"{loc['intent']['plot_price']}还是{loc['intent']['top_movers']}"
    with pytest.raises(ValueError, match="命中 2 个"):
        parse.tool_of(both, loc)


def test_read_request_rejects_ambiguous_choice_value() -> None:
    """输入：同时出现两个取值的 `choice` 参数文本。输出：无。

    预期行为：直接报错——取值短语不互斥是构造 bug，静默取一个会把错误答案当金答案。
    """
    loc = locales.load("zh")
    state = f"{loc['intent']['plot_price']}，{loc['val']['window']['1d']}和{loc['val']['window']['1w']}。"
    with pytest.raises(ValueError, match="window"):
        parse.read_request(state, loc)


def test_rendering_refuses_a_multi_language_table() -> None:
    """输入：合并表。输出：无。

    预期行为：渲染只接受单语言短语表——合并表里一格是多种语言的写法，渲出来会中英混排。
    """
    with pytest.raises(ValueError, match="单语言"):
        render.value_phrase(locales.merged(), "window", "1d", "ticker")


def test_fill_reports_a_leftover_placeholder() -> None:
    """输入：带未知槽位的模板。输出：无。预期行为：报错并点名那个占位符。"""
    with pytest.raises(ValueError, match=r"\{nope\}"):
        render._fill("你好{nope}", name="x")


def test_missing_segment_template_is_an_error(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：被提到却没有分段模板的参数。输出：无。

    预期行为：直接报错——否则那句话会悄悄丢掉一个参数，而 `stated_*` 的答案就跟着错。
    """
    loc = locales.load("zh")
    table = dict(loc["req"]["plot_price"])
    table.pop("window")
    broken = dict(loc)
    broken["req"] = dict(loc["req"])
    broken["req"]["plot_price"] = table
    spec = sp.sample_spec(make_case(tool="plot_price", voice="chatty"), "train", 0)
    with pytest.raises(KeyError, match="window"):
        render.render_request(spec, broken)


def test_reparse_uses_the_merged_table(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：两种语言渲染出来的 state。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，合并表要能认出两种语言的文本。
    """
    for lang in LANGS:
        loc = locales.load(lang)
        state = render.render_state(sp.sample_spec(make_case(), "train", 0), loc)
        parsed = parse.read_request(state, locales.merged())
        assert parsed.tool == "plot_price"
        assert parsed.binding("symbol").value == sp.build_plan(make_case()).binding("symbol").value


def test_merged_table_is_language_agnostic(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：合并表。输出：无。预期行为：两种语言的意图短语都合并进来了。"""
    merged = locales.merged()
    assert set(merged["intent"]) == set(sp.TOOLS)
    for tool, phrases in merged["intent"].items():
        assert len(phrases) == len(LANGS), tool
    assert set(merged["flag"]) == {arg.name for arg in sp.ARG_BY_NAME.values() if not arg.values}
    assert set(merged["val"]["window"]) == set(sp.WINDOWS)
