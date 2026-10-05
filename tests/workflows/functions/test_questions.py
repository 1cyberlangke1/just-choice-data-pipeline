"""题族：题号、选项、金答案与「哪一族什么时候出」。"""

from __future__ import annotations

import random

import pytest

from workflows.functions import locales, questions, render, spec as sp

from tests.workflows.functions.conftest import MakeCase

#: 每个工具**可能**出的全部题族（审计与判别器都按它点名）。
EXPECTED_FAMILIES: dict[str, set[str]] = {
    "plot_price": {
        "tool",
        "arg_symbol",
        "arg_window",
        "stated_style",
        "arg_style",
        "stated_resolution",
        "arg_resolution",
        "stated_include_volume",
    },
    "compare_returns": {
        "tool",
        "member_NVDA",
        "member_AMD",
        "member_AAPL",
        "member_MSFT",
        "member_TSLA",
        "arg_window",
        "stated_normalize",
        "stated_resolution",
        "arg_resolution",
    },
    "rolling_correlation": {
        "tool",
        "arg_symbol",
        "arg_benchmark",
        "arg_window",
        "stated_resolution",
        "arg_resolution",
        "stated_smoothing",
    },
    "top_movers": {"tool", "arg_window", "stated_direction", "arg_direction", "stated_include_etfs"},
}


def _build(case: sp.CallCase) -> list[object]:
    """输入：case。输出：中文题目列表。预期行为：与管线里 `ask` 走同一条路。"""
    spec = sp.sample_spec(case, "train", 0)
    return questions.build(spec.plan, locales.load("zh"), spec.plan.phrasing, random.Random(0))


@pytest.mark.parametrize("tool", sorted(EXPECTED_FAMILIES))
def test_families_of_names_every_reachable_family(tool: str) -> None:
    """输入：每个工具。输出：无。预期行为：题族集合与手写清单一致（多一个少一个都红）。"""
    assert set(questions.families_of(tool)) == EXPECTED_FAMILIES[tool]
    assert questions.families_of(tool) == tuple(sorted(EXPECTED_FAMILIES[tool]))


def test_families_of_rejects_unknown_tool() -> None:
    """输入：不存在的工具。输出：无。预期行为：KeyError。"""
    with pytest.raises(KeyError):
        questions.families_of("nope")


def test_families_is_the_union_of_every_tool() -> None:
    """输入：题族总表。输出：无。预期行为：等于各工具题族的并集（判别器拿它认未知题族）。"""
    union = {family for tool in sp.TOOLS for family in questions.families_of(tool)}
    assert set(questions.FAMILIES) == union
    assert questions.FAMILIES == tuple(sorted(union))


def test_choice_and_noul_family_names_are_consistent() -> None:
    """输入：题族前缀表。输出：无。预期行为：`stated_` / `member_` 只出 noul，`tool` 只出 choice。"""
    assert "tool" in questions.CHOICE_FAMILIES
    for family in questions.FAMILIES:
        if family.startswith(questions.NOUL_FAMILIES):
            assert family != "tool"


def test_build_emits_exactly_the_families_of_the_plan(make_case: MakeCase) -> None:
    """输入：四个工具各一个 case。输出：无。预期行为：出题族 ⊆ 该工具的可达族，且题号唯一。"""
    for tool in sp.TOOLS:
        reached: set[str] = set()
        for voice in sp.VOICES:
            items = _build(make_case(tool=tool, voice=voice))
            families = [item.family for item in items]  # type: ignore[attr-defined]
            assert len(set(families)) == len(families), (tool, voice)
            assert set(families) <= EXPECTED_FAMILIES[tool], (tool, voice)
            reached |= set(families)
        assert reached == EXPECTED_FAMILIES[tool], (tool, sorted(EXPECTED_FAMILIES[tool] - reached))


def test_families_for_mirrors_what_build_emits(make_case: MakeCase) -> None:
    """输入：每个 case 的金计划与它实际出的题族。输出：无。

    预期行为：`families_for(工具, 文本里提到过的参数)` 与 `build` 真正出的题族**完全一致**——
    审计就是拿它算「应有题族」的，两边错开就会出现误报或漏报。
    """
    for tool in sp.TOOLS:
        for voice in sp.VOICES:
            case = make_case(tool=tool, voice=voice)
            plan = sp.build_plan(case)
            stated = frozenset(binding.name for binding in plan.args if binding.stated)
            emitted = {item.family for item in _build(case)}  # type: ignore[attr-defined]
            assert set(questions.families_for(tool, stated)) == emitted, (tool, voice)


def test_families_for_ignores_unstated_optional_args(make_case: MakeCase) -> None:
    """输入：只提到必填参数的请求。输出：无。

    预期行为：没被提到的可选参数只出 `stated_*`，不出 `arg_*`——审计按「实际提到过什么」点名。
    """
    families = questions.families_for("plot_price", frozenset({"symbol", "window"}))
    assert "stated_style" in families
    assert "arg_style" not in families
    assert "arg_symbol" in families and "arg_window" in families
    assert "stated_symbol" not in families and "stated_window" not in families


def test_tool_question_offers_every_function(make_case: MakeCase) -> None:
    """输入：任意一个 case。输出：无。

    预期行为：`tool` 题永远是四选一（四个工具都在选项里），金答案是金工具，选项描述
    来自 `[tool_desc]`——描述写清了每个函数干什么，考生只读题面就能答。
    """
    case = make_case(tool="rolling_correlation", voice="terse")
    item = _build(case)[0]
    assert item.family == "tool"  # type: ignore[attr-defined]
    question = item.question  # type: ignore[attr-defined]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(sp.TOOLS)
    assert item.answer == "rolling_correlation"  # type: ignore[attr-defined]
    assert question["instructions"]


def test_arg_question_options_are_the_value_table(make_case: MakeCase) -> None:
    """输入：一个被提到的 `choice` 参数。输出：无。

    预期行为：选项就是该参数的**全量**取值表（不只本切分见过的那几个），金答案是请求里写的那个。
    """
    case = make_case(tool="plot_price", voice="chatty")
    plan = sp.build_plan(case)
    item = next(item for item in _build(case) if item.family == "arg_window")  # type: ignore[attr-defined]
    assert set(item.question["criteria"]) == set(sp.WINDOWS)  # type: ignore[attr-defined]
    assert item.answer == plan.binding("window").value  # type: ignore[attr-defined]
    assert str(locales.load("zh")["arg_desc"]["window"]) in item.question["instructions"]  # type: ignore[attr-defined]


def test_stated_question_carries_the_default_rule(make_case: MakeCase) -> None:
    """输入：一个可选参数。输出：无。

    预期行为：noul 题的判据写清「没提 = 用默认值」——这条是人为约定，必须写在题面里
    （范式 §9.8），否则考生没法从题面推出答案。
    """
    case = make_case(tool="plot_price", voice="terse")
    item = next(item for item in _build(case) if item.family == "stated_style")  # type: ignore[attr-defined]
    assert item.question["type"] == "noul"  # type: ignore[attr-defined]
    assert set(item.question["criteria"]) == {"true", "false"}  # type: ignore[attr-defined]
    assert "默认" in item.question["criteria"]["false"]  # type: ignore[attr-defined]
    assert item.answer is False  # type: ignore[attr-defined]


def test_required_args_never_get_a_stated_question(make_case: MakeCase) -> None:
    """输入：含必填参数的工具。输出：无。

    预期行为：必填参数恒为 True，是常量答案——教「永远答 True」的题是坏数据（范式 §5.7）。
    """
    for tool, args in sp.TOOLS.items():
        names = {arg.name for arg in args if arg.required}
        for voice in sp.VOICES:
            families = {item.family for item in _build(make_case(tool=tool, voice=voice))}  # type: ignore[attr-defined]
            for name in names:
                assert f"stated_{name}" not in families, (tool, voice, name)


def test_set_argument_gets_one_member_question_per_value(make_case: MakeCase) -> None:
    """输入：`set` 参数。输出：无。

    预期行为：取值表里**每个**成员都有一道 noul 题，答案就是「请求点名要它了吗」。
    """
    case = make_case(tool="compare_returns", voice="chatty")
    members = sp.build_plan(case).binding("symbols").members
    items = [item for item in _build(case) if item.family.startswith("member_")]  # type: ignore[attr-defined]
    assert {item.family for item in items} == {f"member_{value}" for value in sp.STOCKS}  # type: ignore[attr-defined]
    for item in items:
        value = item.family.removeprefix("member_")  # type: ignore[attr-defined]
        assert item.answer is (value in members)  # type: ignore[attr-defined]
        assert item.question["type"] == "noul"  # type: ignore[attr-defined]


def test_flag_argument_only_gets_a_stated_question(make_case: MakeCase) -> None:
    """输入：flag 参数被提到 / 没被提到。输出：无。

    预期行为：flag 没有取值，所以只有 `stated_*` 一族；提到时答案为 True。
    """
    chatty = _build(make_case(tool="plot_price", voice="chatty"))
    item = next(item for item in chatty if item.family == "stated_include_volume")  # type: ignore[attr-defined]
    assert item.answer is True  # type: ignore[attr-defined]
    assert not any(item.family.startswith("arg_include_volume") for item in chatty)  # type: ignore[attr-defined]


def test_phrasing_only_changes_the_wording(make_case: MakeCase) -> None:
    """输入：两套句面。输出：无。

    预期行为：句面只改题面文字——题族、题型与 `tool` 那一题的答案都不变。
    （其余题的答案可以不同：句面是 case 的一维，换一维就换了一份素材指纹，
    计划里的随机取值也跟着换。所以「答案完全相同」只对同 case 成立。）
    """
    zero = _build(make_case(phrasing=0))
    one = _build(make_case(phrasing=1))
    assert [item.family for item in zero] == [item.family for item in one]  # type: ignore[attr-defined]
    assert [item.question["type"] for item in zero] == [  # type: ignore[attr-defined]
        item.question["type"] for item in one  # type: ignore[attr-defined]
    ]
    assert zero[0].answer == one[0].answer == "plot_price"  # type: ignore[attr-defined]
    assert zero[0].question["instructions"] != one[0].question["instructions"]  # type: ignore[attr-defined]


def test_noul_never_shuffles() -> None:
    """输入：一道 noul 题。输出：无。

    预期行为：`shuffle=False`——true / false 的顺序是语义顺序，打乱会让题面自相矛盾。
    """
    item = questions._noul("f", "题", {"true": "是", "false": "否"}, True)
    assert item.shuffle is False
    assert item.answer is True


def test_values_of_and_is_required_reject_unknown_args() -> None:
    """输入：不存在的参数名。输出：无。预期行为：两个辅助函数都报 KeyError。"""
    with pytest.raises(KeyError):
        questions._values_of(sp.Binding("nope", "choice", True, "x"))
    with pytest.raises(KeyError):
        questions._is_required("nope")


def test_fill_leaves_unknown_placeholders_alone() -> None:
    """输入：带未知槽位的题面模板。输出：无。

    预期行为：只替换认识的名字（题面模板里的 `{arg}` / `{value}` 由调用方给全）。
    """
    assert questions._fill("给{arg}取值", arg="时间范围") == "给时间范围取值"
    assert questions._fill("{unknown}", arg="x") == "{unknown}"


def test_option_descriptions_come_from_the_locale(make_case: MakeCase) -> None:
    """输入：一个成员题。输出：无。预期行为：题面里的取值描述来自 `[opt.symbol]`。"""
    item = next(
        item for item in _build(make_case(tool="compare_returns", voice="terse"))  # type: ignore[attr-defined]
        if item.family == "member_NVDA"
    )
    assert str(locales.load("zh")["opt"]["symbol"]["NVDA"]) in item.question["instructions"]  # type: ignore[attr-defined]


def test_instructions_never_leak_a_placeholder(make_case: MakeCase) -> None:
    """输入：全部工具 × 全部语气 × 两套句面。输出：无。

    预期行为：题面与判据里不剩 `{xxx}` 占位符——填漏一个就是考生看不懂的题。
    """
    for tool in sp.TOOLS:
        for voice in sp.VOICES:
            for phrasing in sp.PHRASINGS:
                for item in _build(make_case(tool=tool, voice=voice, phrasing=phrasing)):
                    texts = [item.question["instructions"]]  # type: ignore[attr-defined]
                    texts.extend(item.question.get("criteria", {}).values())  # type: ignore[attr-defined]
                    for text in texts:
                        assert "{" not in text and "}" not in text, (tool, voice, text)


def test_render_and_question_wording_come_from_the_same_locale(make_case: MakeCase) -> None:
    """输入：一个 case。输出：无。

    预期行为：题面里点名的参数描述（`[arg_desc]`）与请求文本同源——不同语言各用各的，
    题面与 state 不会串语言。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        spec = sp.sample_spec(make_case(), "train", 0)
        state = render.render_state(spec, loc)
        items = questions.build(spec.plan, loc, spec.plan.phrasing, random.Random(0))
        assert any(str(loc["arg_desc"]["window"]) in item.question["instructions"] for item in items)
        assert str(loc["intent"]["plot_price"]) in state
