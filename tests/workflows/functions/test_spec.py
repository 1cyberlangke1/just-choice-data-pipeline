"""真值规格：工具表、语气表、case 矩阵与绑定计划的构造不变量。"""

from __future__ import annotations

import random
from itertools import product

import pytest

from workflows.functions import spec as sp
from workflows.functions.workflow import OOD_CONFIG, FunctionsWorkflow


def test_every_tool_has_a_non_empty_arg_list() -> None:
    """输入：工具表。输出：无。预期行为：每个工具都有参数，且名字与键一致。"""
    assert set(sp.TOOLS) == set(sp.TOOL_NAMES)
    for tool, args in sp.TOOLS.items():
        assert args, tool
        assert len({arg.name for arg in args}) == len(args), f"{tool} 有重名参数"


def test_same_arg_name_has_one_definition() -> None:
    """输入：工具表。输出：无。

    预期行为：同名参数在不同工具里必须是同一份定义——`ARG_BY_NAME` 是「参数名 → 定义」的
    唯一出口，两处定义不一致会让读法认出来的取值表和出题用的取值表对不上。
    """
    for args in sp.TOOLS.values():
        for arg in args:
            assert sp.ARG_BY_NAME[arg.name] == arg


def test_required_args_are_stated_in_every_voice() -> None:
    """输入：语气表。输出：无。

    预期行为：必填参数在**每种**语气下都被提到——漏一种语气就会渲染出一句缺必填参数的
    请求，而读法读回来的绑定跟金计划对不上。
    """
    for tool, args in sp.TOOLS.items():
        for arg in args:
            if not arg.required:
                continue
            for voice in sp.VOICES:
                assert arg.name in sp.VOICE_ARGS[tool][voice], (tool, voice, arg.name)


def test_optional_args_are_stated_somewhere_and_silent_somewhere() -> None:
    """输入：语气表。输出：无。

    预期行为：每个可选参数至少有一种语气提到、至少有一种语气不提——**两边都要有**。
    全都提的话 `stated_<参数>` 恒为 True，那一族就成了「永远答 True」的坏数据（范式 §5.7）；
    全都不提则那一族恒为 False，同样退化成常量。
    """
    for tool, args in sp.TOOLS.items():
        for arg in args:
            if arg.required:
                continue
            seen = {arg.name in sp.VOICE_ARGS[tool][voice] for voice in sp.VOICES}
            assert seen == {True, False}, (tool, arg.name, sp.VOICE_ARGS[tool])


def test_voice_args_only_name_real_args() -> None:
    """输入：语气表。输出：无。预期行为：每个工具的语气表键就是全部语气，取值都是它的参数。"""
    assert set(sp.VOICE_ARGS) == set(sp.TOOLS)
    for tool, table in sp.VOICE_ARGS.items():
        assert set(table) == set(sp.VOICES), tool
        names = {arg.name for arg in sp.TOOLS[tool]}
        for voice, stated in table.items():
            assert set(stated) <= names, (tool, voice)


def test_voices_have_distinct_stated_sets_per_tool() -> None:
    """输入：每个工具的语气表。输出：无。

    预期行为：同一工具的各档语气**提到不同的参数集合**。两档语气如果提到的是同一批参数，
    它们渲染出来的句子就只差随机取值——同一个 case 维度上多出一份近似重复的样本，
    而且两个「不同」的 case 很容易渲染出同一句话，切分之间就会撞车。
    """
    for tool, table in sp.VOICE_ARGS.items():
        sets = {frozenset(stated) for stated in table.values()}
        assert len(sets) == len(sp.VOICES), (tool, table)


def test_aliased_tools_are_the_ones_with_a_two_spelling_table() -> None:
    """输入：工具表。输出：无。

    预期行为：`has_alias` 只对真有别名取值表的工具为真；`build_cases` 靠它跳过
    「第二种写法渲染出来一模一样」的组合。
    """
    assert {tool for tool in sp.TOOLS if sp.has_alias(tool)} == {
        "plot_price",
        "compare_returns",
        "rolling_correlation",
    }


def test_case_matrix_skips_alias_values_that_change_nothing() -> None:
    """输入：全开的采样参数。输出：无。

    预期行为：没有别名写法的工具只出现一个 alias 取值——多出来的那一支渲染结果一模一样。
    """
    cases = sp.build_cases(sp.CallConfig())
    aliases_of: dict[str, set[str]] = {}
    for case in cases:
        aliases_of.setdefault(case.tool, set()).add(case.alias)
    for tool, aliases in aliases_of.items():
        expected = set(sp.ALIASES) if sp.has_alias(tool) else {sp.ALIASES[0]}
        assert aliases == expected, (tool, aliases)


def test_subject_args_are_stated_arguments() -> None:
    """输入：主体参数表。输出：无。

    预期行为：主体参数必须在每种语气下都被提到（它的内容写在 `lead` 的 `{subject}` 槽里，
    没提到就没法填那个槽）。
    """
    assert set(sp.SUBJECT_ARGS) == set(sp.TOOLS)
    for tool, names in sp.SUBJECT_ARGS.items():
        for voice in sp.VOICES:
            for name in names:
                assert name in sp.VOICE_ARGS[tool][voice], (tool, voice, name)


def test_symbols_share_the_symbol_phrase_table() -> None:
    """输入：参数表。输出：无。

    预期行为：`symbols`（集合）与 `symbol`（单个）共用一张股票短语表——分成两张表就会
    漏改一张，而且反解要同时认两套短语，纯属自找歧义。
    """
    assert sp.phrase_key_of("symbols") == "symbol"
    assert sp.phrase_key_of("symbol") == "symbol"
    assert sp.phrase_key_of("benchmark") == "benchmark"
    assert sp.ARG_BY_NAME["symbols"].phrase == "symbol"


def test_phrase_key_of_rejects_unknown_arg() -> None:
    """输入：不存在的参数名。输出：无。预期行为：KeyError。"""
    with pytest.raises(KeyError):
        sp.phrase_key_of("nope")


def test_voice_members_fit_every_set_arg() -> None:
    """输入：语气成员数表。输出：无。预期行为：每种语气要的成员数都不超过取值表长度。"""
    for voice, count in sp.VOICE_MEMBERS.items():
        assert voice in sp.VOICES
        assert 1 <= count <= len(sp.STOCKS)


def test_build_plan_is_deterministic(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：同一个 case 造两次计划。输出：无。预期行为：逐字段一致（指纹派生，不掷骰子）。"""
    case = make_case()
    assert sp.build_plan(case) == sp.build_plan(case)
    # 计划不跟着切分 / 序号变——同一素材在任何切分、任何进程里都是同一份计划。
    assert sp.sample_spec(case, "train", 0).plan == sp.sample_spec(case, "val", 7).plan


def test_build_plan_takes_the_window_from_the_case(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：两个只差窗口的 case。输出：无。

    预期行为：窗口取值来自 case 本身（它是留出维度），不跟着指纹掷——否则留出就不成立。
    """
    plan = sp.build_plan(make_case(window="1w"))
    assert plan.binding("window").value == "1w"
    assert sp.build_plan(make_case(window="1d")).binding("window").value == "1d"


def test_set_members_follow_the_value_table_order(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：一个 `set` 参数的绑定。输出：无。

    预期行为：成员按取值表本身的顺序排（不是按字母排）——读法也是按取值表顺序返回的，
    两边顺序一致，「反解出来的计划 == 金计划」这条不变量才成立。
    """
    for voice in sp.VOICES:
        binding = sp.build_plan(make_case(tool="compare_returns", voice=voice)).binding("symbols")
        assert binding.members == tuple(
            value for value in sp.STOCKS if value in binding.members
        )
        assert len(binding.members) == min(sp.VOICE_MEMBERS[voice], len(sp.STOCKS))


def test_flag_binding_is_on_when_stated(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：提到 / 不提 flag 的两种语气。输出：无。

    预期行为：提到时 `value` 恒为 `"on"`（布尔开关没有别的取值），不提时 `stated=False`。
    """
    chatty = sp.build_plan(make_case(tool="plot_price", voice="chatty")).binding("include_volume")
    assert (chatty.stated, chatty.value) == (True, "on")
    terse = sp.build_plan(make_case(tool="plot_price", voice="terse")).binding("include_volume")
    assert (terse.stated, terse.value) == (False, "")


def test_unstated_optional_args_carry_no_value(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：最简语气的计划。输出：无。

    预期行为：没被提到的可选参数 `stated=False` 且不带取值 / 成员——这正是 `stated_*`
    题的 False 形态。
    """
    plan = sp.build_plan(make_case(tool="plot_price", voice="terse"))
    for binding in plan.args:
        if sp.ARG_BY_NAME[binding.name].required:
            continue
        assert binding.stated is False
        assert binding.value == "" and binding.members == ()


def test_plan_binding_rejects_unknown_arg(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：计划里没有的参数名。输出：无。预期行为：KeyError。"""
    with pytest.raises(KeyError):
        sp.build_plan(make_case()).binding("nope")


def test_material_of_separates_every_case(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：全部 case。输出：无。预期行为：素材指纹两两不同（它是分桶与内容种子的输入）。"""
    cases = sp.build_cases(sp.CallConfig())
    materials = [sp.material_of(case) for case in cases]
    assert len(set(materials)) == len(materials)


def test_unseen_dims_reports_held_out_values(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：全留出 / 全见过两个 case。输出：无。预期行为：按固定顺序点名留出的维度。"""
    # `top_movers` 没有别名写法，它的 `alias` 不算留出维度。
    assert make_case(tool="top_movers", window="3mo", alias="name", phrasing=1).unseen_dims() == (
        "tool",
        "window",
        "phrasing",
    )
    assert make_case(tool="plot_price", window="3mo", alias="name", phrasing=1).unseen_dims() == (
        "window",
        "alias",
        "phrasing",
    )
    assert make_case(tool="plot_price", window="1d", alias="ticker", phrasing=0).unseen_dims() == ()


def test_seen_tables_are_real_values() -> None:
    """输入：留出清单。输出：无。预期行为：训练侧「见过」的取值都在取值表里。"""
    assert sp.SEEN_TOOLS < set(sp.TOOL_NAMES)
    assert sp.SEEN_WINDOWS < set(sp.WINDOWS)
    assert sp.SEEN_ALIASES < set(sp.ALIASES)
    assert sp.SEEN_PHRASINGS < set(sp.PHRASINGS)


@pytest.mark.parametrize("field", ["tools", "voices", "windows", "aliases", "phrasings"])
def test_config_rejects_empty_pools(field: str) -> None:
    """输入：把某个取值表清空。输出：无。预期行为：构造期 ValueError，且报错点名字段。"""
    with pytest.raises(ValueError, match=field):
        sp.CallConfig(**{field: ()})


def test_default_config_still_constructs() -> None:
    """输入：全默认参数。输出：无。预期行为：默认值都合法，校验不会误伤正常配置。"""
    sp.CallConfig()


def test_build_cases_covers_the_whole_matrix() -> None:
    """输入：默认采样参数。输出：无。预期行为：case 数 = 五张取值表长度的乘积。"""
    config = sp.CallConfig()
    # 没有别名写法的工具只展开一个 alias 取值，所以要按工具逐个算。
    expected = (
        sum(len(config.aliases) if sp.has_alias(tool) else 1 for tool in config.tools)
        * len(config.voices)
        * len(config.windows)
        * len(config.phrasings)
    )
    assert len(sp.build_cases(config)) == expected


def test_build_cases_ood_only_keeps_unseen_combinations() -> None:
    """输入：`ood_only` 打开与关闭的同一张取值表。输出：无。

    预期行为：只留「至少含一个训练侧没见过的取值」的组合，且留下的每一条都真的含留出取值。
    """
    every = sp.build_cases(sp.CallConfig())
    ood = sp.build_cases(sp.CallConfig(ood_only=True))
    assert 0 < len(ood) < len(every)
    assert all(case.unseen_dims() for case in ood)
    assert not any(case.unseen_dims() for case in every if case not in ood)


def test_case_describe_and_as_dict(make_case) -> None:  # type: ignore[no-untyped-def]
    """输入：一个 case。输出：无。预期行为：五个维度都摊平，描述里五个取值都在。"""
    case = make_case(tool="top_movers", voice="chatty", window="3mo", alias="name", phrasing=1)
    assert case.as_dict() == {
        "tool": "top_movers",
        "voice": "chatty",
        "window": "3mo",
        "alias": "name",
        "phrasing": 1,
    }
    for value in case.as_dict().values():
        assert str(value) in case.describe()


def test_ood_config_keeps_only_unseen_cases() -> None:
    """输入：workflow 自带的 `test_ood` 配置。输出：无。预期行为：每条都含留出取值。"""
    cases = FunctionsWorkflow().build_cases(OOD_CONFIG)
    assert cases and all(case.unseen_dims() for case in cases)
