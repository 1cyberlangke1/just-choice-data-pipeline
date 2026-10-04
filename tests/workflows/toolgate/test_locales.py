"""工具调用门控的语言文件测试：两语言键集一致、题面齐全、禁用语确实拦得住。"""

from __future__ import annotations

import typing

from jc.jsontypes import JsonValue
from workflows.toolgate import features, judges, locales, questions, render


def key_paths(node: JsonValue, prefix: str = "") -> set[str]:
    """输入：解析后的 TOML 节点。输出：所有叶子键的路径集合（用于两语言对比）。"""
    if isinstance(node, dict):
        out: set[str] = set()
        for key, value in node.items():
            out |= key_paths(value, f"{prefix}.{key}" if prefix else str(key))
        return out
    return {prefix}


def test_both_locales_have_identical_key_sets() -> None:
    """输入：zh 与 en。输出：无。预期行为：键路径集合完全相同——少一条文案就会在渲染期炸。"""
    zh = locales.load("zh")
    en = locales.load("en")
    assert key_paths(zh) == key_paths(en)


def test_every_prompt_dimension_value_has_text() -> None:
    """输入：提示词里那两维的全部取值。输出：无。

    预期行为：每个取值都有 brief 与 hint；缺一条 `feature_line` 会抛 KeyError，
    但那时已经在采样阶段了——这里提前钉住。
    """
    loc = locales.load("zh")
    for dim in render.DIM_ORDER:
        for value in features.DIMS[dim]:
            spec = loc["feature"][dim][value]
            assert str(spec["brief"]).strip()
            assert str(spec["hint"]).strip()
            assert "{" not in str(spec["brief"])


def test_prompt_only_asks_for_the_text_dimensions() -> None:
    """输入：无。输出：无。

    预期行为：`DIM_ORDER` 恰好是「由文本承载」的两维，一个不多一个不少——
    多一维就会让意图去写状态行管的事，少一维就有标签没人承载。
    """
    assert render.DIM_ORDER == ("tool", "payload")
    assert set(render.DIM_ORDER) == set(features.CHOICE_OPTIONS) & {"tool", "payload"}


def test_every_family_has_questions_and_criteria() -> None:
    """输入：六个题族。输出：无。

    预期行为：每族至少三条问法（问法轮换才有意义）；choice 族的 criteria 恰好覆盖该族选项；
    noul 族有 true / false；score 族有 `features.SEVERITY_LEVELS` 档。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for family in questions.FAMILIES:
            spec = loc["fam"][family]
            assert len(spec["q"]) >= 3, f"{lang}.{family} 问法太少"
            if family in questions.CHOICE_FAMILIES:
                assert set(spec["criteria"]) == set(features.CHOICE_OPTIONS[family])
            elif family in questions.NOUL_FAMILIES:
                assert set(spec["criteria"]) == {"true", "false"}
            else:
                assert len(spec["levels"]) == features.SEVERITY_LEVELS


def test_trace_options_match_features() -> None:
    """输入：`TraceDraft` 的字面量域、locale 的 criteria。输出：无。

    预期行为：反解草稿的取值域、locale 的选项 key、`features` 的取值表三者必须一致——
    任何一处串了，「反解答对了」与「标签对了」就不是一回事。
    """
    tool_field = judges.TraceDraft.model_fields["tool"].annotation
    payload_field = judges.TraceDraft.model_fields["payload"].annotation
    assert typing.get_args(tool_field) == features.TOOLS
    assert typing.get_args(payload_field) == features.PAYLOADS
    for lang in locales.available():
        loc = locales.load(lang)
        for family in judges.TRACE_FAMILIES:
            assert set(loc["fam"][family]["criteria"]) == set(features.DIMS[family])


def test_rule_table_matches_features_order_and_verdicts() -> None:
    """输入：两语言的策略表。输出：无。

    预期行为：表里的规则名与顺序必须与 `features.RULES` 完全一致、处置必须与
    `features.RAW_DECISIONS` 一致——题面那张表与代码算答案用的是同一张表。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["rules"]) == list(features.RULES)
        for name in features.RULES:
            assert str(loc["rules"][name]["verdict"]) == features.RAW_DECISIONS[name]
            assert str(loc["rules"][name]["condition"]).strip()


def test_rules_are_formattable_and_rotatable() -> None:
    """输入：每条提示词规则、每个问法编号。输出：无。

    预期行为：规则里只允许 `{language}` 占位符且能格式化；问法轮换在三个编号上都能取到文案。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for rule in loc["prompt"]["rules"]:
            text = str(rule)
            assert text.format(language=loc["language_name"])
            assert "{" not in text.format(language=loc["language_name"])
        for index in range(len(loc["prompt"]["rules"])):
            rotated = list(loc["prompt"]["rules"][index:]) + list(loc["prompt"]["rules"][:index])
            assert rotated
        for family in questions.FAMILIES:
            for phrasing in (0, 1, 2):
                assert locales.pick(loc["fam"][family]["q"], phrasing)


def test_rules_forbid_writing_the_state_row() -> None:
    """输入：两语言的提示词规则。输出：无。

    预期行为：规则里必须明确写着「不许写作用范围 / 策略档位 / 审批状态」——
    这是防「意图与状态行打架」的唯一一道文字防线，删了它提示词就自相矛盾了。
    """
    for lang, needles in (
        ("zh", ("作用范围", "策略档位", "审批状态")),
        ("en", ("where the call reaches", "policy tier", "approval status")),
    ):
        joined = " ".join(str(rule) for rule in locales.load(lang)["prompt"]["rules"])
        for needle in needles:
            assert needle in joined, f"{lang} 的规则里没提 {needle}"


def test_rules_forbid_copying_the_tool_identifier() -> None:
    """输入：两语言的提示词规则。输出：无。

    预期行为：规则里必须明确写着「不许把工具名 / 标识符原样照抄进意图」。
    工具那一维的答案就是工具名本身，照抄等于把答案写在题面上——英文批次真机上
    就出现过 `using read-file`（正好是规则里列出来的拼法）。这是防照抄的文字防线，
    配合 `banned_intent_phrases` 里那批工具名一起生效。
    """
    for lang, needles in (
        ("zh", ("工具名", "标识符", "照抄")),
        ("en", ("tool name", "identifier", "verbatim")),
    ):
        joined = " ".join(str(rule) for rule in locales.load(lang)["prompt"]["rules"])
        for needle in needles:
            assert needle in joined, f"{lang} 的规则里没提 {needle}"


def test_banned_phrases_are_present_and_usable() -> None:
    """输入：两语言的禁用语清单。输出：无。

    预期行为：每条禁用语都能被 `render.banned_echo` 认出来（写成一句话就能触发），
    而且清单非空——空清单等于这道防线不存在。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        phrases = loc["banned_intent_phrases"]
        assert phrases, f"{lang} 没有禁用语"
        for phrase in phrases:
            # 禁用语之间允许互相包含（`policy tier` 里就有 `policy`），所以只要求
            # 这条话术确实被认出来，不要求它是唯一命中。
            assert str(phrase) in render.banned_echo(f"我打算{phrase}做这件事", loc)


def test_banned_phrases_do_not_fire_on_the_fixture_intents() -> None:
    """输入：无。输出：无。

    预期行为：假后端那 14 句意图一条都不许撞禁用语——撞了就说明禁用语写得过宽，
    真跑时会把本来能用的数据在渲染期白拒掉。
    """
    from tests.workflows.toolgate.conftest import INTENTS

    for lang in locales.available():
        loc = locales.load(lang)
        for payloads in INTENTS.values():
            for text in payloads.values():
                assert render.banned_echo(text, loc) == [], f"{text} 撞了 {lang} 的禁用语"


def test_meta_placeholders_render() -> None:
    """输入：两语言的 meta。输出：无。

    预期行为：段落头与括号模板都能格式化，分隔符与行前缀都不是空的。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        meta = loc["meta"]
        # 段落头模板必须真的带 `{title}` 占位符，否则所有段落会渲染成同一个名字。
        assert "{title}" in str(meta["header"])
        assert locales.header(loc, "X") == str(meta["header"]).format(title="X")
        assert locales.header(loc, meta["intent_label"]) == str(meta["header"]).format(
            title=meta["intent_label"]
        )
        assert str(meta["paren"]).format(text="t")
        assert str(meta["row_prefix"]).strip()
        assert str(meta["colon"]).strip()


