"""语言文件测试：结构不变量（抬头、因子表、矩阵、档位表、报警通道、占位符）逐条钉住。"""

from __future__ import annotations

import re

import pytest

from workflows.alertrisk import features, judges, locales, questions, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_fact_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：因子表的键与 `TAG_IDS` 一一对应，每条三个字段都非空、因子名与表行两两不同，
    而且表行里**编号排在分隔符前面**（解析就是按这个切回编号的）。
    """
    loc = locales.load(lang)
    table = loc["fact"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.TAG_IDS)
    for tag in features.TAG_IDS:
        entry = render.fact_copy(loc, tag)
        for key in ("name", "rule", "avoid"):
            assert str(entry[key]).strip(), f"{lang}.{tag}.{key}"
    names = [str(table[tag]["name"]) for tag in features.TAG_IDS]
    assert len(set(names)) == len(names)
    sep = str(loc["state"]["tag_sep"])
    lines = [render.fact_line(loc, tag) for tag in features.TAG_IDS]
    assert len(set(lines)) == len(lines)
    for tag, line in zip(features.TAG_IDS, lines, strict=True):
        assert line.partition(sep)[0] == tag


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_avoid_lists_do_not_trip_other_factors(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每条因子的「不许出现」文案**不许误触别的**因子——它是提示词里给模型看的反面清单，
    踩到别的因子就会把模型往错的方向带（提示词里唯一的具体文字就是因子原文与这份反面清单，
    所以它更要钉住）。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        avoid = str(render.fact_copy(loc, tag)["avoid"])
        stray = set(features.hit_set(avoid)) - {tag}
        assert not stray, f"{lang}.{tag}: {avoid!r} 误触了 {sorted(stray)}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_rule_and_avoid_copy_never_carries_braces(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：因子原文与反面文案里**一个花括号都不许有**——这两段会被 `build_prompt` 拼进提示词，
    而花括号是填槽标记（`test_prompts_and_questions_leave_no_placeholder` 会扫「一个都不剩」）。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        entry = render.fact_copy(loc, tag)
        for key in ("rule", "avoid"):
            text = str(entry[key])
            assert "{" not in text and "}" not in text, f"{lang}.{tag}.{key}: {text!r}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_level_and_act_tables_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：等级表覆盖 `LEVELS` 五档、档位表覆盖 `ACTS` 四档，两边的文案都两两不同。
    """
    loc = locales.load(lang)
    levels = loc["level"]
    assert isinstance(levels, dict)
    assert sorted(levels) == [f"L{level}" for level in features.LEVELS]
    texts = [render.level_text(loc, level) for level in features.LEVELS]
    assert len(set(texts)) == len(texts)
    acts = loc["act"]
    assert isinstance(acts, dict)
    assert sorted(acts) == sorted(features.ACTS)
    actions = [render.act_text(loc, act) for act in features.ACTS]
    assert len(set(actions)) == len(actions)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四族的问法池非空；逐因子那一族有 true / false 判据；`highs` 的「一个都没取到」与
    连接词都在（选项描述靠它们拼）。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert isinstance(spec["q"], list) and spec["q"], f"{family} 没有问法"
    assert str(loc["fam"]["fact"]["criteria_true"]).strip()
    assert str(loc["fam"]["fact"]["criteria_false"]).strip()
    assert str(loc["fam"]["highs"]["none"]).strip()
    assert str(loc["fam"]["highs"]["joiner"])


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompts_and_questions_leave_no_placeholder(lang: str) -> None:
    """输入：一种语言 × 全空间 24 组 case。输出：无。

    预期行为：提示词、反解提示词与题面渲染完之后**一个花括号都不剩**——漏填会在 `format` 上抛错，
    而多写的花括号会原样漏给模型，让它以为那是要填的槽。
    """
    loc = locales.load(lang)
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case, 0)
        assert "{" not in prompt and "}" not in prompt, case.describe()
        for item in questions.build(features.plan_for(case, 0), loc, 0):
            instructions = str(item.question["instructions"])
            assert "{" not in instructions and "}" not in instructions
    trace = judges.build_trace_prompt("[state]", loc)
    assert "{" not in trace and "}" not in trace


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_spec_keys_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：格式规格的键与 `KEY_LINES` 一一对应（键名是 state 格式的一部分，所以它写在
    `[prompt.format]` 里、按 `KEY_LINES` 取，键少一个模型就不知道那一行叫什么）。
    """
    loc = locales.load(lang)
    spec = loc["prompt"]["format"]
    assert isinstance(spec, dict)
    assert sorted(spec) == sorted(features.KEY_LINES)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_spec_offers_no_copyable_value(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：格式规格里**不许出现填好的成品**——给一个带数字的 `hosts=`、或者给一个真的
    `reachable=public`，模型照抄下去正好取到那一档（范式 §9.52）。规格只印键名与 `<...>` 占位符。
    """
    loc = locales.load(lang)
    for line in render.format_lines(loc):
        assert not re.search(r"hosts=\d", line), line
        assert "reachable=public" not in line, line
        assert "auth=none" not in line, line


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_pins_the_prose_language(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：顶层有 `language_name`（告警描述的语言），提示词里得有一句把它钉死的约束——不写
    这一句，模型会照它自己的先验换语言。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    assert "{language}" in str(loc["prompt"]["lang_rule"])
    case = features.AlertRiskCase("R1", features.SOURCE_EDR)
    assert str(loc["language_name"]) in render.build_prompt(loc, case, 0)


@pytest.mark.parametrize(
    ("lang", "needle"),
    [("zh", "不要真的去连任何机器"), ("en", "do not actually connect to any host")],
)
def test_prompt_says_the_record_is_text_only(lang: str, needle: str) -> None:
    """输入：一种语言、该语言里那句「只写不执行」的说法。输出：无。

    预期行为：提示词里**明说这一族只要文本**——不许真的去连机器、跑命令、查漏洞库。判据认的也
    只是字面，跑不跑都不改变标签；写清楚是为了让模型不去「验证」它自己编出来的警报（主人明确
    要求过：不许实际执行合成出来的东西）。
    """
    loc = locales.load(lang)
    case = features.AlertRiskCase("R3", features.SOURCE_CLOUD)
    assert needle in render.build_prompt(loc, case, 0)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_direction_table_matches_the_pool(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：手法方向词表长度等于 `features.DIRECTION_POOL`（spec 存的是下标，短一条会在渲染期
    抛 `IndexError`，长一条就是两种语言错位）、条目非空两两不同，且一个花括号都不许有。
    """
    loc = locales.load(lang)
    items = loc["prompt"]["directions"]
    assert isinstance(items, list)
    assert len(items) == features.DIRECTION_POOL
    texts = [str(item) for item in items]
    assert all(text.strip() for text in texts)
    assert len(set(texts)) == len(texts)
    assert all("{" not in text and "}" not in text for text in texts)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_carries_every_direction(lang: str) -> None:
    """输入：一种语言 × 全部方向下标。输出：无。

    预期行为：提示词里出现的是**那一个方向**的词条，而且抽到不同方向的两条样本提示词**逐字不同**
    ——这正是样本级旋钮（范式 §9.7）要的效果：同通道的样本不再共用一份提示词。
    """
    loc = locales.load(lang)
    case = features.AlertRiskCase("R1", features.SOURCE_EDR)
    prompts = [
        render.build_prompt(loc, case, direction) for direction in features.DIRECTION_IDS
    ]
    assert len(set(prompts)) == len(prompts)
    for direction, prompt in zip(features.DIRECTION_IDS, prompts, strict=True):
        assert locales.pick(loc["prompt"]["directions"], direction) in prompt
