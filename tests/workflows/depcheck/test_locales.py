"""语言文件测试：结构不变量（抬头、标记表、定级规则、档位表、生态、占位符）逐条钉住。"""

from __future__ import annotations

import re

import pytest

from workflows.depcheck import features, judges, locales, questions, render


def license_for(case: features.DepCheckCase) -> str:
    """输入：一个 case。输出：与它的判定计划相容的许可证标识符。

    预期行为：从 `features.license_pool` 给的那一半里取第一个——提示词渲染本身不校验这个值，但
    拿一个**相容**的值更贴近真实调用（渲染期的 `check_draft` 就是按它判命中组合的）。
    """
    return features.license_pool(case.tags())[0]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_fact_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：标记表的键与 `TAG_IDS` 一一对应，每条三个字段都非空、标记名与表行两两不同，而且
    表行里**编号排在分隔符前面**（解析就是按这个切回编号的）。
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
def test_avoid_lists_do_not_trip_other_tags(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每条标记的「不许出现」文案**不许误触别的**标记——它是提示词里给模型看的反面清单，
    踩到别的标记就会把模型往错的方向带（提示词里唯一的具体文字就是标记原文与这份反面清单，
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

    预期行为：标记原文与反面文案里**一个花括号都不许有**——这两段会被 `build_prompt` 拼进提示词，
    而花括号是填槽标记（`test_prompts_and_questions_leave_no_placeholder` 会扫「一个都不剩」）。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        entry = render.fact_copy(loc, tag)
        for key in ("rule", "avoid"):
            text = str(entry[key])
            assert "{" not in text and "}" not in text, f"{lang}.{tag}.{key}: {text!r}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_eco_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：生态表覆盖 `ECOSYSTEMS`，描述两两不同。"""
    loc = locales.load(lang)
    table = loc["eco"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.ECOSYSTEMS)
    descs = [render.eco_copy(loc, eco) for eco in features.ECOSYSTEMS]
    assert all(desc.strip() for desc in descs)
    assert len(set(descs)) == len(descs)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四族的问法池非空；逐标记那一族有 true / false 判据；`combo` 的「一个都没命中」与
    连接词都在（选项描述靠它们拼）。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert isinstance(spec["q"], list) and spec["q"], f"{family} 没有问法"
    assert str(loc["fam"]["flag"]["criteria_true"]).strip()
    assert str(loc["fam"]["flag"]["criteria_false"]).strip()
    assert str(loc["fam"]["combo"]["none"]).strip()
    assert str(loc["fam"]["combo"]["joiner"])


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompts_and_questions_leave_no_placeholder(lang: str) -> None:
    """输入：一种语言 × 全空间 48 组 case。输出：无。

    预期行为：提示词、反解提示词与题面渲染完之后**一个花括号都不剩**——漏填会在 `format` 上抛错，
    而多写的花括号会原样漏给模型，让它以为那是要填的槽。
    """
    loc = locales.load(lang)
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case, 0, 47, license_for(case))
        assert "{" not in prompt and "}" not in prompt, case.describe()
        for item in questions.build(features.plan_for(case, 0), loc, 0):
            instructions = str(item.question["instructions"])
            assert "{" not in instructions and "}" not in instructions
    trace = judges.build_trace_prompt("[state]", loc)
    assert "{" not in trace and "}" not in trace


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_spec_offers_no_copyable_value(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：格式规格里**不许出现填好的成品**——给一个 `count=3`、或者给一个真的 `archived`，
    模型照抄下去正好命中那一档（范式 §9.52）。规格只印键名与 `<...>` 占位符。
    """
    loc = locales.load(lang)
    for line in render.format_lines(loc):
        assert not re.search(r"count=[1-9]", line), line
        assert "AGPL-3.0-only" not in line, line
        assert "0.9.1" not in line, line


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_pins_the_prose_language(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：顶层有 `language_name`（依赖备注的语言），提示词里得有一句把它钉死的约束——不写
    这一句，模型会照它自己的先验换语言。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    assert "{language}" in str(loc["prompt"]["lang_rule"])
    case = features.DepCheckCase("D1", features.ECO_NPM)
    assert str(loc["language_name"]) in render.build_prompt(loc, case, 0, 47, license_for(case))


@pytest.mark.parametrize(
    ("lang", "needle"),
    [("zh", "不要真的去装这个包"), ("en", "do not actually install the package")],
)
def test_prompt_says_the_record_is_text_only(lang: str, needle: str) -> None:
    """输入：一种语言、该语言里那句「只写不执行」的说法。输出：无。

    预期行为：提示词里**明说这一族只要文本**——不许真的去装包、连注册表、查漏洞库。判据认的也
    只是字面，查不查都不改变标签；写清楚是为了让模型不去「验证」它自己编出来的记录（主人明确
    要求过：不许实际执行合成出来的东西）。
    """
    loc = locales.load(lang)
    case = features.DepCheckCase("D3", features.ECO_CARGO)
    assert needle in render.build_prompt(loc, case, 0, 47, license_for(case))


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_category_table_matches_the_pool(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：依赖类别词表长度等于 `features.CATEGORY_POOL`（spec 存的是下标，短一条会在渲染期
    抛 `IndexError`，长一条就是两种语言错位）、条目非空两两不同，且一个花括号都不许有。
    """
    loc = locales.load(lang)
    items = loc["prompt"]["categories"]
    assert isinstance(items, list)
    assert len(items) == features.CATEGORY_POOL
    texts = [str(item) for item in items]
    assert all(text.strip() for text in texts)
    assert len(set(texts)) == len(texts)
    assert all("{" not in text and "}" not in text for text in texts)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_carries_every_category(lang: str) -> None:
    """输入：一种语言 × 全部类别下标。输出：无。

    预期行为：提示词里出现的是**那一个类别**的词条，而且抽到不同类别的两条样本提示词**逐字不同**
    ——这正是样本级旋钮（范式 §9.57 / §9.58）要的效果：同生态的样本不再共用一份提示词。
    """
    loc = locales.load(lang)
    case = features.DepCheckCase("D1", features.ECO_NPM)
    prompts = [
        render.build_prompt(loc, case, category, 47, license_for(case))
        for category in features.CATEGORY_IDS
    ]
    assert len(set(prompts)) == len(prompts)
    for category, prompt in zip(features.CATEGORY_IDS, prompts, strict=True):
        assert locales.pick(loc["prompt"]["categories"], category) in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_version_line_carries_the_placeholder_and_the_pool(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：版本号那一行带 `{version}` 占位符（渲染时填的是 `x.y.<尾段>` 那个形状）、
    条目非空、没有多余的花括号；抽到不同尾段的两条样本提示词**逐字不同**——这正是第二个样本级
    旋钮（范式 §9.60）要的效果：`version:` 不再反复用同一个数字。
    """
    loc = locales.load(lang)
    line = str(loc["prompt"]["version_line"])
    assert "{version}" in line
    assert line.replace("{version}", "").count("{") == 0
    case = features.DepCheckCase("D1", features.ECO_NPM)
    prompts = [
        render.build_prompt(loc, case, 0, hint, license_for(case)) for hint in features.VERSION_HINTS
    ]
    assert len(set(prompts)) == len(prompts)
    for hint, prompt in zip(features.VERSION_HINTS, prompts, strict=True):
        assert render.version_shape(hint) in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_license_line_carries_the_placeholder_and_the_pool(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：许可证那一行带 `{license}` 占位符（渲染时填的是 `spec.license`）、条目非空、没有
    多余的花括号；抽到不同标识符的两条样本提示词**逐字不同**——这正是第三个样本级旋钮
    （范式 §9.61）要的效果：`license:` 不再整批都写 `MIT`。
    """
    loc = locales.load(lang)
    line = str(loc["prompt"]["license_line"])
    assert "{license}" in line
    assert line.replace("{license}", "").count("{") == 0
    case = features.DepCheckCase("D1", features.ECO_NPM)
    prompts = [render.build_prompt(loc, case, 0, 47, item) for item in features.LICENSE_COPYLEFT]
    assert len(set(prompts)) == len(prompts)
    for item, prompt in zip(features.LICENSE_COPYLEFT, prompts, strict=True):
        assert item in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_verdict_words_are_present_and_safe(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：结论词表非空、条目非空两两不同、一个花括号都不许有；而且**要求写的内容**（「维护」
    这种）不许在表里——表里只放结论词，否则写手写合格备注时会被自己的闸拦下来（范式 §9.59）。
    """
    loc = locales.load(lang)
    words = render.verdict_words(loc)
    assert words
    assert all(word.strip() for word in words)
    assert len(set(words)) == len(words)
    assert all("{" not in word and "}" not in word for word in words)
    note = str(loc["prompt"]["format"]["note"])
    # 备注那一行的规格里要**明说**不许写哪些结论（列出来当反面例子）；写手照抄这一整段也会被
    # 结论闸拦下——那正是我们要的。
    assert render.note_verdicts(note, loc)
