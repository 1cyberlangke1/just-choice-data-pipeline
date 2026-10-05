"""语言文件测试：结构不变量（抬头、判据表、处置档位、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.posttriage import features, locales, questions, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_signal_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：判据表的键与 `SIGNAL_IDS` 一一对应，每条四个字段都非空、判据名与表行两两不同，
    而且表行里**编号排在分隔符前面**（解析就是按这个切回编号的）。
    """
    loc = locales.load(lang)
    table = loc["signal"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.SIGNAL_IDS)
    for signal in features.SIGNAL_IDS:
        entry = render.signal_copy(loc, signal)
        for key in ("name", "rule", "example", "avoid"):
            assert str(entry[key]).strip(), f"{lang}.{signal}.{key}"
    names = [str(table[signal]["name"]) for signal in features.SIGNAL_IDS]
    assert len(set(names)) == len(names)
    sep = str(loc["state"]["signal_sep"])
    lines = [render.signal_line(loc, signal) for signal in features.SIGNAL_IDS]
    assert len(set(lines)) == len(lines)
    for signal, line in zip(features.SIGNAL_IDS, lines, strict=True):
        assert line.partition(sep)[0] == signal


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_examples_hit_exactly_their_own_signal(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每条的示例片段只命中它自己那一条——示例同时命中两条，提示词就会把模型往错的方向带。
    """
    loc = locales.load(lang)
    for signal in features.SIGNAL_IDS:
        example = str(render.signal_copy(loc, signal)["example"])
        assert features.signal_set(example) == (signal,), f"{lang}.{signal}: {example!r}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_no_example_is_a_decoy(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四条判据的示例**都不是**诱饵形状——示例既是真命中的样本又是诱饵的话，
    `decoy=none` 那一侧照着例子写就必然被 `check_draft` 拒掉。
    """
    loc = locales.load(lang)
    for signal in features.SIGNAL_IDS:
        example = str(render.signal_copy(loc, signal)["example"])
        assert not features.has_decoy(example), f"{lang}.{signal}: {example!r}"
    assert features.has_decoy(features.DECOY_EXAMPLE)
    assert not features.signal_set(features.DECOY_EXAMPLE)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_vary_rule_bans_the_examples_by_name(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：`vary_rule` 里点名**每一个**例子本身——只说「自己另编」时模型照样照抄（真机 en 侧
    第一批：6 条带百分号的帖子里 4 条写的都是例子那个 8.4%），等于把「背下这个值 = 命中这条判据」
    的捷径长出来。这条尺子同时钉住「例子一改，禁止清单跟着改」。
    """
    loc = locales.load(lang)
    rule = str(loc["prompt"]["vary_rule"])
    for signal in features.SIGNAL_IDS:
        example = str(render.signal_copy(loc, signal)["example"])
        assert example in rule, f"{lang}.{signal} 的例子 {example!r} 没进 vary_rule"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四族的问法池非空；逐判据那一族有 true / false 判据；count 与 combo 的选项表齐全。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert isinstance(spec["q"], list) and spec["q"], f"{family} 没有问法"
    assert str(loc["fam"]["flag"]["criteria_true"]).strip()
    assert str(loc["fam"]["flag"]["criteria_false"]).strip()
    assert sorted(loc["fam"]["count"]["opt"]) == [
        f"c{value}" for value in range(len(features.SIGNAL_IDS) + 1)
    ]
    assert str(loc["fam"]["combo"]["none"]).strip()
    assert str(loc["fam"]["combo"]["joiner"])
