"""语言文件测试：结构不变量（抬头、判据表、处置档位、通道、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.turntake import features, locales, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_marks_are_non_empty_and_pairwise_non_prefix(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：三个抬头两两不互为前缀——`【未完判据】` 与 `【未完判据附注】` 那种关系会让解析先
    命中短的那个，整段 state 当场错位。
    """
    loc = locales.load(lang)
    marks = [str(loc["state"][key]) for key in ("tag_mark", "rule_mark", "content_mark")]
    assert all(marks)
    assert len(set(marks)) == len(marks)
    for left in marks:
        for right in marks:
            if left != right:
                assert not right.startswith(left), f"{left!r} 是 {right!r} 的前缀"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_channel_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：通道表的键与 `CHANNELS` 一一对应，每种形态的说明都非空、两两不同——
    三种通道写成同一句话，这一维的留出就白留了。
    """
    loc = locales.load(lang)
    table = loc["channel"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.CHANNELS)
    descs = [render.channel_copy(loc, channel) for channel in features.CHANNELS]
    assert all(desc.strip() for desc in descs)
    assert len(set(descs)) == len(descs)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_pins_the_language(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：顶层有 `language_name`，提示词里有一句把语言钉死的约束——不写这一句，模型会照它
    自己的先验换语言（范式 §10.96 那条尺子的由来）。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    assert "{language}" in str(loc["prompt"]["lang_rule"])
    case = features.TurnTakeCase("T1", features.CHANNEL_IM)
    assert str(loc["language_name"]) in render.build_prompt(loc, case)
