"""语言文件测试：结构不变量（抬头、判据表、处置档位、通道、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.logtriage import features, locales, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_pins_the_language(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：顶层有 `language_name`，提示词里有一句把语言钉死的约束——不写这一句，模型会照它
    自己的先验换语言（范式 §10.96 那条尺子的由来）。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    assert "{language}" in str(loc["prompt"]["lang_rule"])
    case = features.LogTriageCase("N1", features.SOURCE_APP)
    assert str(loc["language_name"]) in render.build_prompt(loc, case)
