"""语言文件测试：结构不变量（抬头、判据表、处置档位、运行时、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.debugstep import features, locales, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_runtime_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：运行时表的键与 `RUNTIMES` 一一对应，每种形态的说明都非空、两两不同——三种运行
    时写成同一句话，这一维的留出就白留了。
    """
    loc = locales.load(lang)
    table = loc["runtime"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.RUNTIMES)
    descs = [render.runtime_copy(loc, runtime) for runtime in features.RUNTIMES]
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
    case = features.DebugStepCase("D1", features.RUNTIME_PYTHON)
    assert str(loc["language_name"]) in render.build_prompt(loc, case)
