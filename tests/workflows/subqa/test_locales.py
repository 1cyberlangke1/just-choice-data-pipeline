"""语言文件测试：结构不变量（抬头、判据表、处置档位、素材类型、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.subqa import features, judges, locales, questions, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_rule_and_avoid_copy_never_carries_braces(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：判据原文与反面文案里**一个花括号都不许有**——这两段会被 `build_prompt` 拼进
    提示词，而花括号是填槽标记（`test_prompts_and_questions_leave_no_placeholder` 会扫
    「一个都不剩」）。这里踩过一次：S2 的原文里写了 `{反斜杠+短代码}` 那个具体形状，提示词的
    填槽自检当场把它当成没填的槽报出来。要举例就写「一对花括号里夹着反斜杠开头的短代码」这种
    说法，别把花括号本身写进去。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        entry = render.tag_copy(loc, tag)
        for key in ("rule", "avoid"):
            text = str(entry[key])
            assert "{" not in text and "}" not in text, f"{lang}.{tag}.{key}: {text!r}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_track_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：素材类型表的键与 `TRACKS` 一一对应，每种形态的说明都非空、两两不同——三种素材
    写成同一句话，这一维的留出就白留了。
    """
    loc = locales.load(lang)
    table = loc["track"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.TRACKS)
    descs = [render.track_copy(loc, track) for track in features.TRACKS]
    assert all(desc.strip() for desc in descs)
    assert len(set(descs)) == len(descs)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompts_and_questions_leave_no_placeholder(lang: str) -> None:
    """输入：一种语言 × 全空间 48 组 case。输出：无。

    预期行为：提示词、反解提示词、题面与选项渲染完之后**一个花括号都不剩**——漏填会在 `format`
    上抛错，而多写的花括号会原样漏给模型，让它以为那是要填的槽。
    """
    loc = locales.load(lang)
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case, features.TIMECODE_HINTS[0])
        assert "{" not in prompt and "}" not in prompt, case.describe()
        for item in questions.build(features.plan_for(case, 0), loc, 0):
            instructions = str(item.question["instructions"])
            assert "{" not in instructions and "}" not in instructions
    trace = judges.build_trace_prompt("[state]", loc)
    assert "{" not in trace and "}" not in trace


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_pins_both_languages(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：顶层有 `language_name`（译文与备注的语言）与 `source_language`（原文那一行的
    语言），两者**必须不同**——这一族的样本是一对**原文与译文**，两种语言相同的话（en 侧的目标
    语言就是英文）译文会塌成原文的复制品，整条样本不再是双语字幕对。真机上量到过：修之前
    `sq-10a-en` 十条里九条的 `Target:` 是 `Source:` 的复制品。

    提示词里还得有一句把两种语言都钉死的约束——不写这一句，模型会照它自己的先验换语言
    （范式 §10.96 那条尺子的由来）。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    assert str(loc["source_language"]).strip()
    assert str(loc["source_language"]) != str(loc["language_name"])
    assert "{language}" in str(loc["prompt"]["lang_rule"])
    assert "{source_language}" in str(loc["prompt"]["lang_rule"])
    case = features.SubQaCase("S1", features.TRACK_MOVIE)
    prompt = render.build_prompt(loc, case, features.TIMECODE_HINTS[0])
    assert str(loc["language_name"]) in prompt
    assert str(loc["source_language"]) in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_timing_line_carries_the_placeholder_and_the_pool(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：时间码那一行带 `{timecode}` 占位符（渲染时填的是 `spec.timecode`）、条目非空、没有
    多余的花括号；抽到不同时间码的两条样本提示词**逐字不同**——这正是样本级旋钮（范式 §9.61）
    要的效果：`Timing:` 不再整批共用同一串数字。池子里每个取值的形状也在这一条里一起验掉。
    """
    loc = locales.load(lang)
    line = str(loc["prompt"]["timing_line"])
    assert "{timecode}" in line
    assert line.replace("{timecode}", "").count("{") == 0
    assert all(features.TIMECODE_SHAPE.match(item) for item in features.TIMECODE_HINTS)
    case = features.SubQaCase("S3", features.TRACK_MOVIE)
    prompts = [render.build_prompt(loc, case, item) for item in features.TIMECODE_HINTS]
    assert len(set(prompts)) == len(prompts)
    for item, prompt in zip(features.TIMECODE_HINTS, prompts, strict=True):
        assert item in prompt
