"""渲染与反解：逐格往返、噪声行、缺格报错、多行正文。"""

from __future__ import annotations

from typing import Any

import pytest

from workflows.guardrail import locales, parse, render
from workflows.guardrail import spec as sp

LANGS = ("zh", "en")


@pytest.mark.parametrize("lang", LANGS)
def test_render_parse_roundtrip(lang: str, make_spec: Any) -> None:
    """输入：一份规格、一份语言文件。输出：无。预期行为：字段与正文逐格还原。"""
    loc = locales.load(lang)
    spec = make_spec(mentions=("meds", "privacy", "spam"))
    message = parse.parse_message(render.build(spec, loc), parse.label_index(loc))
    assert message.content_id == spec.content_id
    assert message.reach == loc["reach"][spec.case.reach]
    assert message.direction == loc["direction"][spec.case.direction]
    assert message.audience == loc["audience"][spec.case.audience]
    assert message.history == loc["history"][spec.case.history]
    assert message.body == render.body_lines(spec, loc)


def test_render_is_pure(make_spec: Any) -> None:
    """输入：同一份规格渲染两次。输出：无。预期行为：逐字符一致。"""
    loc = locales.load("zh")
    spec = make_spec()
    assert render.build(spec, loc) == render.build(spec, loc)


def test_body_lines_follow_the_slot_order(make_spec: Any) -> None:
    """输入：一份规格。输出：无。预期行为：正文句数 = 违规句 + 擦边句 + 填充句。"""
    spec = make_spec(mentions=("harass", "scam", "cheat"))
    lines = render.body_lines(spec, locales.load("zh"))
    assert len(lines) == len(spec.case.violations) + len(spec.case.mentions) + sp.FILLER_SLOTS


def test_noise_lines_are_ignored(make_spec: Any) -> None:
    """输入：字段之间插了一行无关文本。输出：无。预期行为：噪声被忽略，字段照旧。"""
    loc = locales.load("zh")
    spec = make_spec()
    text = render.build(spec, loc)
    noisy = text.replace("\n", "\n（系统提示：本行是噪声）\n", 1)
    message = parse.parse_message(noisy, parse.label_index(loc))
    assert message.content_id == spec.content_id
    assert message.body == render.body_lines(spec, loc)


def test_missing_field_reports_the_label(make_spec: Any) -> None:
    """输入：删掉「历史」那一行。输出：无。预期行为：报人话并点名缺哪一格。"""
    loc = locales.load("zh")
    spec = make_spec()
    label = loc["meta"]["history_label"]
    lines = [line for line in render.build(spec, loc).splitlines() if not line.startswith(label)]
    with pytest.raises(ValueError, match="history_label"):
        parse.parse_message("\n".join(lines), parse.label_index(loc))


def test_reparse_is_language_independent(make_spec: Any) -> None:
    """输入：中英两版 state。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，靠合并标签表反解——两版都认得出，
    且正文就是各语言的原文。
    """
    spec = make_spec()
    for lang in LANGS:
        text = render.build(spec, locales.load(lang))
        message = parse.parse_message(text, parse.merged_labels())
        assert message.body == render.body_lines(spec, locales.load(lang))


def test_body_label_swallows_the_rest(make_spec: Any) -> None:
    """输入：正文之后又跟了一行「内容号：...」。输出：无。

    预期行为：`正文` 之后的行**全部**算正文——正文里出现「标签：值」形状的句子时
    不能被当成字段（否则一条消息里带冒号就会被切坏）。
    """
    loc = locales.load("zh")
    spec = make_spec()
    text = render.build(spec, loc) + "\n" + loc["meta"]["content_label"] + loc["meta"]["colon"] + "MSG-00000"
    message = parse.parse_message(text, parse.label_index(loc))
    assert message.content_id == spec.content_id
    assert message.body[-1].endswith("MSG-00000")
