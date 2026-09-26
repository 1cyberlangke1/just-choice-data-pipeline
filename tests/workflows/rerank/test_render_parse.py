"""渲染与反解：逐格往返、噪声行、缺格报错、措辞只改文本。"""

from __future__ import annotations

from typing import Any

import pytest

from workflows.rerank import locales, parse, render
from workflows.rerank import spec as sp

LANGS = ("zh", "en")


@pytest.mark.parametrize("lang", LANGS)
def test_render_parse_roundtrip(lang: str, make_spec: Any) -> None:
    """输入：一份规格、一份语言文件。输出：无。预期行为：查询与四条候选逐格还原。"""
    loc = locales.load(lang)
    spec = make_spec(traps=("forwarded", "edited"), wording="titled")
    parsed = parse.parse_list(render.build(spec, loc), parse.label_index(loc))
    assert parsed.query == loc["query"]["text"].format(
        topic=loc["topic"][spec.case.topic], aspect=loc["aspect"][spec.case.aspect]
    )
    assert [cand.cid for cand in parsed.cands] == list(render.CAND_IDS)
    for index, cand in enumerate(parsed.cands):
        source = spec.case.shape.cands[index]
        assert cand.source == loc["source"][source.source]
        assert cand.day == source.day
        assert cand.body == render.cand_lines(spec.case, index, loc)


def test_render_is_pure(make_spec: Any) -> None:
    """输入：同一份规格渲染两次。输出：无。预期行为：逐字符一致。"""
    loc = locales.load("zh")
    spec = make_spec()
    assert render.build(spec, loc) == render.build(spec, loc)


def test_cand_ids_are_the_expected_ones() -> None:
    """输入：四个下标。输出：无。预期行为：编号就是 C1~C4（判别器认的词汇表）。"""
    assert [render.cand_id(index) for index in range(4)] == ["C1", "C2", "C3", "C4"]


def test_cand_lines_are_body_plus_traps_plus_example(make_spec: Any) -> None:
    """输入：带两条噪声句的规格。输出：无。

    预期行为：正文行数 = 1（主题 / 方面）+ 噪声句 + 1（示例），示例句在最后一行。
    """
    loc = locales.load("zh")
    spec = make_spec(traps=("saved", "edited"))
    lines = render.cand_lines(spec.case, 0, loc)
    assert len(lines) == 1 + 2 + 1
    assert lines[1:3] == (loc["sentence"]["trap"]["saved"], loc["sentence"]["trap"]["edited"])
    assert lines[-1] in (
        loc["sentence"]["example"]["plain"]["yes"],
        loc["sentence"]["example"]["plain"]["no"],
    )


def test_trap_order_follows_the_key_table(make_spec: Any) -> None:
    """输入：乱序写下的噪声句组合。输出：无。预期行为：按 `TRAPS_KEYS` 归一。"""
    case = make_spec(traps=("edited", "forwarded")).case
    assert render.ordered_traps(case) == ("forwarded", "edited")


def test_noise_lines_between_fields_are_ignored(make_spec: Any) -> None:
    """输入：字段之间插了一行无关文本。输出：无。预期行为：噪声被忽略，字段照旧。"""
    loc = locales.load("zh")
    spec = make_spec()
    clean = parse.parse_list(render.build(spec, loc), parse.label_index(loc))
    noisy = parse.parse_list(
        render.build(spec, loc).replace("\n", "\n（系统提示：本行是噪声）\n", 1),
        parse.label_index(loc),
    )
    assert noisy == clean


def test_missing_query_is_reported(make_spec: Any) -> None:
    """输入：删掉「查询」那一行。输出：无。预期行为：报人话，点名缺的是查询。"""
    loc = locales.load("zh")
    spec = make_spec()
    label = loc["meta"]["query_label"]
    lines = [line for line in render.build(spec, loc).splitlines() if not line.startswith(label)]
    with pytest.raises(ValueError, match="查询"):
        parse.parse_list("\n".join(lines), parse.label_index(loc))


def test_missing_candidates_are_reported(make_spec: Any) -> None:
    """输入：只留抬头与查询。输出：无。预期行为：报「一条候选都没有」。"""
    loc = locales.load("zh")
    spec = make_spec()
    head = render.build(spec, loc).split(loc["meta"]["cand_label"])[0]
    with pytest.raises(ValueError, match="一条候选都没有"):
        parse.parse_list(head, parse.label_index(loc))


def test_candidate_without_body_is_reported(make_spec: Any) -> None:
    """输入：把 C3 的正文行删掉。输出：无。预期行为：点名那条候选一行正文都没有。"""
    loc = locales.load("zh")
    spec = make_spec()
    lines = render.build(spec, loc).splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("C3"))
    end = next(index for index in range(start + 1, len(lines)) if lines[index].startswith("C4"))
    broken = lines[: start + 1] + lines[end:]
    with pytest.raises(ValueError, match="C3"):
        parse.parse_list("\n".join(broken), parse.label_index(loc))


def test_reparse_is_language_independent(make_spec: Any) -> None:
    """输入：中英两版 state。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，靠合并标签表反解——两版都认得出，
    且正文就是各语言的原文。
    """
    spec = make_spec()
    for lang in LANGS:
        loc = locales.load(lang)
        parsed = parse.parse_list(render.build(spec, loc), parse.merged_labels())
        assert [cand.cid for cand in parsed.cands] == list(render.CAND_IDS)
        assert parsed.cands[0].body == render.cand_lines(spec.case, 0, loc)


def test_header_needs_a_separator_and_a_label() -> None:
    """输入：三种「长得像表头」的行。输出：无。

    预期行为：只有「编号 + `|` + 标签: 值」才算表头；正文行里带 `|` 不会被误认，
    没有 `|` 的那一行也不是表头。
    """
    labels = parse.label_index(locales.load("zh"))
    assert parse._header("C1 | 来源：官方 | 更新：2024-07-05", labels) is not None
    assert parse._header("这段有 | 竖线但不是表头", labels) is None
    assert parse._header("C1 来源：官方", labels) is None


def test_header_missing_a_field_is_reported() -> None:
    """输入：表头缺「更新」那一格。输出：无。预期行为：报人话点名缺哪一格。"""
    labels = parse.label_index(locales.load("zh"))
    with pytest.raises(ValueError, match="day_label"):
        parse._header("C1 | 来源：官方", labels)


def test_other_words_never_collide_with_the_query() -> None:
    """输入：全部主题 / 方面与四个候选位。输出：无。

    预期行为：「另一个主题 / 另一个方面」永远不等于查询那一个——这是「主题不一致」
    那条判据能被读出来的前提。
    """
    for topic in sp.TOPICS:
        for index in range(4):
            assert render.other_topic(topic, index) != topic
    for aspect in sp.ASPECTS:
        for index in range(4):
            assert render.other_aspect(aspect, index) != aspect


def test_cand_words_follow_the_shape_bits(make_spec: Any) -> None:
    """输入：一条规格。输出：无。预期行为：布尔位为假时正文写的是「另一个」词。"""
    case = make_spec(topic="cache", aspect="failures").case
    for index in range(4):
        topic, aspect = render.cand_words(case, index)
        cand = case.shape.cands[index]
        assert (topic == case.topic) is cand.topic_ok
        assert (aspect == case.aspect) is cand.aspect_ok
