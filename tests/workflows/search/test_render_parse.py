"""渲染与反解的往返：文本 → 字段 → 文本必须一致，标签表必须语言无关。"""

from __future__ import annotations

import random

import pytest

from workflows.search import judges, locales, parse, render
from workflows.search import spec as sp
from workflows.search import workflow as wf


def sample_cases(limit: int = 24) -> list[sp.SearchCase]:
    """输入：条数上限。输出：从留出档里均匀取的一批 case（八维都在动）。"""
    cases = sp.build_cases(wf.OOD_CONFIG)
    step = max(1, len(cases) // limit)
    return cases[::step][:limit]


@pytest.mark.parametrize("locale", ["zh", "en"])
def test_render_then_parse_roundtrips(locale: str) -> None:
    """输入：一批 case × 一种语言。输出：无。

    预期行为：渲染出来的文本反解回去，请求头四个字段与每一条的七格全部逐字相等。
    """
    loc = locales.load(locale)
    for index, case in enumerate(sample_cases()):
        spec = sp.sample_spec(random.Random(index), case, "train", index, 0)
        state = render.build(spec, loc)
        page = parse.parse_page(state, parse.label_index(loc), (loc["meta"]["field_sep"],))
        assert judges.roundtrip_diff(spec, page, loc) == []


def test_reparse_uses_the_merged_table() -> None:
    """输入：一条英文 state。输出：反解结果。

    预期行为：`reparse` 没有语言参数，但合并表能认英文标签与英文分隔符。
    """
    spec = sp.sample_spec(
        random.Random(3),
        sp.SearchCase("webhook", "empty", sp.MIXES[0], (3, 4), 5, "both", 2, 1),
        "train",
        0,
        0,
    )
    state = render.build(spec, locales.load("en"))
    page = wf.SearchWorkflow().reparse(state)
    assert page.query == render.query_text(spec, locales.load("en"))
    assert [item.index for item in page.results] == [row.index for row in spec.results]
    assert page.results[0].doc_id == spec.results[0].doc_id
    assert page.points == tuple(locales.load("en")["topic"]["webhook"]["points"])


def test_parse_ignores_unknown_lines() -> None:
    """输入：混了噪声行的 state。输出：反解结果。预期行为：噪声被忽略，字段不受污染。"""
    loc = locales.load("zh")
    spec = sp.sample_spec(
        random.Random(4),
        sp.SearchCase("refund", "full", sp.MIXES[1], (1, 2), 3, "none", 1, 0),
        "train",
        0,
        0,
    )
    lines = render.build(spec, loc).splitlines()
    noisy = "\n".join([lines[0], "补充说明：这不是字段", *lines[1:]])
    page = parse.parse_page(noisy, parse.label_index(loc), (loc["meta"]["field_sep"],))
    assert len(page.results) == len(spec.results)
    assert page.want_kind == loc["doc_kind"][spec.want_kind]


def test_parse_reports_a_broken_result_line() -> None:
    """输入：一条少了两格的标题行 / 一行摘要跑到结果之前。输出：异常。

    预期行为：报人话并带上原文——渲染错位必须当场炸，不能静默错位。
    """
    loc = locales.load("zh")
    labels = parse.label_index(loc)
    seps = (loc["meta"]["field_sep"],)
    with pytest.raises(ValueError, match="五格"):
        parse.parse_page("  [1] KB-2001 · 标题 · 操作手册", labels, seps)
    with pytest.raises(ValueError, match="摘要行出现在任何结果之前"):
        parse.parse_page(f"{loc['meta']['snippet_label']}：没有结果", labels, seps)
    with pytest.raises(ValueError, match="不是"):
        parse.parse_page("  [] KB-2001 · 标题 · 操作手册 · 官方文档 · 更新 2024-01-01", labels, seps)


def test_merged_seps_cover_every_language() -> None:
    """输入：全部语言。输出：无。预期行为：每种语言的标题分隔符都在合并表里。"""
    seps = parse.merged_seps()
    for lang in locales.available():
        assert locales.load(lang)["meta"]["field_sep"] in seps


def test_query_text_follows_the_case_style() -> None:
    """输入：同一主题的三种问法。输出：三句话。预期行为：都不一样，且都含主题名。"""
    loc = locales.load("zh")
    texts = {
        render.query_text(
            sp.sample_spec(
                random.Random(1),
                sp.SearchCase("refund", "full", sp.MIXES[1], (1, 2), 3, "conflict", 1, style),
                "train",
                0,
                0,
            ),
            loc,
        )
        for style in range(3)
    }
    assert len(texts) == 3
    assert all(loc["topic"]["refund"]["label"] in text for text in texts)
