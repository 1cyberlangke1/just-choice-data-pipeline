"""打分与评分的多语言文案测试：词表与代码表必须一一对得上。"""

from __future__ import annotations

import pytest

from workflows.rubric import features, judges, locales, questions, render


def all_keys(node: object, path: str = "") -> set[str]:
    """输入：解析后的 TOML 节点、路径前缀。输出：全部叶子路径。

    预期行为：列表按「一个叶子」处理（不展开下标）——两个语言文件的**结构**要一致，
    值当然不一样。
    """
    if isinstance(node, dict):
        out: set[str] = set()
        for key, value in node.items():
            out |= all_keys(value, f"{path}.{key}")
        return out
    return {path}


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_locale_covers_every_criterion_and_family(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：五条判据的名字、题族的文案都齐全，且 choice 的选项名与代码里的取值表
    **完全一致**（多一个少一个都会让「答对了」与「标签对了」不是一回事）。
    """
    loc = locales.load(lang)
    assert set(loc["criterion"]) == set(features.CRITERIA)
    for name in features.CRITERIA:
        assert str(loc["criterion"][name]).strip()
    assert set(loc["fam"]) == set(questions.FAMILIES)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert [str(item).strip() for item in spec["q"]]
        if family in questions.CHOICE_FAMILIES:
            assert set(spec["criteria"]) == set(features.CHOICE_OPTIONS[family]), family
        elif family in questions.NOUL_FAMILIES:
            assert set(spec["criteria"]) == {"true", "false"}, family
        else:
            assert len(spec["levels"]) == len(features.BANDS), family


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_locale_covers_the_trace_questions(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：反解问的两维与 `judges.TRACE_FAMILIES` 一致，且每维的选项名就是
    `features` 里的取值——反解用的口径与渲染期用的机械口径必须是同一份。
    """
    loc = locales.load(lang)
    assert set(loc["trace"]) == set(judges.TRACE_FAMILIES)
    for dim in judges.TRACE_FAMILIES:
        assert str(loc["trace"][dim]["q"]).strip()
        assert set(loc["trace"][dim]["criteria"]) == set(features.DIMS[dim]), dim
        for desc in loc["trace"][dim]["criteria"].values():
            assert str(desc).strip()


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_locale_meta_and_columns(lang: str) -> None:
    """输入：一个语言。输出：无。预期行为：段落名、行前缀与四张表的列名齐全。"""
    loc = locales.load(lang)
    meta = loc["meta"]
    for key in render.SECTION_KEYS:
        assert str(meta[key]).strip()
    for key in ("header", "row_prefix", "paren", "colon", "count_mark"):
        assert str(meta[key]).strip()
    for key in ("manifest_columns", "score_columns", "band_columns", "gate_columns"):
        assert str(meta[key]).strip()
    assert str(loc["verify"]["intro"]).strip()
    assert str(loc["verify"]["option_line"]).strip()
    assert "{name}" in str(loc["verify"]["option_line"])


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_banned_phrases_are_the_table_names(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：禁写清单里就是那几个表名——材料是一段自然语言，「篇幅」这类词在正常叙述里
    会出现，收进清单会大面积误伤（见 `render.banned_echo` 的口径）。
    """
    loc = locales.load(lang)
    banned = [str(item) for item in loc["banned_intent_phrases"]]
    assert banned
    # 比对时统一小写：英文的段落名首字母大写，而禁写清单按小写匹配（`banned_echo` 也是）
    titles = {str(loc["meta"][key]).lower() for key in render.SECTION_KEYS}
    assert {item.lower() for item in banned} == titles


def test_trace_draft_matches_the_dimension_tables() -> None:
    """输入：反解草稿的取值域。输出：无。

    预期行为：`TraceDraft` 的 Literal 与 `features.PLACEHOLDERS` / `CITATIONS` 完全一致，
    而且反解问的就是 `TRACE_FAMILIES` 那两族。
    """
    holders = judges.TraceDraft.model_fields["placeholders"]
    cites = judges.TraceDraft.model_fields["citations"]
    assert set(holders.annotation.__args__) == set(features.PLACEHOLDERS)
    assert set(cites.annotation.__args__) == set(features.CITATIONS)
    assert judges.TRACE_FAMILIES == ("placeholders", "citations")
