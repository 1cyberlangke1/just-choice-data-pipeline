"""来源选择的文案测试：选项名与特征表对齐、禁用语可用且不误伤。"""

from __future__ import annotations

from workflows.sourceselect import features, locales, questions, render

from tests.workflows.sourceselect.conftest import QUERIES


def test_criteria_keys_match_the_feature_tables() -> None:
    """输入：两语言文案。输出：无。

    预期行为：每个 choice / noul 题族的选项名，必须与特征层那一维的取值完全一致——
    对不上就会出现「选项里没有的答案」或「答不出选项里的答案」。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["fam"]["need"]["criteria"]) == list(features.NEEDS)
        assert list(loc["fam"]["enough"]["criteria"]) == ["true", "false"]
        assert list(loc["fam"]["near_miss"]["criteria"]) == ["true", "false"]
        assert len(loc["fam"]["cover"]["levels"]) == questions.COVER_LEVELS
        assert str(loc["fam"]["pick"]["opt"]["none"]).strip()


def test_every_prompt_dimension_value_has_text() -> None:
    """输入：两语言文案。输出：无。预期行为：提示词里那一维的每个取值都有 brief 与 hint。"""
    for lang in locales.available():
        loc = locales.load(lang)
        for dim in render.DIM_ORDER:
            for value in features.DIMS[dim]:
                assert str(loc["feature"][dim][value]["brief"]).strip()
                assert str(loc["feature"][dim][value]["hint"]).strip()


def test_banned_phrases_do_not_fire_on_the_fixture_queries() -> None:
    """输入：夹具里那几句查询。输出：无。

    预期行为：一条都不许撞禁用语——撞了就说明禁用语写得过宽，真跑时会把本来能用的
    数据在渲染期白拒掉。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for query in QUERIES.values():
            assert render.banned_echo(query, loc) == [], query


def test_need_words_are_not_banned() -> None:
    """输入：用「官方 / 最新 / 最全」说要求的查询。输出：无。

    预期行为：这几个词正是首要要求在自然语言里的说法，禁用语清单里不能有它们。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for word in ("官方", "最新", "最全", "official", "latest", "complete"):
            assert render.banned_echo(word, loc) == [], word


def test_placeholder_names_are_banned() -> None:
    """输入：把「虚构 / 虚拟」当成名字用的查询。输出：无。

    预期行为：真机 `batch-sourceselect-zh` 10 条里有 4 条把提示词那句「一律用虚构的名字」读成了
    字面名字（`虚构指标Y` / `虚构项目Alpha` / `虚拟技术`）——那是数据损坏，闸门必须拦住；
    而正常说法（`虚拟化平台`）与英文的 `virtualization` 不许误伤（`虚拟` 只收组合词）。
    """
    loc = locales.load("zh")
    for query in (
        "请提供虚构指标Y的官方文档",
        "请提供虚拟技术的最新公告",
        "虚构项目Alpha的完整清单",
    ):
        assert render.banned_echo(query, loc), query
    assert render.banned_echo("请提供虚拟化平台的最新官方文档", loc) == []
    assert render.banned_echo("official docs about virtualization", loc) == []


def test_meta_placeholders_render() -> None:
    """输入：两语言 meta。输出：无。预期行为：段落头、括号模板与行号模板都能格式化。"""
    for lang in locales.available():
        loc = locales.load(lang)
        assert "X" in locales.header(loc, "X")
        assert loc["meta"]["paren"].format(text="T").strip()
        assert str(loc["meta"]["row_ref"]).format(n=2).strip()
