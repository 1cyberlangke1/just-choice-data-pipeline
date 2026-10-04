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


def test_bare_placeholder_names_are_banned() -> None:
    """输入：拿「X / 某」当名字用的查询（`5x`~`9x` 五个真机批逐批扫出来的那几种写法）。输出：无。

    预期行为：这些组合词全拦下；而**自造代号不许误伤**——`X12` / `X-Zone` / `星辰X` 正是提示词
    要求「自己造一个代号」时要的东西，拦下就等于把本来能用的数据在渲染期白拒掉。
    """
    loc = locales.load("zh")
    for query in (
        "提供关于X事件的完整清单及对照表",
        "给出某项目的完整汇总",
        "给出 X 品牌 2025 年发布的规范原文",
        "提供某某科技发布的安全测试标准",
    ):
        assert render.banned_echo(query, loc), query
    for query in (
        "搜索星行系统X12协议的完整清单",
        "检索《X-Zone无线网络协议》的官方原始标准全文",
        "在公开数据库检索星辰X最近发布的版本说明",
    ):
        assert render.banned_echo(query, loc) == [], query


def test_meta_placeholders_render() -> None:
    """输入：两语言 meta。输出：无。预期行为：段落头、括号模板与行号模板都能格式化。"""
    for lang in locales.available():
        loc = locales.load(lang)
        assert "X" in locales.header(loc, "X")
        assert loc["meta"]["paren"].format(text="T").strip()
        assert str(loc["meta"]["row_ref"]).format(n=2).strip()


def test_prompt_rules_do_not_restate_tier_markers() -> None:
    """输入：两语言文案。输出：无。

    预期行为：提示词规则里不许出现某一档的特征词（「官方文档」「完整清单」这类）。
    真机 `sourceselect-3x-zh` 24 条里 21 条、`sourceselect-4x-zh` 21 条全落在同一档：
    规则里那三条「要权威就写成官方文档…」把别档的词塞给了模型，它写 coverage 查询时顺手
    带上「官方文档里的完整清单」，反解就被读成 authority。特征词只能住在 `[feature.need.*]`，
    规则只讲「别同时提多项、别交代去哪儿找」。
    """
    markers = {
        "zh": (
            "官方文档",
            "标准原文",
            "原始公告",
            "最新版本",
            "最近的进展",
            "完整清单",
            "全量对照",
        ),
        "en": (
            "official docs",
            "the standard itself",
            "the original announcement",
            "the newest release",
            "the most recent progress",
            "a full list",
            "an end-to-end comparison",
        ),
    }
    assert set(markers) == set(locales.available()), "新增语言要在这里补上它的特征词"
    for lang, words in markers.items():
        loc = locales.load(lang)
        hints = " ".join(str(loc["feature"]["need"][value]["hint"]) for value in features.NEEDS)
        rules = " ".join(str(rule) for rule in loc["prompt"]["rules"])
        for word in words:
            assert word in hints, f"{lang}: {word!r} 不再是特征词了，测试该更新"
            assert word not in rules, f"{lang}: 提示词规则里漏了特征词 {word!r}"
