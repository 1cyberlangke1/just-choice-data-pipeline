"""文案表的不变量：这些不是「文案好不好」，而是「答案对不对」的前提。"""

from __future__ import annotations

import pytest

from workflows.lint import locales, render
from workflows.lint.spec import GENRES, LONG_LIMIT, RULES, SIZE

LANGS = locales.available()


def test_available_locales_are_sorted_and_nonempty() -> None:
    """输入：无。输出：无。预期行为：至少两种语言，顺序稳定。"""
    assert LANGS == sorted(LANGS)
    assert {"zh", "en"} <= set(LANGS)


def test_unknown_locale_reports_available_ones() -> None:
    """输入：不存在的语言。输出：无。预期行为：报错并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="没有语言文件"):
        locales.load("xx")


def test_pick_wraps_around() -> None:
    """输入：长度为 2 的池、编号 3。输出：无。预期行为：回绕到第 1 个（不报错）。"""
    assert locales.pick(["a", "b"], 3) == "b"
    assert locales.pick(["a", "b"], 4) == "a"


def test_merged_carries_every_rule_phrase_from_every_locale() -> None:
    """输入：无。输出：无。

    预期行为：合并表把各语言的规则短语取并集（`reparse` 没有语言参数，只能靠它）。
    """
    merged = locales.merged()
    for name in RULES:
        for lang in LANGS:
            assert locales.load(lang)["rule"][name] in merged["rule"][name]


@pytest.mark.parametrize("lang", LANGS)
def test_locales_share_the_same_shape(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：语言文件的**键集合完全一致**——少一条规则的短语会让那种语言直接跑不动，
    少一格题面会让 `pick` 回绕到别的问法上（答案照样发出去，只是题变得莫名其妙）。
    """
    base = locales.load("zh")
    other = locales.load(lang)
    assert set(other["rule"]) == set(base["rule"])
    assert len(other["q"]) == len(base["q"])
    assert set(other["lead"]) == set(base["lead"])
    assert set(other["bg"]) == set(base["bg"])
    for genre in GENRES:
        assert len(other["bg"][genre]) == len(base["bg"][genre]) == SIZE
    assert set(other["hit"]) == set(base["hit"])
    for rule in RULES:
        assert set(other["hit"][rule]) == set(base["hit"][rule])
        assert len(other["hit"][rule]["slots"]) == SIZE
    for family in ("line", "count", "gate", "spot"):
        assert set(other["fam"][family]) == set(base["fam"][family])
    assert len(other["ts"]) == len(base["ts"]) == SIZE


@pytest.mark.parametrize("lang", LANGS)
def test_rule_phrases_are_mutually_exclusive(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：规则短语两两不互为子串——否则一段文本会同时命中两条规则，`rule_of` 直接
    报「命中 2 条」，整个 workflow 一条样本都出不来。
    """
    phrases = locales.load(lang)["rule"]
    for first in RULES:
        for second in RULES:
            if first != second:
                assert phrases[first] not in phrases[second]


@pytest.mark.parametrize("lang", LANGS)
def test_every_hit_line_really_matches_its_rule(lang: str) -> None:
    """输入：每种语言的命中行池。输出：无。

    预期行为：8 个槽**每一个**都命中它那条规则——这是「计划说命中、文本也命中」的唯一保证；
    少一个槽命中，`count` / `gate` / `spot` 的答案就整批错。
    """
    from workflows.lint.spec import matches

    loc = locales.load(lang)
    for rule in RULES:
        for genre in GENRES:
            lines = [render.hit_line(loc, rule, genre, slot) for slot in range(SIZE)]
            for slot, line in enumerate(lines):
                assert matches(rule, line), f"{lang}/{rule}/{genre}/s{slot} 不命中：{line!r}"
            assert len(set(lines)) == SIZE, f"{lang}/{rule}/{genre} 命中行有重复"


@pytest.mark.parametrize("lang", LANGS)
def test_no_background_line_matches_any_rule(lang: str) -> None:
    """输入：每种语言的背景行池。输出：无。

    预期行为：背景行**一条都不许**命中任何规则，而且都不超过超长行阈值——背景行串味会
    让「命中行数」当场变成错的数字（而且是静默错的）。
    """
    from workflows.lint.spec import matches

    loc = locales.load(lang)
    for genre in GENRES:
        lines = [render.bg_line(loc, genre, slot) for slot in range(SIZE)]
        for slot, line in enumerate(lines):
            hits = [rule for rule in RULES if matches(rule, line)]
            assert not hits, f"{lang}/bg/{genre}/s{slot} 命中 {hits}：{line!r}"
            assert len(line) <= LONG_LIMIT, f"{lang}/bg/{genre}/s{slot} 超过阈值：{line!r}"
        assert len(set(lines)) == SIZE, f"{lang}/bg/{genre} 背景行有重复"


@pytest.mark.parametrize("lang", LANGS)
def test_long_hit_lines_exceed_the_threshold(lang: str) -> None:
    """输入：每种语言的 `long` 命中行池。输出：无。

    预期行为：**每一种体裁**的 8 行都超过阈值——体裁模板的抬头长度不同（文档体裁没有抬头），
    抬头长的那两种会把短 payload 抬过线，只有文档体裁会露馅。
    """
    loc = locales.load(lang)
    for genre in GENRES:
        for slot in range(SIZE):
            line = render.hit_line(loc, "long", genre, slot)
            assert len(line) > LONG_LIMIT, f"{lang}/long/{genre}/s{slot} 太短：{len(line)}"


@pytest.mark.parametrize("lang", LANGS)
def test_no_line_carries_braces_or_a_rule_phrase(lang: str) -> None:
    """输入：每种语言的全部行。输出：无。

    预期行为：① 行里没有 `{` / `}`——题面里的行原文要经过严格填槽，带花括号会误报成
    没填完的占位符；② 行里没有规则短语——否则 `rule_of` 会命中两条规则。
    """
    loc = locales.load(lang)
    phrases = list(loc["rule"].values())
    for rule in RULES:
        for genre in GENRES:
            for slot in range(SIZE):
                line = render.hit_line(loc, rule, genre, slot)
                assert "{" not in line and "}" not in line, line
                assert not any(phrase in line for phrase in phrases), line
    for genre in GENRES:
        for slot in range(SIZE):
            line = render.bg_line(loc, genre, slot)
            assert "{" not in line and "}" not in line, line
            assert not any(phrase in line for phrase in phrases), line


@pytest.mark.parametrize("lang", LANGS)
def test_gate_instructions_match_the_code_thresholds(lang: str) -> None:
    """输入：每种语言的 `gate` 题面。输出：无。

    预期行为：题面里写明的处置办法与 `spec.gate_of` 的分档**一致**——判据写在题面里
    （范式 §9.8），所以改边界时必须同时改文案，这条会先红。
    """
    loc = locales.load(lang)
    for question in loc["fam"]["gate"]["q"]:
        text = str(question)
        assert "0~2" in text
        assert "3~5" in text
        assert "6" in text


@pytest.mark.parametrize("lang", LANGS)
def test_ts_slots_are_all_distinct(lang: str) -> None:
    """输入：每种语言的时刻表。输出：无。

    预期行为：8 个时刻互不相同——命中行与背景行共用这张表，时刻重复会让片段里出现
    两行完全一样的日志行（`spot` 的选项描述会撞车）。
    """
    stamps = locales.load(lang)["ts"]
    assert len(set(stamps)) == SIZE


@pytest.mark.parametrize("lang", LANGS)
def test_background_log_lines_use_the_shared_timestamps(lang: str) -> None:
    """输入：每种语言的日志体裁背景行。输出：无。

    预期行为：日志背景行的时刻取自 `[ts]`——命中行的时刻也取自同一张表，所以「带不带
    时刻」不携带命中信息（这正是当初把时刻抽成公共槽的原因）。
    """
    loc = locales.load(lang)
    stamps = set(loc["ts"])
    for slot in range(SIZE):
        line = render.bg_line(loc, "log", slot)
        assert any(stamp in line for stamp in stamps), line
