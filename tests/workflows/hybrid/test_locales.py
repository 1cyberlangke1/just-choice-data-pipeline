"""语言文件的契约测试：词表、模板、问法、等级表、合并标签。

这组测试是**锁**：`spec` 的词表容量常量、`render` 的模板占位符、`questions` 的题族名，
只要有一处漂移就立刻红灯。
"""

from __future__ import annotations

import pytest

from workflows.hybrid import locales, questions, render
from workflows.hybrid import spec as sp

LOCALES = locales.available()


def test_both_languages_are_present() -> None:
    """输入：无。输出：语言列表。预期行为：中英各一份（管线默认 zh，但 en 也得能跑）。"""
    assert LOCALES == ["en", "zh"]


@pytest.mark.parametrize("lang", LOCALES)
def test_word_pools_are_long_enough_and_unique(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：spec 里存的是词表**下标**，所以会议室 / 主持人词表不能短于 `ROOM_POOL` /
    `LEAD_POOL`；词条还要两两不同——同名会议室会让 `room_of` 有两个正确答案。
    """
    loc = locales.load(lang)
    for group, size in (("room", sp.ROOM_POOL), ("lead", sp.LEAD_POOL)):
        items = loc["word"][group]["items"]
        assert len(items) >= size, group
        assert len(set(items)) == len(items), group


@pytest.mark.parametrize("lang", LOCALES)
def test_weekday_and_duration_tables_match_the_code(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：星期名恰好 7 条且顺序与 `WEEKDAY_KEYS` 一致（下标 = 星期几 - 1）；
    时长文案恰好 4 条且覆盖 `DUR_RANGES` 的全部取值（下标 = 时长 - 1）。
    """
    loc = locales.load(lang)
    names = loc["word"]["weekday"]["names"]
    assert len(names) == len(questions.WEEKDAY_KEYS) == 7
    assert len(set(names)) == 7
    durations = loc["word"]["duration"]["items"]
    covered = {hours for lo, hi in sp.DUR_RANGES.values() for hours in range(lo, hi + 1)}
    assert len(durations) == max(covered) == 4
    for hours in covered:
        assert render.duration_text(hours, loc)


@pytest.mark.parametrize("lang", LOCALES)
def test_templates_carry_their_placeholders(lang: str) -> None:
    """输入：语言文件。输出：无。预期行为：模板占位符齐全（少一个渲染就炸）。"""
    loc = locales.load(lang)
    assert "{h:02d}" in loc["time"]["tpl"]
    for part in ("{y:04d}", "{m:02d}", "{d:02d}"):
        assert part in loc["date"]["tpl"]
    for key in ("start", "end"):
        assert "{" + key + "}" in loc["meta"]["span"]
    for key in ("n", "span", "sep", "room", "lead"):
        assert "{" + key + "}" in loc["bullet"]["line"]
    assert set(loc["column"]) == {"n", "start", "end", "room", "lead", "title", "agenda"}


@pytest.mark.parametrize("lang", LOCALES)
def test_prompt_rules_carry_count_and_language(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：每条规则都能被 `count` / `language` 格式化，且至少有一条用到场数——
    场数不说清楚，模型就不知道该回几条（少一条就整条样本被拒）。
    """
    loc = locales.load(lang)
    assert loc["language_name"]
    rules = loc["prompt"]["rules"]
    assert rules and any("{count}" in rule for rule in rules)
    for rule in rules:
        rule.format(count=3, language=loc["language_name"])


@pytest.mark.parametrize("lang", LOCALES)
def test_ask_covers_every_choice_family_and_texts_are_unique(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：每个 choice 题族都有问法，且**所有问法文本两两不同**——
    `locate` 是拿题面反查题族的，两句话撞车就会把题认错（判别器随之失效）。
    带「第 {n} 场」的要按场数展开后再比。
    """
    loc = locales.load(lang)
    assert set(loc["ask"]) == set(questions.CHOICE_FAMILIES)
    texts: list[str] = []
    for family, entry in loc["ask"].items():
        assert entry["q"], family
        if family in questions.SLOT_FAMILIES:
            texts.extend(
                text.format(n=n) for text in entry["q"] for n in range(1, 5)
            )
        else:
            texts.extend(entry["q"])
    fam_texts = [text for family in loc["fam"].values() for text in family["q"]]
    assert len(set(texts)) == len(texts)
    assert len(set(fam_texts)) == len(fam_texts)
    assert not (set(texts) & set(fam_texts))


@pytest.mark.parametrize("lang", LOCALES)
def test_family_texts_and_score_levels(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：`fam` 只覆盖 noul / score 题族；noul 的 criteria 恰好 true / false；
    score 的等级表是 `BUSY_CAP + 1` 档，且每档都点名自己的等级数字。
    """
    loc = locales.load(lang)
    assert set(loc["fam"]) == set(questions.NOUL_FAMILIES) | set(questions.SCORE_FAMILIES)
    for family in questions.NOUL_FAMILIES:
        assert set(loc["fam"][family]["criteria"]) == {"true", "false"}
    levels = loc["fam"]["busy_level"]["levels"]
    assert len(levels) == sp.BUSY_CAP + 1
    for index, level in enumerate(levels):
        assert level.startswith(str(index))


@pytest.mark.parametrize("lang", LOCALES)
def test_merged_labels_do_not_collide_across_languages(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：本语言的标签不出现在别的语言里——`Workflow.reparse` 用合并标签表，
    两个语言共用一个标签就会指向两个槽，语言无关的反解立刻不可靠。
    """
    mine = {
        locales.load(lang)["bullet"][f"{slot}_label"] for slot in ("title", "agenda")
    }
    for other in LOCALES:
        if other != lang:
            theirs = {
                locales.load(other)["bullet"][f"{slot}_label"] for slot in ("title", "agenda")
            }
            assert not (mine & theirs)
    assert set(locales.merged_labels()) == {
        locales.load(name)["bullet"][f"{slot}_label"]
        for name in LOCALES
        for slot in ("title", "agenda")
    }


def test_room_and_weekday_words_are_merged_over_languages() -> None:
    """输入：无。输出：两个词集合。

    预期行为：文本卫生检查用的是**各语言合并**的全集——一份中文日程里冒出英文会议室名
    同样是夹带骨架事实。
    """
    rooms = locales.room_words()
    weekdays = locales.weekday_words()
    for lang in LOCALES:
        loc = locales.load(lang)
        assert set(loc["word"]["room"]["items"]) <= rooms
        assert set(loc["word"]["weekday"]["names"]) <= weekdays


@pytest.mark.parametrize("lang", LOCALES)
def test_meta_punctuation_comes_from_the_locale(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：冒号与分隔符由语言文件给；英文里不能有中日韩标点——
    否则英文日程会带着「：」，看着就不像英文数据。
    """
    loc = locales.load(lang)
    assert loc["meta"]["colon"] and loc["meta"]["sep"]
    if lang == "en":
        for key in ("colon", "sep"):
            assert not (set("：；。") & set(loc["meta"][key]))


def test_missing_language_file_raises() -> None:
    """输入：不存在的语言代码。输出：FileNotFoundError。预期行为：报错并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="可用"):
        locales.load("fr")


def test_pick_wraps_and_word_of_rejects_out_of_range() -> None:
    """输入：越界问法编号 / 越界词表下标。输出：回绕 / IndexError。

    预期行为：问法池可能比列表长，所以 `pick` 回绕；词表下标越界是数据错位，必须报错。
    """
    texts = locales.load("zh")["ask"]["duration_of"]["q"]
    assert locales.pick(texts, len(texts)) == texts[0]
    assert locales.pick(texts, len(texts) + 1) == texts[1]
    with pytest.raises(IndexError, match="取不到下标"):
        locales.word_of(locales.load("zh"), "room", 99)


def test_merged_labels_rejects_a_colliding_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两个语言共用一个标签却指向不同槽。输出：ValueError。

    预期行为：合并标签表一旦撞车就报错——语言无关的反解全靠它，
    撞车会让 `reparse` 把「议题」读成「标题」而没人发现。
    """
    fake = {
        "en": {"bullet": {"title_label": "标题", "agenda_label": "议题"}},
        "zh": {"bullet": {"title_label": "议程", "agenda_label": "标题"}},
    }
    locales.merged_labels.cache_clear()
    monkeypatch.setattr(locales, "available", lambda: ["en", "zh"])
    monkeypatch.setattr(locales, "load", lambda lang: fake[lang])
    try:
        with pytest.raises(ValueError, match="指的不是同一件事"):
            locales.merged_labels()
    finally:
        locales.merged_labels.cache_clear()
