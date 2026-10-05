"""语言文件：结构词元两边一致、标签不撞车、题族两边齐、判据写全。"""

from __future__ import annotations

from typing import Final

import pytest

from workflows.planstep import locales, parse, questions, render
from workflows.planstep.features import TASK_COUNT

LANGS: Final[tuple[str, ...]] = ("zh", "en")


def test_merged_words_cover_every_key() -> None:
    """输入：合并反查表。输出：无。

    预期行为：每个任务键在合并表里都至少有一个名字、指向的键都在 `TASK_KEYS` 里，状态词
    两种语言都在——漏一个就会让那种语言的产物整片读成「表外名字」。
    """
    merged = locales.merged()
    tasks = merged["task"]
    assert isinstance(tasks, dict)
    assert set(tasks.values()) == set(render.TASK_KEYS)
    assert merged["status"] == sorted({"已完成", "待开始", "done", "pending"})
    assert merged["deps_none"] == sorted({"无前置", "no deps"})
    assert merged["deps_prefix"] == sorted({"前置=", "deps="})


def test_status_words_do_not_collide_across_languages() -> None:
    """输入：两份语言文件。输出：无。

    预期行为：一种语言的「已完成」不许等于另一种语言的「待开始」——合并表按整串认状态，
    撞车会把「做完了」读成「没开始」。
    """
    zh = parse.tables_of(locales.load("zh"))
    en = parse.tables_of(locales.load("en"))
    assert not zh.done_words & en.pending_words
    assert not zh.pending_words & en.done_words
    assert not zh.done_words & zh.pending_words
    assert not en.done_words & en.pending_words


@pytest.mark.parametrize("lang", LANGS)
def test_task_names_are_distinct_and_complete(lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：五个任务名互不相同，键与 `TASK_KEYS` 一一对应。"""
    table = locales.load(lang)["task"]
    assert isinstance(table, dict)
    assert set(table) == set(render.TASK_KEYS), sorted(table)
    names = [str(value) for value in table.values()]
    assert len(set(names)) == len(names), names


@pytest.mark.parametrize("lang", LANGS)
def test_next_option_does_not_collide_with_a_task_name(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：「没有可开工的项」与五个任务名互不相同——组装器会拒掉重名选项，那是运行期
    才炸，这里提前钉住。
    """
    options = questions.next_options(locales.load(lang))
    assert set(options) == {f"t{index}" for index in range(1, 6)} | {"none"}
    assert len(set(options.values())) == len(options)


@pytest.mark.parametrize("lang", LANGS)
def test_families_have_phrasings_and_valid_criteria(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个题族至少一条问法；两个 score 族的等级表各 `TASK_COUNT + 1` 档；两个
    noul 族真 / 假两行——档位数与代码里那一套完全一致（下标就是答案）。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        assert loc["fam"][family]["q"], family
    assert len(loc["fam"]["remaining"]["levels"]) == TASK_COUNT + 1
    assert len(loc["fam"]["rounds"]["levels"]) == TASK_COUNT + 1
    assert set(loc["fam"]["unlocks"]["criteria"]) == {"true", "false"}


def test_structural_tokens_are_identical_across_languages() -> None:
    """输入：两份语言文件。输出：无。

    预期行为：格分隔符 / 前置分隔符 / 规则行前缀三种**结构词元**中英完全一致——反解只有
    一份合并表，各语言用不同的分隔符会让另一种语言的产物整片读不出来。
    """
    zh = locales.load("zh")["meta"]
    en = locales.load("en")["meta"]
    for key in ("cell_sep", "deps_sep", "rules_prefix"):
        assert zh[key] == en[key], key
    assert zh["deps_prefix"] != en["deps_prefix"], "前置前缀两边撞车，反解会认错"
