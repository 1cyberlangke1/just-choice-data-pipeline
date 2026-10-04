"""receiptcat 语言文件的契约：两种语言必须定义同一批键，占位符必须填得齐。

输入：`workflows/receiptcat/locales/*.toml`。输出：无（断言）。
预期行为：缺一个键就会在渲染时 `KeyError`（或者更糟：拿到半句话的提示词），所以这里把
「同一批键」变成机器可查——跨 workflow 的那三条不变量在 `tests/core/test_locale_templates.py`，
这里只管本 workflow 自己的形状。
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Final

import pytest

from workflows.receiptcat import features, locales, questions, render

#: 语言文件目录。
LOCALE_DIR: Final[Path] = Path("workflows/receiptcat/locales")

#: 提示词表里允许出现的占位符（多了就是模板写歪，少了就是槽没填）。
PROMPT_SLOTS: Final[frozenset[str]] = frozenset(
    {"min", "max", "language", "category", "word", "amount", "names", "decoy"}
)

#: 反解提示词允许出现的占位符（目前一个都没有）。
VERIFY_SLOTS: Final[frozenset[str]] = frozenset()


def load(lang: str) -> dict[str, object]:
    """输入：语言代码。输出：TOML 字典。预期行为：读不到就抛（测试不该静默跳过）。"""
    with (LOCALE_DIR / f"{lang}.toml").open("rb") as handle:
        return tomllib.load(handle)


#: 两种语言（按文件名）。
LANGS: Final[tuple[str, ...]] = tuple(locales.available())


def test_two_languages_are_available() -> None:
    """输入：无。输出：无。预期行为：至少 zh / en 两种，且顺序稳定。"""
    assert LANGS == ("en", "zh")


@pytest.mark.parametrize("lang", LANGS)
def test_file_declares_its_own_language(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：`lang` 字段与文件名一致、`language_name` 非空。"""
    data = load(lang)
    assert data["lang"] == lang
    assert isinstance(data["language_name"], str) and str(data["language_name"]).strip()


def test_both_languages_define_the_same_keys() -> None:
    """输入：无。输出：无。

    预期行为：顶层键、`[state]`、四个类别的名字、关键词表、诱饵表、档位表与题族都逐层一致——
    少一个键就会在渲染时 `KeyError`，多一个键说明有一边没跟上改动。
    """
    tables = {lang: load(lang) for lang in LANGS}
    reference = tables[LANGS[0]]
    for lang in LANGS[1:]:
        assert set(tables[lang]) == set(reference), f"{lang} 的顶层键与 {LANGS[0]} 不一致"
        for table in ("state", "category", "keyword", "decoy", "tier", "fam", "prompt", "verify"):
            assert set(_table(tables[lang], table)) == set(_table(reference, table)), (
                f"{lang} 的 [{table}] 键与 {LANGS[0]} 不一致"
            )


def _table(data: dict[str, object], name: str) -> dict[str, object]:
    """输入：语言字典、表名。输出：那张表。预期行为：不是表就当场报错。"""
    node = data.get(name)
    assert isinstance(node, dict), f"缺 [{name}] 表"
    return node


@pytest.mark.parametrize("lang", LANGS)
def test_state_marks_are_distinguishable(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四个抬头标记两两**不互为前缀**——解析是「逐行看开头」，一个标记是另一个的前缀
    时，同一行会被两块抢走（这条坑在别家踩过，见范式 §9.22）。
    """
    state = _table(load(lang), "state")
    heads = [str(state[key]) for key in ("cat_mark", "cap_mark", "tier_mark", "content_mark")]
    for left in heads:
        for right in heads:
            if left != right:
                assert not left.startswith(right), f"{lang}: {right} 是 {left} 的前缀"


@pytest.mark.parametrize("lang", LANGS)
def test_category_and_keyword_keys_match_the_code(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：类别 / 关键词 / 诱饵三张表的键都正好是 `CATEGORY_IDS`。"""
    data = load(lang)
    for table in ("category", "keyword", "decoy"):
        assert list(_table(data, table)) == list(features.CATEGORY_IDS)


@pytest.mark.parametrize("lang", LANGS)
def test_tier_keys_match_the_code(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：档位表的键正好是 `TIERS`，每个都有非空的 name / rule。"""
    table = _table(load(lang), "tier")
    assert list(table) == list(features.TIERS)
    for key, entry in table.items():
        assert isinstance(entry, dict), f"档位 {key} 不是表"
        for field in ("name", "rule"):
            assert isinstance(entry.get(field), str) and entry[field].strip(), f"档位 {key} 缺 {field}"


@pytest.mark.parametrize("lang", LANGS)
def test_family_keys_match_the_code(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：题族表正好是 `questions.FAMILIES`，每个都有非空问法列表。"""
    table = _table(load(lang), "fam")
    assert list(table) == list(questions.FAMILIES)
    for key, entry in table.items():
        assert isinstance(entry, dict), f"题族 {key} 不是表"
        templates = entry.get("q")
        assert isinstance(templates, list) and templates, f"题族 {key} 缺问法模板"
        assert all(isinstance(text, str) and text.strip() for text in templates)


@pytest.mark.parametrize("lang", LANGS)
def test_noul_families_have_both_criteria(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：两个 noul 题族都有 true / false 两句判据文案。"""
    table = _table(load(lang), "fam")
    for family in ("over", "near"):
        entry = table[family]
        assert isinstance(entry, dict)
        for key in ("true", "false"):
            assert isinstance(entry.get(key), str) and entry[key].strip(), f"{family} 缺 {key}"


@pytest.mark.parametrize("lang", LANGS)
def test_prompt_slots_are_exactly_the_ones_we_fill(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：提示词模板里的占位符集合正好是 `PROMPT_SLOTS`——多了会在渲染时 `KeyError`，
    少了说明某个槽白填（模型拿到的是没填的半句话）。
    """
    found: set[str] = set()
    for text in _strings(_table(load(lang), "prompt")):
        found.update(re.findall(r"\{([a-z_]+)\}", text))
    assert found == PROMPT_SLOTS


@pytest.mark.parametrize("lang", LANGS)
def test_verify_prompt_has_no_slots(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：反解提示词没有占位符（它只插 state）。"""
    found: set[str] = set()
    for text in _strings(_table(load(lang), "verify")):
        found.update(re.findall(r"\{([a-z_]+)\}", text))
    assert found == VERIFY_SLOTS


def _strings(node: object) -> list[str]:
    """输入：TOML 解析出来的任意节点。输出：它下面所有字符串（递归展开表与数组）。"""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        out: list[str] = []
        for value in node.values():
            out.extend(_strings(value))
        return out
    if isinstance(node, list):
        out: list[str] = []
        for value in node:
            out.extend(_strings(value))
        return out
    return []


@pytest.mark.parametrize("lang", LANGS)
def test_category_names_and_keyword_lists_are_unique(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四个类别名两两不同、四张关键词清单两两不同——`_choice` 拿「描述重复」当构造
    bug 报，这里提前把它钉住，免得等到出题时才炸。
    """
    loc = locales.load(lang)
    names = [render.category_name(loc, cid) for cid in features.CATEGORY_IDS]
    assert len(set(names)) == len(names)
    lists = [questions.category_text(loc, cid) for cid in features.CATEGORY_IDS]
    assert len(set(lists)) == len(lists)

