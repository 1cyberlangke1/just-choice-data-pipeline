"""workflow 的公开接口：切分名单、判别器装配、语言名单，以及两处「接线错了要炸」。"""

from __future__ import annotations

import pytest

from workflows.context import judges, locales, parse
from workflows.context import spec as sp
from workflows.context.workflow import ContextWorkflow

WF = ContextWorkflow()


def test_split_names_and_config_for() -> None:
    """输入：四个切分名。输出：无。

    预期行为：`test_ood` 用全开的留出配置，其余三个切分共用训练侧取值表（只换 `split`）。
    """
    assert WF.split_names() == ("train", "val", "test_iid", "test_ood")
    ood = WF.config_for("test_ood")
    assert ood.ood_only and set(ood.keep_forms) == set(sp.KEEP_FORMS)
    assert set(ood.extras) == set(sp.ACTIONS) and set(ood.orders) == set(sp.ORDERS)
    for split in ("train", "val", "test_iid"):
        config = WF.config_for(split)
        assert config.split == split and not config.ood_only
        assert set(config.keep_forms) == set(sp.SEEN_KEEP_FORMS)
        assert config.phrasings == tuple(sorted(sp.SEEN_PHRASINGS))


def test_config_for_rejects_an_unknown_split() -> None:
    """输入：不存在的切分名。输出：无。预期行为：报人话并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        WF.config_for("nope")


def test_make_judges_builds_both_and_rejects_unknown_names() -> None:
    """输入：两个判别器名 / 一个不存在的名字。输出：无。预期行为：都报人话。"""
    built = WF.make_judges(("roundtrip", "label"), "zh")
    assert [judge.name for judge in built] == ["roundtrip", "label"]
    assert all(isinstance(judge, (judges.RoundtripJudge, judges.LabelJudge)) for judge in built)
    with pytest.raises(KeyError, match="未知判别器"):
        WF.make_judges(("nope",), "zh")


def test_available_locales_and_declarations() -> None:
    """输入：无。输出：无。

    预期行为：语言名单来自语言目录；题型是 choice / noul，`needs` 为空（纯构造式）。
    """
    assert WF.available_locales() == locales.available()
    assert WF.kinds == frozenset({"choice", "noul"})
    assert WF.needs == frozenset()
    assert WF.name == "context"


def test_bind_backend_is_a_wiring_error() -> None:
    """输入：一个假的后端。输出：无。

    预期行为：纯构造式的 workflow 根本不该被绑定后端，收到就报错而不是静默吞掉。
    """
    with pytest.raises(AssertionError, match="纯构造式"):
        WF.bind_backend(object())


def test_field_value_prefers_the_longest_label() -> None:
    """输入：两个互为前缀的候选标签。输出：无。

    预期行为：取最长的那个——否则「会话」会把「会话号」那一格提前吃掉。
    """
    assert parse._field_value("会话号：S-1", ("会话", "会话号")) == "S-1"
    assert parse._field_value("会话：S-1", ("会话", "会话号")) == "S-1"


def test_field_value_needs_a_separator_right_after_the_label() -> None:
    """输入：以标签开头但没有分隔符的一行。输出：无。

    预期行为：不算这一格——标题行确实可能以某个标签开头，收紧才不会误认。
    """
    assert parse._field_value("会话号错了", ("会话号",)) is None
    assert parse._field_value("今天天气不错。", ("会话号",)) is None


def test_merged_rejects_mismatched_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言用不同的分隔符。输出：无。

    预期行为：报人话——合并表只按一个分隔符切抬头行，不一致就没法同时读两门语言。
    """
    zh = locales.load("zh")
    en = dict(locales.load("en"))
    en["meta"] = dict(en["meta"])
    en["meta"]["block_sep"] = " | "
    monkeypatch.setattr(locales, "available", lambda: ["zh", "en"])
    monkeypatch.setattr(locales, "load", lambda lang: {"zh": zh, "en": en}[lang])
    with pytest.raises(ValueError, match="分隔符必须一致"):
        parse.merged()


def test_merged_rejects_colliding_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言的会话号标签写成一样。输出：无。预期行为：报人话（合并表有歧义）。"""
    zh = locales.load("zh")
    en = dict(locales.load("en"))
    en["meta"] = dict(en["meta"])
    en["meta"]["session_label"] = zh["meta"]["session_label"]
    monkeypatch.setattr(locales, "available", lambda: ["zh", "en"])
    monkeypatch.setattr(locales, "load", lambda lang: {"zh": zh, "en": en}[lang])
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_rejects_colliding_vocabularies(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言把同一个类型写成同一个词。输出：无。预期行为：报人话。"""
    zh = locales.load("zh")
    en = dict(locales.load("en"))
    en["kind"] = dict(en["kind"])
    en["kind"]["ask"] = zh["kind"]["ask"]
    monkeypatch.setattr(locales, "available", lambda: ["zh", "en"])
    monkeypatch.setattr(locales, "load", lambda lang: {"zh": zh, "en": en}[lang])
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()
