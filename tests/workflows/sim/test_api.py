"""workflow 协议层的单元测试：切分配置、判别器装配、声明项与接线错误。"""

from __future__ import annotations

import pytest

from workflows.sim import judges, locales, parse
from workflows.sim import spec as sp
from workflows.sim.workflow import SimWorkflow

WF = SimWorkflow()


def test_split_names_and_configs() -> None:
    """输入：四个切分名。输出：无。

    预期行为：`test_ood` 用全开的留出配置，其余三个切分共用训练侧取值表（只换 `split`）。
    """
    assert WF.split_names() == ("train", "val", "test_iid", "test_ood")
    ood = WF.config_for("test_ood")
    assert ood.ood_only and set(ood.boards) == set(sp.BOARDS)
    assert set(ood.layouts) == set(sp.LAYOUTS)
    assert set(ood.phrasings) == set(sp.PHRASINGS)
    for split in ("train", "val", "test_iid"):
        config = WF.config_for(split)
        assert config.split == split and not config.ood_only
        assert set(config.layouts) == set(sp.SEEN_LAYOUTS)
        assert config.phrasings == tuple(sorted(sp.SEEN_PHRASINGS))
        assert set(config.boards) == set(sp.BOARDS)


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

    预期行为：语言名单来自语言目录；题型是 choice / noul / score，`needs` 为空（纯构造式）。
    """
    assert WF.available_locales() == locales.available()
    assert WF.kinds == frozenset({"choice", "noul", "score"})
    assert WF.needs == frozenset()
    assert WF.name == "sim"
    assert WF.judges == ("roundtrip", "label")


def test_bind_backend_is_a_wiring_error() -> None:
    """输入：一个假的后端。输出：无。

    预期行为：纯构造式的 workflow 根本不该被绑定后端，收到就报错而不是静默吞掉。
    """
    with pytest.raises(AssertionError, match="纯构造式"):
        WF.bind_backend(object())


def test_case_label_and_stats_dimensions() -> None:
    """输入：一个 case。输出：无。预期行为：标签与维度取值都从公开接口来。"""
    case = next(iter(WF.build_cases(WF.config_for("train"))))
    assert WF.case_label(case) == case.describe()
    assert set(WF.stats_dimensions(case)) == set(WF.dimension_titles())
    assert all(title for title in WF.dimension_titles().values())


def test_reparse_uses_the_merged_table() -> None:
    """输入：一份规格、英文渲染出来的文本。输出：无。

    预期行为：`reparse` 没有语言参数，靠合并表也能把英文文本读回同一份键。
    """
    case = next(iter(WF.build_cases(WF.config_for("train"))))
    built = sp.sample_spec(case, "train", 0, 0, None)
    text = WF.render(built, None, "en")
    board = WF.reparse(text)
    assert isinstance(board, parse.ParsedBoard)
    assert board.request_id == built.request_id
    assert len(board.rows) == len(sp.ACTIONS)


def test_render_and_ask_ignore_the_rng() -> None:
    """输入：同一份规格、两次调用（RNG 不同）。输出：无。

    预期行为：渲染与出题都不掷骰子——这个 workflow 的每个自由度都已经在 case 里。
    """
    case = next(iter(WF.build_cases(WF.config_for("train"))))
    built = sp.sample_spec(case, "train", 0, 0, None)
    assert WF.render(built, None, "zh") == WF.render(built, None, "zh")
    first = WF.ask(built, None, "zh")
    second = WF.ask(built, None, "zh")
    assert [(item.family, item.answer) for item in first] == [
        (item.family, item.answer) for item in second
    ]
