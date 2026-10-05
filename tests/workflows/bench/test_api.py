"""workflow 协议层的单元测试：切分配置、判别器装配、声明项与接线错误。"""

from __future__ import annotations

import pytest

from workflows.bench import judges, locales, parse
from workflows.bench import spec as sp
from workflows.bench.workflow import BenchWorkflow

WF = BenchWorkflow()


def test_split_names_and_configs() -> None:
    """输入：四个切分名。输出：无。

    预期行为：`test_ood` 用全开的留出配置，其余三个切分共用训练侧取值表（只换 `split`）。
    """
    assert WF.split_names() == ("train", "val", "test_iid", "test_ood")
    ood = WF.config_for("test_ood")
    assert ood.ood_only and set(ood.tables) == set(sp.TABLES)
    assert set(ood.layouts) == set(sp.LAYOUTS)
    assert set(ood.phrasings) == set(sp.PHRASINGS)
    for split in ("train", "val", "test_iid"):
        config = WF.config_for(split)
        assert config.split == split and not config.ood_only
        assert set(config.layouts) == set(sp.SEEN_LAYOUTS)
        assert config.phrasings == tuple(sorted(sp.SEEN_PHRASINGS))
        assert set(config.tables) == set(sp.TABLES)


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
    record = WF.reparse(text)
    assert isinstance(record, parse.ParsedRecord)
    assert record.request_id == built.request_id
    assert len(record.rows) == sp.TOTAL


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
