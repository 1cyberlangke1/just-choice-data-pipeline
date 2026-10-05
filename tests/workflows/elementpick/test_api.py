"""workflow 协议层的单元测试：切分配置、判别器装配、声明项与接线错误。"""

from __future__ import annotations

import pytest

from workflows.elementpick import judges, locales, parse
from workflows.elementpick import spec as sp
from workflows.elementpick.workflow import ElementPickWorkflow

WF = ElementPickWorkflow()


def test_split_names_and_configs() -> None:
    """输入：四个切分名。输出：无。

    预期行为：`test_ood` 用全开的留出配置，其余三个切分共用训练侧取值表（只换 `split`）。
    """
    assert WF.split_names() == ("train", "val", "test_iid", "test_ood")
    ood = WF.config_for("test_ood")
    assert ood.ood_only and set(ood.labels) == set(sp.LABELS)
    assert set(ood.layouts) == set(sp.LAYOUTS)
    assert set(ood.shapes) == set(sp.SHAPES)
    assert set(ood.extras) == set(sp.EXTRA_COUNTS)
    assert set(ood.shadows) == set(sp.SHADOWS)
    assert set(ood.phrasings) == set(sp.PHRASINGS)
    for split in ("train", "val", "test_iid"):
        config = WF.config_for(split)
        assert config.split == split and not config.ood_only
        assert set(config.layouts) == set(sp.SEEN_LAYOUTS)
        assert config.phrasings == tuple(sorted(sp.SEEN_PHRASINGS))


def test_reparse_uses_the_merged_table() -> None:
    """输入：一条规格、英文渲染出来的文本。输出：无。

    预期行为：`reparse` 没有语言参数，靠合并表也能把英文文本读回同一份键。
    """
    case = next(iter(WF.build_cases(WF.config_for("train"))))
    built = sp.sample_spec(case, "train", 0, 0, None)
    text = WF.render(built, None, "en")
    screen = WF.reparse(text)
    assert isinstance(screen, parse.ParsedScreen)
    assert screen.screen_id == built.screen_id
    assert len(screen.rows) == sp.SIZE
