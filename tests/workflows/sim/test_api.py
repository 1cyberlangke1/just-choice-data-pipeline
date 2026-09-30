"""workflow 协议层的单元测试：切分配置、判别器装配、声明项与接线错误。"""

from __future__ import annotations


from workflows.sim import locales, parse
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


def test_available_locales_and_declarations() -> None:
    """输入：无。输出：无。

    预期行为：语言名单来自语言目录；题型是 choice / noul / score，`needs` 为空（纯构造式）。
    """
    assert WF.available_locales() == locales.available()
    assert WF.kinds == frozenset({"choice", "noul", "score"})
    assert WF.needs == frozenset()
    assert WF.name == "sim"
    assert WF.judges == ("roundtrip", "label")


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
