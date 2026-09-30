"""workflow 协议层的单元测试：切分配置、判别器装配、声明项与接线错误。"""

from __future__ import annotations


from workflows.skillpick import locales, parse
from workflows.skillpick import spec as sp
from workflows.skillpick.workflow import SkillPickWorkflow

WF = SkillPickWorkflow()


def test_split_names_and_configs() -> None:
    """输入：四个切分名。输出：无。

    预期行为：`test_ood` 用全开的留出配置，其余三个切分共用训练侧取值表（只换 `split`）。
    """
    assert WF.split_names() == ("train", "val", "test_iid", "test_ood")
    ood = WF.config_for("test_ood")
    assert ood.ood_only and set(ood.tags) == set(sp.TAGS)
    assert set(ood.layouts) == set(sp.LAYOUTS)
    assert set(ood.grants) == set(sp.GRANTS)
    assert set(ood.inuse_modes) == set(sp.INUSE_KINDS)
    assert set(ood.phrasings) == set(sp.PHRASINGS)
    for split in ("train", "val", "test_iid"):
        config = WF.config_for(split)
        assert config.split == split and not config.ood_only
        assert set(config.layouts) == set(sp.SEEN_LAYOUTS)
        assert config.phrasings == tuple(sorted(sp.SEEN_PHRASINGS))


def test_available_locales_and_declarations() -> None:
    """输入：无。输出：无。

    预期行为：语言名单来自语言目录；题型是 choice / noul，`needs` 为空（纯构造式）。
    """
    assert WF.available_locales() == locales.available()
    assert WF.kinds == frozenset({"choice", "noul"})
    assert WF.needs == frozenset()
    assert WF.name == "skillpick"
    assert WF.judges == ("roundtrip", "label")


def test_reparse_uses_the_merged_table() -> None:
    """输入：一条规格、英文渲染出来的文本。输出：无。

    预期行为：`reparse` 没有语言参数，靠合并表也能把英文文本读回同一份键。
    """
    case = next(iter(WF.build_cases(WF.config_for("train"))))
    built = sp.sample_spec(case, "train", 0, 0, None)
    text = WF.render(built, None, "en")
    request = WF.reparse(text)
    assert isinstance(request, parse.ParsedRequest)
    assert request.request_id == built.request_id
    assert len(request.rows) == len(sp.CATALOG)
