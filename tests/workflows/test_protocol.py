"""契约一致性：每个注册的 workflow 都真的实现了 `Workflow` 协议。

为什么要有：协议是 `runtime_checkable` 的，但**没人检查过**——少一个方法（实测踩到过
`case_label`）只会在管线跑到最后一步写 manifest 时才炸，而且是在「跑完一整批、烧完配额」之后。
这条把它提前到导入期，零 token。

只查「方法在不在、声明项对不对、公开接口能不能拿第一个 case 走通」——不跑生成、不碰网络。
"""

from __future__ import annotations

import pytest

from jc.workflow import Workflow, discover, load

#: 已注册的 workflow 名（导入即注册）。名单为空说明自注册坏了，下面几条会一起报出来。
WORKFLOWS = discover()


def test_discover_finds_every_workflow() -> None:
    """输入：无。输出：无。预期行为：注册表非空，且每个名字都能加载。"""
    assert len(WORKFLOWS) > 20
    for name in WORKFLOWS:
        assert load(name).name == name


@pytest.mark.parametrize("name", WORKFLOWS)
def test_workflow_satisfies_the_protocol(name: str) -> None:
    """输入：每个注册的 workflow 名。输出：无。预期行为：协议要求的方法与声明项一个不缺。"""
    instance = load(name)
    assert isinstance(instance, Workflow), f"{name} 没有实现完整的 Workflow 协议"
    assert instance.version
    assert isinstance(instance.kinds, frozenset) and instance.kinds
    assert isinstance(instance.needs, frozenset)
    assert isinstance(instance.judges, tuple) and instance.judges


@pytest.mark.parametrize("name", WORKFLOWS)
def test_every_split_yields_cases_and_readable_labels(name: str) -> None:
    """输入：每个 workflow × 它的每个切分。输出：无。

    预期行为：每个切分都有 case、case 有可读标签、维度表非空——这几条是管线写 manifest 与统计
    报告时一定会用到的（`case_label` 少一个就会在这里当场炸，而不是跑完整批之后）。
    """
    instance = load(name)
    assert instance.dimension_titles()
    for split in instance.split_names():
        cases = instance.build_cases(instance.config_for(split))
        assert cases, f"{name}/{split} 一个 case 都没有"
        case = cases[0]
        assert instance.case_label(case)
        assert instance.stats_dimensions(case)


@pytest.mark.parametrize("name", WORKFLOWS)
def test_locales_are_files_on_disk(name: str) -> None:
    """输入：每个 workflow。输出：无。预期行为：声明的语言都真的读得出来。"""
    instance = load(name)
    available = instance.available_locales()
    assert available
    assert sorted(available) == available
