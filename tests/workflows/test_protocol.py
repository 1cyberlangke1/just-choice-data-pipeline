"""契约一致性：每个注册的 workflow 都真的实现了 `Workflow` 协议。

为什么要有：协议是 `runtime_checkable` 的，但**没人检查过**——少一个方法（实测踩到过
`case_label`）只会在管线跑到最后一步写 manifest 时才炸，而且是在「跑完一整批、烧完配额」之后。
这条把它提前到导入期，零 token。

只查「方法在不在、声明项对不对、公开接口能不能拿第一个 case 走通」——不跑生成、不碰网络。

为什么不是每家一条参数化用例：下面两条查的是**同一段通用代码**（协议自检、`build_cases` /
`case_label` / `stats_dimensions` 的通用路径），换的只是各家的数据。51 个 workflow 摊成 153 条
用例，等于把同一件事报 51 遍；收成循环之后失败信息里照样点名是哪个 workflow、哪个切分，
用例数从 153 降到 2。原来那条「语言文件在磁盘上」被
`tests/workflows/test_shared.py::test_every_workflow_ships_zh_and_en` 完全覆盖（它连
「声明与磁盘逐字相同」都查），在这里删掉。
"""

from __future__ import annotations

from jc.workflow import Workflow, discover, load

#: 已注册的 workflow 名（导入即注册）。名单为空说明自注册坏了，下面几条会一起报出来。
WORKFLOWS = discover()


def test_discover_finds_every_workflow() -> None:
    """输入：无。输出：无。预期行为：注册表非空，且每个名字都能加载。"""
    assert len(WORKFLOWS) > 20
    for name in WORKFLOWS:
        assert load(name).name == name


def test_workflow_satisfies_the_protocol() -> None:
    """输入：无（名单现场列）。输出：无。预期行为：协议要求的方法与声明项一个不缺。

    失败信息把**每一家**的问题都攒起来一起报（`bad` 列表），不因为第一家炸掉就看不见其余。
    """
    bad: list[str] = []
    for name in WORKFLOWS:
        instance = load(name)
        if not isinstance(instance, Workflow):
            bad.append(f"{name} 没有实现完整的 Workflow 协议")
            continue
        if not instance.version:
            bad.append(f"{name} 没有 version")
        if not (isinstance(instance.kinds, frozenset) and instance.kinds):
            bad.append(f"{name} 的 kinds 不是非空 frozenset：{instance.kinds!r}")
        if not isinstance(instance.needs, frozenset):
            bad.append(f"{name} 的 needs 不是 frozenset：{instance.needs!r}")
        if not (isinstance(instance.judges, tuple) and instance.judges):
            bad.append(f"{name} 的 judges 不是非空 tuple：{instance.judges!r}")
    assert not bad, bad


def test_every_split_yields_cases_and_readable_labels() -> None:
    """输入：无（名单现场列）。输出：无。

    预期行为：每个切分都有 case、case 有可读标签、维度表非空——这几条是管线写 manifest 与统计
    报告时一定会用到的（`case_label` 少一个就会在这里当场炸，而不是跑完整批之后）。
    """
    bad: list[str] = []
    for name in WORKFLOWS:
        instance = load(name)
        if not instance.dimension_titles():
            bad.append(f"{name} 的 dimension_titles() 是空的")
        for split in instance.split_names():
            cases = instance.build_cases(instance.config_for(split))
            if not cases:
                bad.append(f"{name}/{split} 一个 case 都没有")
                continue
            case = cases[0]
            if not instance.case_label(case):
                bad.append(f"{name}/{split} 的 case_label 是空的")
            if not instance.stats_dimensions(case):
                bad.append(f"{name}/{split} 的 stats_dimensions 是空的")
    assert not bad, bad
