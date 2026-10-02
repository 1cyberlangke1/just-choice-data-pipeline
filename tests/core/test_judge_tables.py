"""全库尺子：每个 workflow 声明的判别器名单，都必须真的造得出来。

为什么要有这条：名单表（`Workflow.judge_table`）里写的是**类名字符串**——`standard_judges`
用 `getattr(judges 模块, 类名)` 取类，写错一个字母静态检查看不见（`ruff` 的 F821 只认
真名字，`getattr` 的字符串参数它够不到）。这张表把「写错」变成一条当场失败的测试：

1. `judge_table` 的键集合必须与 `judges` 声明完全一致（多了、少了都报）；
2. 声明里的每个名字都必须造得出实例，且实例的 `name` 与声明一致（这一条同时证明类名没写错、
   吃法没写错——`BARE` / `LOCALE` / `BACKEND` 三档各有各的调用形状，传错参数当场 `TypeError`）；
3. 纯构造式的 workflow 不许在表里放 `BACKEND` 那一档（没有后端可传，`make_judges` 会报错）。

输入：`jc.workflow.discover()` 列出的全部 workflow。输出：无（断言）。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.workflow import discover, load

#: 语言：随便挑一个——判别器只在构造时存下它，这一步不发请求。
LOCALE: Final[str] = "zh"


class StubBackend:
    """输入：无。输出：一个什么都不做的后端。

    预期行为：只为把 `BACKEND` 那一档判别器**造出来**——构造不发请求，所以它不需要真能力。
    """


def _workflows() -> list[tuple[str, object]]:
    """输入：无。输出：[(名字, workflow 实例)]。预期行为：实例化失败直接抛（那是真问题）。"""
    return [(name, load(name)) for name in discover()]


def test_every_workflow_declares_a_matching_table() -> None:
    """输入：无。输出：无。预期行为：`judges` 与 `judge_table` 的键集合逐家相等。"""
    for name, workflow in _workflows():
        table = set(workflow.judge_table)
        declared = set(workflow.judges)
        assert declared == table, f"{name}: judges={sorted(declared)} 与 judge_table={sorted(table)} 对不上"


def test_declared_judges_can_all_be_built() -> None:
    """输入：无。输出：无。预期行为：声明里的每个名字都造得出，且名字与声明一致。"""
    for name, workflow in _workflows():
        if workflow.needs:
            workflow.bind_backend(StubBackend())
        built = workflow.make_judges(tuple(workflow.judges), LOCALE)
        assert [judge.name for judge in built] == list(workflow.judges), f"{name} 造出来的判别器对不上"


def test_unknown_judge_name_is_rejected_everywhere() -> None:
    """输入：无。输出：无。预期行为：不存在的名字在任何一家都报 `KeyError`。"""
    for name, workflow in _workflows():
        if workflow.needs:
            workflow.bind_backend(StubBackend())
        with pytest.raises(KeyError):
            workflow.make_judges(("不存在的判别器",), LOCALE)
