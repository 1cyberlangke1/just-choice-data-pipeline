"""命令审批 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.shell import spec as sp

#: 默认维度：bash 上一条「查看 build/」的只读命令，目标一致、写法最朴素。
CASE_DEFAULTS: dict[str, Any] = {
    "dialect": "bash",
    "op": "list_dir",
    "target": "build_dir",
    "goal_target": "build_dir",
    "wrapper": "plain",
    "goal_style": "imperative",
    "reads": 2,
}


def _make_case(**overrides: Any) -> sp.ShellCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.ShellCase(**fields)


def _make_spec(seed: int = 5, **overrides: Any) -> sp.CommandSpec:
    """输入：种子与 case 覆盖项。输出：真值规格。

    预期行为：同种子必得同一条规格（构造器只用传进来的 RNG，不碰全局随机源）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train") -> sp.ShellConfig:
    """输入：切分名。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：两种方言都在，三个操作覆盖「只读 / 写 / 凭据对象」三类；
    写法给两种（`plain` 与提权），问法给两种。
    """
    return sp.ShellConfig(
        split=split,
        dialects=sp.DIALECTS,
        ops=("list_dir", "remove_tree", "read_file"),
        wrappers=("plain", "elevated"),
        goal_styles=("imperative",),
        read_counts=(1,),
        phrasings=(0, 1),
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(op="remove_tree", target="dist_dir")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(7, op="read_file", target="ssh_key")` 造一条真值规格。"""
    return _make_spec
