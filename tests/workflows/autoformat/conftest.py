"""结构恢复 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

from typing import Any

import pytest

from workflows.autoformat import spec as sp

#: 默认 case：中文主题、行宽 64、纯标题 + 正文的文档。
CASE_DEFAULTS: dict[str, Any] = {
    "topic": "build",
    "wrap": 64,
    "shape": "plain",
    "variant": "na",
    "code_family": "na",
}


def _make_case(**overrides: Any) -> sp.DocCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.DocCase(**fields)


def _make_spec(**overrides: Any) -> sp.DocSpec:
    """输入：case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（计划由素材指纹派生，不掷骰子）。
    """
    return sp.sample_spec(_make_case(**overrides), "train", 0, 0)


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.DocConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：四个形状覆盖四个题族（`plain` 只有并段与分类，`list` 带列表项与有序性，
    `callout` 带提示框种类，`code` 带代码块）；变体全开，免得某一族在同分布侧退化成常量。
    """
    return sp.DocConfig(
        split=split,
        topics=("build", "incident"),
        wraps=(56, 64),
        shapes=("plain", "list", "code", "callout"),
        variants=sp.ALL_VARIANTS,
        code_families=("na", "build"),
        phrasings=(0, 1),
        ood_only=ood_only,
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(shape="list", variant="ordered")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(shape="callout", variant="warning")` 造一条真值规格。"""
    return _make_spec
