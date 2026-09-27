"""行级规则检查 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。"""

from __future__ import annotations

from typing import Any

import pytest

from workflows.lint import spec as sp

#: 默认 case：硬编码凭据、源码体裁、命中 1 行、第一套句面。
CASE_DEFAULTS: dict[str, Any] = {
    "rule": "secret",
    "genre": "code",
    "hits": 1,
    "phrasing": 0,
}


def _make_case(**overrides: Any) -> sp.LintCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.LintCase(**fields)


def _make_spec(**overrides: Any) -> sp.LintSpec:
    """输入：case 覆盖项。输出：真值规格。

    预期行为：同 case 必得同一份规格（计划由素材指纹派生，不掷骰子）。
    """
    return sp.sample_spec(_make_case(**overrides), "train", 0)


def tiny_config(split: str = "train", *, ood_only: bool = False) -> sp.LintConfig:
    """输入：切分名、是否只留 OOD。输出：极小采样参数（零 token，跑得动整条管线）。

    预期行为：四张表**全开**——少一条规则会让它那一族答案退化成常量，少一档命中行数会让
    `count` / `gate` 的分布塌掉（均衡断言当场红）。
    """
    return sp.LintConfig(
        split=split,
        rules=sp.RULES,
        genres=sp.GENRES,
        hits=sp.HITS,
        phrasings=sp.PHRASINGS,
        ood_only=ood_only,
    )


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(rule="todo", hits=3)` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(genre="log")` 造一条真值规格。"""
    return _make_spec
