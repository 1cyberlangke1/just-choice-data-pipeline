"""单据 workflow 测试的公共夹具：小规格构造器（不联网、不落盘）。

`make_case` 造一个 8 维组合，`make_spec` 由它造出真值规格；
两者都是**工厂夹具**（返回可调用对象），测试自己决定种子与覆盖项。
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.invoice import spec as sp

#: 默认维度：一条 3 行、1 处错、带正确税额的普通单据（其余维度各取第一档）。
CASE_DEFAULTS: dict[str, Any] = {
    "layout": "labeled_lines",
    "labels": "canonical",
    "currency": "cny",
    "date_format": "dashed",
    "lines": 3,
    "defects": 1,
    "subtotal_ok": True,
    "tax_state": "ok",
}


def _make_case(**overrides: Any) -> sp.InvoiceCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.InvoiceCase(**fields)


def _make_spec(seed: int = 5, **overrides: Any) -> sp.InvoiceSpec:
    """输入：种子与 case 覆盖项。输出：真值规格。

    预期行为：同种子必得同一份规格（构造器只用传进来的 RNG，不碰全局随机源）。
    """
    return sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(tax_state="none")` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(7, lines=4)` 造一份真值规格。"""
    return _make_spec
