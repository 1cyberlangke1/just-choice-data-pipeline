"""采样旋钮校验（`jc.knobs`）的单元测试：空表 / 区间长度 / 类型 / 上下界。"""

from __future__ import annotations

import pytest

from jc.knobs import require_bounds, require_pool


@pytest.mark.parametrize("field", ["algos", "themes"])
def test_require_pool_rejects_empty(field: str) -> None:
    """输入：空表。输出：无。预期行为：报人话，且点名是哪个字段。"""
    with pytest.raises(ValueError, match=field):
        require_pool(field, ())


def test_require_pool_accepts_single_value() -> None:
    """输入：只有一个取值的表。输出：无。预期行为：合法——小批量就是靠收窄取值表做的。"""
    require_pool("themes", (0,))


def test_require_bounds_accepts_pair() -> None:
    """输入：(3, 7)。输出：无。预期行为：正常区间放行。"""
    require_bounds("size", (3, 7))


def test_require_bounds_accepts_equal_bounds() -> None:
    """输入：(4, 4)。输出：无。预期行为：上下界相等合法（只有一个取值的区间）。"""
    require_bounds("marks_range", (4, 4))


@pytest.mark.parametrize("bounds", [(), (3,), (1, 2, 3)])
def test_require_bounds_rejects_wrong_length(bounds: tuple[int, ...]) -> None:
    """输入：不是两个元素的区间。输出：无。预期行为：报人话并写出实际有几个。"""
    with pytest.raises(ValueError, match="两个数"):
        require_bounds("size", bounds)


@pytest.mark.parametrize("bounds", [("3", 7), (3, 7.0), (True, 7)])
def test_require_bounds_rejects_non_int(bounds: tuple[object, ...]) -> None:
    """输入：含非整数（含 bool——它是 int 的子类）的区间。输出：无。预期行为：报人话。"""
    with pytest.raises(ValueError, match="整数"):
        require_bounds("size", bounds)  # type: ignore[arg-type]


def test_require_bounds_rejects_inverted() -> None:
    """输入：下界大于上界。输出：无。预期行为：报人话并写出两个值。"""
    with pytest.raises(ValueError, match="下界不能大于上界"):
        require_bounds("questions_range", (6, 3))
