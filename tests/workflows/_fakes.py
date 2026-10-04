"""假后端共用的构造件：造一个填满记账字段的 `Draft`，以及各家假后端的共用底座。

输入：草稿数据、原始文本。输出：`Draft`。预期行为：记账字段填假值（`fake`），不进产物——
三十来个包的 conftest 各抄一份纯属重复，抄歪了还会让「假后端长什么样」各包不一致。
"""

from __future__ import annotations

import pytest

from typing import TypeVar

from jc.backend.types import Draft
from jc.jsontypes import JsonValue


class ScriptedBackend:
    """各家脚本化假后端的共用底座：只放「逐字同构」的那一个方法。

    输入：任务列表。输出：与入参顺序一致的草稿列表。预期行为：逐个走子类的 `generate`——
    38 个包的假后端原来各抄一份，而那三行本来就只有一句。
    """

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：与入参顺序一致的草稿列表。"""
        return [self.generate(task) for task in tasks]


def draft(data: dict[str, JsonValue], raw_text: str = "<fake>") -> Draft:
    """输入：草稿数据、原始文本。输出：`Draft`。预期行为：记账字段填假值，不进产物。"""
    return Draft(
        data=data,
        raw_text=raw_text,
        backend="fake",
        provider="fake",
        model="fake",
        mode="fake",
    )


BackendT = TypeVar("BackendT")


def instance_fixture(cls: type[BackendT]) -> object:
    """输入：某个包自己的假后端类。输出：一个 pytest 夹具（叫 `backend`，调用即返回**实例**）。

    预期行为：`backend` 就是「一个完全兑现计划的假后端」——23 家原来各抄一份那三行，
    只有类名不同，收在这里只留一份。
    """

    @pytest.fixture
    def backend() -> BackendT:
        return cls()

    return backend


def factory_fixture(cls: type[BackendT]) -> object:
    """输入：某个包自己的假后端类。输出：一个 pytest 夹具（叫 `fake_backend`，返回**那个类**）。

    预期行为：测试拿它按开关现造后端，例如 `fake_backend(trace_lies=True)`——开关名各家不同，
    看各家 `FakeBackend.__init__`。37 家原来各抄一份那三行，只有类名不同，收在这里只留一份。
    """

    @pytest.fixture
    def fake_backend() -> type[BackendT]:
        return cls

    return fake_backend
