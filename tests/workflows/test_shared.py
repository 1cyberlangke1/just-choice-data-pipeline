"""全库通用的协议级不变量：一份测试跑遍所有已注册的 workflow。

输入：无（workflow 名单由 `jc.workflow.discover()` 现场列出来）。
输出：无。
预期行为：把「每个 workflow 都该满足」的通用断言只写一遍——各 workflow 自己的测试包只留
领域特有的断言，通用断言在这里统一钉住，不再 48 个包里各抄一份。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from jc import workflow as workflow_mod

#: 已注册的 workflow 名单（导入即注册，现场列，不写死）。
WORKFLOWS: list[str] = workflow_mod.discover()


def locales_of(name: str) -> ModuleType:
    """输入：workflow 名。输出：它的 `locales` 模块。

    预期行为：语言包是每个 workflow 都该有的通用件，缺了就直接报错，不静默跳过。
    """
    return importlib.import_module(f"workflows.{name}.locales")


@pytest.mark.parametrize("name", WORKFLOWS)
def test_every_workflow_ships_zh_and_en(name: str) -> None:
    """输入：一个 workflow。输出：无。预期行为：语言列表恰好是 en / zh，顺序稳定。"""
    assert locales_of(name).available() == ["en", "zh"]


@pytest.mark.parametrize("name", WORKFLOWS)
def test_pick_wraps_around(name: str) -> None:
    """输入：越界的问法编号。输出：无。预期行为：回绕取模，不抛 IndexError。"""
    pick = getattr(locales_of(name), "pick", None)
    if pick is None:
        pytest.skip(f"{name} 的语言包没有 pick（maze 用 phrasing 系列）")
    assert pick(["a", "b"], 0) == "a"
    assert pick(["a", "b"], 1) == "b"
    assert pick(["a", "b"], 3) == "b"
    assert pick(["a", "b"], 4) == "a"


@pytest.mark.parametrize("name", WORKFLOWS)
def test_unknown_locale_is_rejected(name: str) -> None:
    """输入：不存在的语言。输出：无。预期行为：报 FileNotFoundError 并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="没有语言文件"):
        locales_of(name).load("xx")


@pytest.mark.parametrize("name", WORKFLOWS)
def test_audit_on_an_empty_dir_reports_missing_artifacts(
    name: str, tmp_path: Path
) -> None:
    """输入：空目录、两个切分。输出：无。

    预期行为：两个切分的样本数都是 0，且各报一条缺失——审计只读产物，缺文件不许静默当
    「没有样本」。
    """
    report, missing = workflow_mod.load(name).audit(tmp_path, ("train", "val"), "zh")
    if "train" not in report:
        pytest.skip(f"{name} 的审计报告不按 train/val 分键（maze 自带格式）")
    assert report["train"]["samples"] == 0
    assert report["val"]["samples"] == 0
    assert len(missing) >= 2


@pytest.mark.parametrize("name", WORKFLOWS)
def test_pure_construction_workflows_refuse_a_backend(name: str) -> None:
    """输入：一个假后端。输出：无。

    预期行为：`needs` 为空的 workflow 根本不该被绑后端——收到就报错，不静默吞掉；
    需要后端的 workflow 由各自测试包覆盖，这里跳过。
    """
    wf = workflow_mod.load(name)
    if wf.needs:
        pytest.skip(f"{name} 需要后端")
    with pytest.raises(AssertionError, match="纯构造式"):
        wf.bind_backend(object())
