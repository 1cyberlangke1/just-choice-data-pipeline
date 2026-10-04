"""裸装（只 `pip install -e .`）必须跑得动核心层与纯构造 workflow。

输入：`tests/core/_no_llm_deps.py` 在**挡掉 `[llm]` 依赖**的解释器里跑出来的结果。
输出：无（断言）。
预期行为：核心层 import、`jc workflows`、零 token 的 `maze` 生成三条路都不许崩；
真要走到后端时报的是人话（指名缺什么、怎么装）。

为什么值得有：`pyproject.toml` 写着 `dependencies = []`，README 写着「通用层与纯构造 workflow
只用标准库」，但 `jc.backend/__init__.py` 是模块级 import `engine` / `probe` / `llm` 的，
而 `jc.pipeline` 只用 `backend.types` 的类型标注——于是裸装环境下连 `python -m jc workflows`
都是 `ModuleNotFoundError: No module named 'instructor'`（2026-10-04 冒烟实测）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

#: 仓库根：本文件在 `<仓库>/tests/core/` 下。
ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: 裸装模拟器。
HELPER: Final[Path] = Path(__file__).resolve().parent / "_no_llm_deps.py"


def run_check(name: str) -> subprocess.CompletedProcess[str]:
    """输入：检查名。输出：子进程结果。

    预期行为：另起一个解释器（`sys.executable`）跑 `HELPER`，`PYTHONPATH` 指向本仓库的
    `src/` 与根目录；挡包器由 `HELPER` 自己在 import `jc` 之前装上。
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    return subprocess.run(
        [sys.executable, str(HELPER), name],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=300,
    )


def test_core_layer_imports_without_the_llm_extra() -> None:
    """输入：裸装下 import 核心层。输出：无。预期行为：成功。"""
    proc = run_check("import")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "ok:" in proc.stdout


def test_workflows_command_works_without_the_llm_extra() -> None:
    """输入：裸装下跑 `jc workflows`。输出：无。

    预期行为：列得出名字（至少构造式那 14 家）；走 LLM 的那几家被跳过时，理由里必须给装法
    （`.[llm]`），不能只甩一句 `No module named 'pydantic'` 让人自己猜。
    """
    proc = run_check("cli")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "maze" in proc.stdout.split()
    assert ".[llm]" in proc.stderr


def test_constructive_generation_runs_without_the_llm_extra() -> None:
    """输入：裸装下生成一批 `maze`。输出：无。预期行为：退出码 0，且真的产出了样本。"""
    proc = run_check("maze")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "ok:" in proc.stdout


def test_missing_backend_dependency_says_how_to_install() -> None:
    """输入：裸装下取 `build_backend`。输出：无。预期行为：报错文本里有装法。"""
    proc = run_check("hint")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
