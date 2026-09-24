"""通用层测试：workflow 注册表与组装器（不依赖任何具体 workflow）。"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from jc import assemble
from jc.workflow import LabeledQuestion, discover, family_of, load, register, registered

# 仓库根：CLI 子进程要以它为工作目录，才能 import 到 workflows/ 包。
ROOT = Path(__file__).resolve().parents[2]


def test_family_of_strips_suffix() -> None:
    assert family_of("can_move") == "can_move"
    assert family_of("can_move_2") == "can_move"
    assert family_of("off_by_one_11") == "off_by_one"


def test_register_rejects_duplicate() -> None:
    class Dummy:
        name = "dummy-for-test"

    register("dummy-for-test")(lambda: Dummy())
    with pytest.raises(ValueError):
        register("dummy-for-test")(lambda: Dummy())
    assert "dummy-for-test" in registered()


def test_load_unknown_workflow_lists_known() -> None:
    with pytest.raises(KeyError):
        load("no-such-workflow")


def test_load_maze_by_dynamic_import() -> None:
    """通用层不静态依赖 maze：靠动态导入触发自注册。"""
    workflow = load("maze")
    assert workflow.name == "maze"
    assert workflow.needs == frozenset()  # 纯构造式


def test_discover_finds_maze_package() -> None:
    """输入：无。输出：已注册的 workflow 名。

    预期行为：扫描 `workflows/` 包并逐个导入以触发自注册——否则注册表只在
    「有人恰好 import 过」时才有内容。
    """
    assert "maze" in discover()


def test_cli_workflows_lists_maze_in_fresh_process() -> None:
    """输入：独立子进程执行 `python -m jc workflows`。输出：进程输出与退出码。

    预期行为：新进程里注册表是空的，`jc workflows` 仍必须自己发现并列出 maze
    （否则这条命令恒为空，而 `--workflow` 的帮助文本正指着它）。
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    proc = subprocess.run(
        [sys.executable, "-m", "jc", "workflows"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "maze" in proc.stdout.split()


def test_assemble_shuffles_and_maps_answer_to_index() -> None:
    labeled = [
        LabeledQuestion(
            family="pick",
            question={
                "type": "choice",
                "instructions": "which?",
                "criteria": {"a": None, "b": None, "c": None},
            },
            answer="b",
        )
    ]
    sample = assemble.assemble_sample("maze-train-0000000", "s", labeled, random.Random(1))
    criteria = list(sample["questions"]["pick"]["criteria"])
    assert sample["answers"]["pick"] == {"type": "choice", "choice": criteria.index("b")}


def test_assemble_suffixes_repeated_families() -> None:
    labeled = [
        LabeledQuestion("ask", {"type": "noul", "instructions": "x"}, True, shuffle=False),
        LabeledQuestion("ask", {"type": "noul", "instructions": "y"}, False, shuffle=False),
    ]
    sample = assemble.assemble_sample("maze-train-0000000", "s", labeled, random.Random(1))
    assert list(sample["questions"]) == ["ask", "ask_2"]
    assert list(sample["answers"]) == ["ask", "ask_2"]
