"""裸装模拟器：把 LLM 后端要用的第三方包挡掉，再按参数跑一项检查（人与测试共用）。

输入：命令行第一个参数，四种检查之一：

- `import`：import 核心层（`jc.pipeline` 与 `jc.backend.types`）；
- `cli`：走一遍 `jc workflows` 那条路（`jc.cli.main(["workflows"])`）；
- `maze`：走一遍零 token 的 `maze` 生成；
- `hint`：真要 `from jc.backend import build_backend` 时，报的必须是人话。

输出：stdout 一行结论；退出码 0 = 这项检查通过。
预期行为：挡包器必须在 import `jc` **之前**装上——否则量到的还是「装了依赖」那条路。

为什么值得有：`pyproject.toml` 写着 `dependencies = []`，README 写着「通用层与纯构造 workflow
只用标准库」，可 `jc.backend/__init__.py` 是**模块级** import `engine` / `probe` / `llm` 的，
而 `jc.pipeline` 又 `from .backend.types import GenerationBackend`——于是裸装（只
`pip install -e .`）连 `jc workflows` 都起不来（2026-10-04 冒烟实测：直接
`ModuleNotFoundError: No module named 'instructor'`）。这条尺子把那句承诺钉住。
"""

from __future__ import annotations

import importlib.abc
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Final

#: LLM 后端（可选 extra `[llm]`）要用的第三方包：核心层与纯构造 workflow 一个都不该碰。
BLOCKED: Final[tuple[str, ...]] = (
    "instructor",
    "openai",
    "pydantic",
    "pyrate_limiter",
    "tenacity",
)


class Blocker(importlib.abc.MetaPathFinder):
    """挡包器：名字在 `BLOCKED` 里就当成「这台机器没装它」。"""

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> object:
        """输入：模块全名与查找上下文。输出：None（不接管查找）。

        预期行为：被挡的包**当场抛** `ModuleNotFoundError`（返回 None 只是「找不到」，
        不会阻止后面的 finder 接着找），这样才真的等价于「没装」。
        """
        top = fullname.split(".")[0]
        if top in BLOCKED:
            raise ModuleNotFoundError(f"No module named {top!r}", name=top)
        return None


def block() -> None:
    """输入：无。输出：无。预期行为：把挡包器插到 `sys.meta_path` 最前面（重复调用只插一次）。"""
    if not any(isinstance(finder, Blocker) for finder in sys.meta_path):
        sys.meta_path.insert(0, Blocker())


def check_import() -> int:
    """输入：无。输出：退出码。预期行为：核心层在裸装下 import 得动。"""
    import jc.pipeline as pipeline_mod
    from jc.backend.types import GenerationBackend

    print(f"ok: 核心层裸装可用（{pipeline_mod.RunConfig.__name__} / {GenerationBackend.__name__}）")
    return 0


def check_cli() -> int:
    """输入：无。输出：退出码。预期行为：`jc workflows` 在裸装下照样列得出名字。"""
    from jc import cli

    return cli.main(["workflows"])


def check_maze() -> int:
    """输入：无。输出：退出码。预期行为：零 token 的 maze 生成在裸装下跑得完、产得出样本。"""
    from jc import cli, jsonio

    with tempfile.TemporaryDirectory() as tmp:
        code = cli.main(
            [
                "gen",
                "--workflow",
                "maze",
                "--split",
                "train",
                "--locale",
                "zh",
                "--out",
                tmp,
                "--run-id",
                "bare-maze",
            ]
        )
        if code != 0:
            return code
        rows = list(jsonio.iter_jsonl(Path(tmp) / "bare-maze" / "train.jsonl"))
        print(f"ok: 裸装生成了 {len(rows)} 条 maze 样本")
        return 0 if rows else 1


def check_hint() -> int:
    """输入：无。输出：退出码。预期行为：缺依赖时报人话（指名缺什么、怎么装）。"""
    try:
        from jc.backend import build_backend
    except ModuleNotFoundError as exc:
        text = str(exc)
        print(text)
        return 0 if ".[llm]" in text else 1
    print(f"意外：缺依赖时 build_backend 竟然 import 成功了（{build_backend!r}）")
    return 1


#: 检查名 → 检查函数。加一项就在这里挂一行。
CHECKS: Final[dict[str, Callable[[], int]]] = {
    "import": check_import,
    "cli": check_cli,
    "maze": check_maze,
    "hint": check_hint,
}


def main(argv: list[str]) -> int:
    """输入：命令行参数。输出：退出码。预期行为：装上挡包器之后跑对应的那一项检查。"""
    block()
    name = argv[0] if argv else "import"
    check = CHECKS.get(name)
    if check is None:
        print(f"未知检查：{name!r}（可选：{sorted(CHECKS)}）")
        return 2
    return check()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
