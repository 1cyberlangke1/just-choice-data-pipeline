"""README 里那两条命令（装 `.[llm,dev]`、跑 `python -m pytest -q`）必须真的跑得起来。

输入：`pyproject.toml` 声明的依赖，加上 `src/` / `workflows/` / `tests/` 里所有顶层 import。
输出：无（断言）。
预期行为：代码 import 的每个第三方顶层包，要么在 `dependencies` / 某个 extra 里声明过，
要么在 `TRANSITIVE` 里写明「由谁带进来」——README 的装法就这么多，没声明的包装不上。

为什么值得有：2026-10-04 实测——按 README 装完 `.[llm,dev]` 之后 `python -m pytest -q`
直接 `No module named pytest`（`dev` extra 里只有 ruff，而 README 的「零 token 的验证闸」
那一节正是让你跑 pytest）。这条尺子把「装法够不够跑文档里那条命令」变成机器可查，
而不是靠人记得同步两处。
"""

from __future__ import annotations

import ast
import fnmatch
import re
import sys
import tomllib
from pathlib import Path
from typing import Final

#: 仓库根：本文件在 `<仓库>/tests/core/` 下。
ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: 仓库自己的包（不是第三方依赖，不该出现在 pyproject 的依赖表里）。
LOCAL: Final[frozenset[str]] = frozenset({"jc", "workflows", "tests", "conftest", "just_choice"})

#: 不直接声明、由已声明的包带进来的第三方包 → 理由。写清「谁带进来的」，否则下次没人敢删。
TRANSITIVE: Final[dict[str, str]] = {
    "httpx2": "openai 3.x 的 HTTP 客户端依赖（openai 自己的 Requires-Dist 里就有它）",
}

#: 依赖声明里包名的形态：`pyrate-limiter>=4.0` → `pyrate_limiter`。
REQUIREMENT: Final[re.Pattern[str]] = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _normalize(name: str) -> str:
    """输入：包名。输出：import 时用的写法（`pyrate-limiter` → `pyrate_limiter`，统一小写）。"""
    return name.replace("-", "_").lower()


def declared_packages() -> set[str]:
    """输入：无。输出：`dependencies` 与所有 extra 里声明过的包名（归一化后）。"""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    specs: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)
    return {_normalize(match.group(1)) for spec in specs if (match := REQUIREMENT.match(spec))}


def imported_third_party() -> dict[str, set[str]]:
    """输入：无。输出：{第三方顶层包名: {import 了它的文件（相对仓库根、正斜杠）}}。

    预期行为：连函数体里的 import 也算——`jc.cli._load_backend` 就是在函数里才 import
    `jc.backend` 的，那同样是「跑这条命令要装什么」的一部分。
    """
    found: dict[str, set[str]] = {}
    for folder in ("src", "workflows", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in LOCAL or name in sys.stdlib_module_names:
                        continue
                    key = _normalize(name)
                    found.setdefault(key, set()).add(
                        str(path.relative_to(ROOT)).replace("\\", "/")
                    )
    return found


def test_every_imported_package_is_declared_or_explained() -> None:
    """输入：无。输出：无。

    预期行为：没有「被 import 了、却既没声明也没写理由」的第三方包。报了就把包名与
    头几个文件一起点名，省得再去翻。
    """
    declared = declared_packages()
    missing = {
        name: sorted(files)[:3]
        for name, files in imported_third_party().items()
        if name not in declared and name not in TRANSITIVE
    }
    assert not missing, (
        "这些第三方包被 import 了，但 pyproject 没声明、TRANSITIVE 里也没写理由——"
        f"照 README 的装法装不上它们：{missing}"
    )


#: `[tool.setuptools.package-dir]` 的两条映射（包名前缀 → 磁盘目录）。新加顶层包要同步这里。
PACKAGE_ROOTS: Final[dict[str, str]] = {"jc": "src/jc", "workflows": "workflows"}


def package_dirs() -> dict[str, Path]:
    """输入：无。输出：{包名: 磁盘目录}，按 `PACKAGE_ROOTS` 现算。

    预期行为：只有带 `__init__.py` 的目录才算包（`locales` / `__pycache__` 不算），
    子包按点分名字给全（`src/jc/backend` → `jc.backend`）。
    """
    out: dict[str, Path] = {}
    for prefix, folder in PACKAGE_ROOTS.items():
        base = ROOT / folder
        if not (base / "__init__.py").exists():
            continue
        out[prefix] = base
        for init in sorted(base.rglob("__init__.py")):
            parts = init.parent.relative_to(base).parts
            if parts:
                out[".".join((prefix, *parts))] = init.parent
    return out


def setuptools_table() -> dict[str, object]:
    """输入：无。输出：`pyproject.toml` 的 `[tool.setuptools]` 表。"""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    table = data["tool"]["setuptools"]
    assert isinstance(table, dict), "[tool.setuptools] 必须是表"
    return table


def test_pyproject_ships_every_package_on_disk() -> None:
    """输入：无。输出：无。

    预期行为：`packages` 与磁盘上的包**逐名相等**。少一格 = `pip install .` 出来的 wheel
    少了那一家 workflow（editable 装也按这张表找包）；多一格 = 表里留着已经删掉的包。
    为什么值得有：2026-10-04 实测——把 `workflows.sentiment` 从表里删掉，全库测试**一条都不红**，
    因为测试是从仓库根 import 的，走不到打包路径。
    """
    packages = setuptools_table()["packages"]
    assert isinstance(packages, list), "packages 必须是数组"
    declared = {str(name) for name in packages}
    on_disk = set(package_dirs())
    assert declared == on_disk, (
        f"pyproject 的 packages 与磁盘对不上：少 {sorted(on_disk - declared)}、"
        f"多 {sorted(declared - on_disk)}"
    )


def test_pyproject_package_data_covers_every_locale_file() -> None:
    """输入：无。输出：无。

    预期行为：每个包目录下的 `locales/**/*.toml` 都被该包 `package-data` 的某个 glob 命中，
    且 `package-data` 的键都是真的包名。语言文件就是判据原文与题面模板，漏一个，装出来的包
    里那家 workflow 渲染不出题面。
    """
    table = setuptools_table()
    data = table["package-data"]
    assert isinstance(data, dict), "package-data 必须是表"
    packages = package_dirs()
    bad_keys = sorted(key for key in data if key not in packages)
    assert not bad_keys, f"package-data 里有不是包的键：{bad_keys}"
    missed: list[str] = []
    for name, folder in packages.items():
        globs = [str(pattern) for pattern in data.get(name, [])]
        for toml in sorted((folder / "locales").rglob("*.toml")):
            rel = toml.relative_to(folder).as_posix()
            if not any(fnmatch.fnmatch(rel, pattern) for pattern in globs):
                missed.append(f"{name}: {rel}")
    assert not missed, f"这些语言文件没被打进包：{missed}"
