"""全仓库禁用显式 `Any`。

为什么要有：`Any` 会让类型检查器在该处彻底闭嘴——等于把「没标注」伪装成「标注了」。
用户原话：「类型标注全是 any 和没标注有什么区别」。这条把它钉死，防止再长回来。

输入：`src/` `workflows/` `tests/` 下的全部 `*.py` 加仓库根那一层的全部 `*.py`。
输出：无（断言）。
预期行为：用 **AST** 扫，不用 grep——grep 会把注释、文档、字符串里的 `Any` 也算进去，
而 AST 只看真正的 import 与注解。想放宽时改 `EXEMPT` 并写清理由，不许整目录豁免。

禁用范围（漏一条就等于没禁）：

1. `from typing import Any`、`from typing_extensions import Any`（含 `as` 改名）；
2. `from typing import *` / `from typing_extensions import *`——星号导入会把 `Any`
   悄悄带进作用域，没法证明它没被用；
3. 裸名字 `Any`，以及属性 `typing.Any` / `t.Any` / `typing_extensions.Any`；
4. **字符串注解**（`x: "list[Any]"`）与 `# type:` 注释里的 `Any`——它们以字符串存在，
   AST 里看不见名字，只能把字符串再解析一遍；
5. `getattr(typing, "Any")` 这种字面量取属性。

不在此列的：文档串、普通注释、测试里当**数据**用的字符串（AST 看不见它们）。
动态拼出来的名字（`getattr(m, "An" + "y")`）静态查不到，属于明知漏网，不假装能查。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: 扫描入口：这几个目录递归扫 `*.py`。
ROOTS: tuple[str, ...] = ("src", "workflows", "tests")

#: 仓库根那一层的散装脚本（`conftest.py`，以及将来可能出现的 `setup.py` / `noxfile.py`）。
#: 用 glob 而不是写死文件名：写死的话，新加一个根脚本就等于在规则外开了个口子。
ROOT_GLOB: str = "*.py"

#: 豁免文件：键是相对路径，值是理由。空 = 一个都不豁免。
EXEMPT: dict[str, str] = {}

#: 会提供 `Any` 的模块——星号导入与 `Any` 名字都从这两处进来。
_ANY_MODULES = frozenset({"typing", "typing_extensions"})

#: 要找的名字本身。
_BANNED = "Any"


def _sources() -> list[str]:
    """输入：无。输出：待检查文件的相对路径（排序后）。

    预期行为：目录递归取 `*.py` 并跳过 `__pycache__`；仓库根那一层的 `*.py` 也全收
    （写死 `conftest.py` 会让新加的根脚本漏网）。
    """
    found: list[str] = []
    for entry in ROOTS:
        path = Path(entry)
        for child in path.rglob("*.py"):
            if "__pycache__" in child.parts:
                continue
            found.append(child.as_posix())
    found.extend(path.as_posix() for path in Path().glob(ROOT_GLOB))
    return sorted(set(found))


def _hits_in(node: ast.AST) -> list[int]:
    """输入：任意 AST 子树。输出：其中裸 `Any` / `.Any` 的行号。

    预期行为：只看名字与属性节点；字符串常量不算（字符串由 `_string_sites` 单独处理）。
    """
    found: list[int] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == _BANNED:
            found.append(sub.lineno)
        elif isinstance(sub, ast.Attribute) and sub.attr == _BANNED:
            found.append(sub.lineno)
    return found


def _parse_annotation_text(text: str) -> ast.expr | None:
    """输入：注解的字符串写法（如 `list[Any]`）。输出：解析后的表达式；解析不了给 None。

    预期行为：用 `eval` 模式解析——注解本来就是个表达式。解析失败说明它压根不是
    合法注解（比如 `# type:` 后面写了句人话），按「查不到」处理，不抛异常。
    """
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError:
        return None


def _parse_module(source: str, filename: str) -> ast.Module:
    """输入：源码文本与文件名。输出：模块 AST。

    预期行为：先带 `type_comments=True` 解析——不带这个开关时 `# type:` 注释根本不会
    挂到 `type_comment` 字段上，第 4 类写法就查不出来。注释写歪了（`# type:` 后面写人话）
    会直接抛 SyntaxError，那就退回不带类型注释再解析一遍：歪注释里的 `Any` 认不出来，
    但绝不因此崩掉整条测试。
    """
    try:
        return ast.parse(source, filename=filename, type_comments=True)
    except SyntaxError:
        return ast.parse(source, filename=filename)


def _string_annotation_hits(annotation: ast.AST) -> list[int]:
    """输入：一个注解子树。输出：其中字符串注解里出现 `Any` 的行号。

    预期行为：字符串注解在 AST 里只是 `Constant`，得把它的值再解析一遍才看得见 `Any`。
    行号取那个字符串常量本身的行号——内层解析出来的行号是相对字符串自己的，没法用。
    """
    found: list[int] = []
    for sub in ast.walk(annotation):
        if not isinstance(sub, ast.Constant) or not isinstance(sub.value, str):
            continue
        parsed = _parse_annotation_text(sub.value)
        if parsed is not None and _hits_in(parsed):
            found.append(sub.lineno)
    return found


def _string_sites(tree: ast.AST) -> list[int]:
    """输入：整棵 AST。输出：字符串注解与 `# type:` 注释里出现 `Any` 的行号。

    预期行为：遍历所有「能挂注解」的节点取注解子树；`# type:` 注释不挂在注解字段上，
    而是躺在语句节点的 `type_comment` 字符串字段里，单独解析，行号用语句自己的行号。
    """
    found: list[int] = []
    for node in ast.walk(tree):
        annotation: ast.AST | None = None
        if isinstance(node, ast.arg):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        elif isinstance(node, ast.AnnAssign):
            annotation = node.annotation
        if annotation is not None:
            found.extend(_string_annotation_hits(annotation))
        comment = getattr(node, "type_comment", None)
        if isinstance(comment, str):
            parsed = _parse_annotation_text(comment)
            if parsed is not None and _hits_in(parsed):
                found.append(node.lineno)
    return found


def _getattr_literal_hits(tree: ast.AST) -> list[int]:
    """输入：整棵 AST。输出：`getattr(..., "Any")` 这种字面量取属性的行号。

    预期行为：只认第二个实参是字面量 `"Any"` 的调用；动态名字静态查不到，不假装能查。
    """
    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "getattr"):
            continue
        if len(node.args) >= 2:
            name = node.args[1]
            if isinstance(name, ast.Constant) and name.value == _BANNED:
                found.append(node.lineno)
    return found


def any_sites_in_source(source: str, filename: str = "<source>") -> list[int]:
    """输入：Python 源码文本与其文件名。输出：显式 `Any` 出现的行号（去重升序）。

    预期行为：文档串、普通注释、当数据用的字符串一律不算；其余六类写法全算（见模块
    文档）。返回行号而不是布尔，方便报错时直接指到行。
    """
    tree = _parse_module(source, filename)
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _ANY_MODULES:
            lines.extend(node.lineno for alias in node.names if alias.name in {_BANNED, "*"})
        elif isinstance(node, ast.Name) and node.id == _BANNED:
            lines.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == _BANNED:
            lines.append(node.lineno)
    lines.extend(_string_sites(tree))
    lines.extend(_getattr_literal_hits(tree))
    return sorted(set(lines))


def any_sites(path: Path) -> list[int]:
    """输入：一个 `*.py` 路径。输出：显式 `Any` 出现的行号（去重升序）。

    预期行为：按 UTF-8 读文件再交给 `any_sites_in_source`；文件读不了就抛（不吞错）。
    """
    return any_sites_in_source(path.read_text(encoding="utf-8"), filename=str(path))


#: 干净源码：文档串和注释里提 `Any` 不算，因为 AST 看不见它们。
_CLEAN_SOURCE = '''
"""这个文档串里写着 Any，但它不该被算进去。"""
# 这行注释里也写着 Any。


def add(a: int, b: int) -> int:
    """返回两数之和。"""
    return a + b
'''

#: 当数据用的字符串里有 `Any`，也不该算——它不是注解。
_DATA_ONLY_SOURCE = 'NAME = "Any"\nTEXT = "from typing import Any"\n'


def test_clean_source_has_no_sites() -> None:
    """输入：文档串与注释里提 `Any` 的源码。输出：无。预期行为：一个行号都不报。"""
    assert any_sites_in_source(_CLEAN_SOURCE) == []


def test_data_only_strings_are_ignored() -> None:
    """输入：把 `Any` 当数据字符串用的源码。输出：无。预期行为：不当成注解，不报。"""
    assert any_sites_in_source(_DATA_ONLY_SOURCE) == []


def test_broken_type_comment_does_not_crash() -> None:
    """输入：`# type:` 后面写了句人话的源码。输出：无。

    预期行为：解析不了就跳过，不抛异常——否则一个手滑的注释会把整条测试炸掉。
    """
    assert any_sites_in_source("x = 1  # type: 这里写的不是类型\n") == []


@pytest.mark.parametrize(
    ("source", "expect"),
    [
        ("from typing import Any\n", [1]),
        ("from typing import Any as Whatever\n", [1]),
        ("from typing_extensions import Any\n", [1]),
        ("from typing import *\n", [1]),
        ("from typing_extensions import *\n", [1]),
        ("import typing\n\ndef f(x: typing.Any) -> None: ...\n", [3]),
        ("import typing as t\n\nx: t.Any = 1\n", [3]),
        ("def f(x: Any) -> None: ...\n", [1]),
        ("def f(x: 'Any') -> None: ...\n", [1]),
        ("def f(x: list['Any']) -> None: ...\n", [1]),
        ("def f(x: 'list[typing.Any]') -> None: ...\n", [1]),
        ("x = []  # type: list[Any]\n", [1]),
        ("import typing\n\nT = getattr(typing, 'Any')\n", [3]),
        ("import typing\n\nAny = 1\n", [3]),
    ],
)
def test_banned_shapes_are_found(source: str, expect: list[int]) -> None:
    """输入：一段含 `Any` 的源码与期望行号。输出：无。预期行为：报出的行号与期望一致。"""
    assert any_sites_in_source(source) == expect


def test_any_sites_reads_file(tmp_path: Path) -> None:
    """输入：临时目录里一个含 `Any` 的文件。输出：无。预期行为：路径版与文本版结果一致。"""
    target = tmp_path / "sample.py"
    target.write_text("import typing\n\nAlias = typing.Any\n", encoding="utf-8")
    assert any_sites(target) == [3]


@pytest.mark.parametrize("rel", _sources())
def test_no_explicit_any(rel: str) -> None:
    """输入：一个相对路径。输出：无。

    预期行为：文件里不许出现显式 `Any`；报了就把行号一起说出来，方便直接去改。
    """
    if rel in EXEMPT:
        pytest.skip(EXEMPT[rel])
    sites = any_sites(Path(rel))
    assert not sites, (
        f"{rel} 第 {sites} 行还有显式 Any——换成 JsonValue（JSON）/ object（任意值）/ 具体类型"
    )
