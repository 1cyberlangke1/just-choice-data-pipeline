"""生产代码里不许硬写语言：`locales.load("zh")` 这种写法必须绝迹。

为什么要有：判别器要按语言认 state 里的行首标记（「拟执行命令：」/「Proposed command: 」），
写死一种语言会让另一种语言的样本解析出**空串**——而空串常常恰好蒙对（「命令里没有 cmdlet」
= 通过），于是 bug 只在一部分判别器上显形、接受率悄悄掉一半，测试全绿。
真机实测：shell 的方言 / 破坏性 / 提权三个判别器硬写 zh，en 侧接受率 4/10，
八个拒稿的理由里全是 `命令 ''`（范式 §9.30）。

输入：`src/` 与 `workflows/` 下的全部 `*.py`。
输出：无（断言）。
预期行为：用 **AST** 扫，注释与文档串里的 `load("...")` 不算；只查「`X.load(<字符串字面量>)`」。
语言**必须**由调用方传进来（`self.locale` / 形参）；带名字的常量（如 `locales.DEFAULT`）
不算硬写——它有一个名字、能被全局改，也搜得到。

已知漏网：先 `LOCALE = "zh"` 再 `load(LOCALE)`。那要类型流分析，静态查不到，不假装能查。
"""

from __future__ import annotations

import ast
from pathlib import Path

#: 扫描入口：目录递归扫 `*.py`。
ROOTS: tuple[str, ...] = ("src", "workflows")

#: 豁免文件：键是相对路径，值是理由。空 = 一个都不豁免。
EXEMPT: dict[str, str] = {}


def _sources() -> list[str]:
    """输入：无。输出：待检查文件的相对路径（排序后）。

    预期行为：递归取 `*.py` 并跳过 `__pycache__`——与 `test_no_any` 同一套取法，
    两把尺子扫的必须是同一批文件。
    """
    found: list[str] = []
    for entry in ROOTS:
        for child in Path(entry).rglob("*.py"):
            if "__pycache__" in child.parts:
                continue
            found.append(child.as_posix())
    return sorted(found)


def literal_load_sites(source: str, filename: str = "<source>") -> list[int]:
    """输入：Python 源码文本与其文件名。输出：硬写语言的 `load(...)` 的行号（去重升序）。

    预期行为：只看 `X.load("字面量")` 这一种形状。实参不是字符串字面量（变量、常量名、
    属性取值）一律放过——那种写法的语言来自别处，是这一层的正当输入。
    """
    tree = ast.parse(source, filename=filename)
    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "load"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.append(node.lineno)
    return sorted(set(found))


def test_literal_argument_is_found() -> None:
    """输入：写死语言的源码。输出：无。预期行为：报出那一行。"""
    assert literal_load_sites('loc = locales.load("zh")\n') == [1]
    assert literal_load_sites('x = 1\nloc = locales_mod.load("en")\n') == [2]


def test_named_constant_and_variable_are_allowed() -> None:
    """输入：用常量名 / 变量当实参的源码。输出：无。预期行为：都不报。"""
    assert literal_load_sites("loc = locales.load(lang)\n") == []
    assert literal_load_sites("loc = locales.load(locales.DEFAULT)\n") == []
    assert literal_load_sites("loc = locales.load(self.locale)\n") == []


def test_docstring_and_comment_are_ignored() -> None:
    """输入：文档串与注释里写着 `load("zh")` 的源码。输出：无。预期行为：一个行号都不报。"""
    source = '"""导入本包即自注册（`jc.workflow.load("shell")` 可用）。"""\n# load("zh")\n'
    assert literal_load_sites(source) == []


def test_no_literal_locale_load() -> None:
    """输入：无（扫 `_sources()` 列出的每个文件）。输出：无。

    预期行为：全仓库不许出现 `load("字面量")`；有就把**每一个**违规点聚合在一条断言里
    （路径 + 行号 + 修法），一次看全。

    为什么不按文件参数化：被测对象是「这个仓库」，文件只是遍历的粒度，不是 485 个各自
    独立的被测对象；同族的 `test_locale_templates.py` 与 `test_no_any.py` 都是循环写法。
    """
    offenders: list[str] = []
    for rel in _sources():
        if rel in EXEMPT:
            continue
        sites = literal_load_sites(Path(rel).read_text(encoding="utf-8"), filename=rel)
        if sites:
            offenders.append(
                f"{rel} 第 {sites} 行把语言写死了——改成从调用方取（`self.locale` / 形参），"
                "否则另一种语言的样本会解析出空串"
            )
    assert not offenders, "\n".join(offenders)
