"""全仓库：workflow 的「草稿契约检查」必须抛 `jc.workflow.DraftError`。

为什么要有：`DraftError` 带 `retryable_for_case = True`，管线看到就把这一条**重渲染**一次
（真机上「模型没照契约写」占四到六成，单发等于白丢样本，见 PLAN 10.84 / 范式 §9.27）；
裸 `ValueError` 是单发。这条把口径钉死，防止新写的 check 又退回裸 `ValueError`。

输入：`workflows/*/render.py` 与注册表里的每个 workflow。
输出：无（断言）。
预期行为：用 **AST** 扫，不用 grep——grep 会把注释、文档、字符串里的名字也算进去。
名字以 `check_` 开头的函数就是「草稿契约检查」（`check_draft` / `check_command` /
`check_parts` / `check_official_domain`），它们：

1. 不许抛裸 `ValueError`；
2. 至少要有一处 `raise DraftError(...)`；
3. `needs` 非空的 workflow（真调后端的那些）必须**有**这样的函数。

不在此列的：`workflows/*/judges.py` 里的 `check_*`（判别器返回 `Verdict`、不抛异常），
以及 `render.py` 里不叫 `check_` 的函数（语言文件读错、未知取值这类属于生成器 bug，该单发）。

两条扫描都收成循环：它们查的是**同一段 AST 扫描**，51 个 workflow 摊成 51 条参数化用例
只是把同一件事报 51 遍。失败信息攒在 `bad` 里一起报，点名到文件、函数、行号。
"""

from __future__ import annotations

import ast
from pathlib import Path

from jc.workflow import discover, load

#: 契约检查函数的前缀——名字就是口径。
PREFIX: str = "check_"

#: 注册表里的 workflow 名（导入即注册）。
WORKFLOWS = discover()


def _render_files() -> list[Path]:
    """输入：无。输出：`workflows/*/render.py`（按名字排序）。"""
    return sorted(path for path in Path("workflows").glob("*/render.py"))


def _check_functions(path: Path) -> list[ast.FunctionDef]:
    """输入：一个 `*.py` 路径。输出：其中名字以 `check_` 开头的顶层函数定义。

    预期行为：只认顶层函数（`check_draft` 那一族都在顶层）；嵌套函数不算。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith(PREFIX)
    ]


def _raised_names(func: ast.FunctionDef) -> list[str]:
    """输入：一个函数定义。输出：它抛出的异常类名（`raise X(...)` 与 `raise X` 都算）。

    预期行为：`raise` 后面直接给名字或调用都认；其它形态（变量、属性链）原样跳过——
    静态查不到的属于明知漏网，不假装能查。
    """
    names: list[str] = []
    for sub in ast.walk(func):
        if not isinstance(sub, ast.Raise) or sub.exc is None:
            continue
        exc: ast.expr = sub.exc.func if isinstance(sub.exc, ast.Call) else sub.exc
        if isinstance(exc, ast.Name):
            names.append(exc.id)
        elif isinstance(exc, ast.Attribute):
            names.append(exc.attr)
    return names


def test_there_are_checks_to_scan() -> None:
    """输入：无。输出：无。预期行为：扫到的契约检查函数够多（否则这条测试是空转）。"""
    found = [func for path in _render_files() for func in _check_functions(path)]
    assert len(found) >= 17, f"只扫到 {len(found)} 个 check_* 函数，口径变了就改这条测试"


def test_contract_checks_raise_draft_error() -> None:
    """输入：无（`workflows/*/render.py` 现场列）。输出：无。

    预期行为：它每个 `check_*` 函数都不抛裸 `ValueError`，且这些函数**合起来**至少抛一处
    `DraftError`（有的函数只是把判据委托给同模块的另一个 check，自己一处都不抛——
    rerank 的 `check_draft` 就是这样）。报了就把函数名与行号一起说出来，方便直接去改。
    """
    bad: list[str] = []
    for path in _render_files():
        funcs = _check_functions(path)
        if not funcs:
            # 构造式 workflow（bench / functions / invoice / match / maze / sim / skillpick）
            # 的文本不来自模型，没有草稿契约可查——由下面那条「needs 非空必须有 check」兜住。
            continue
        raised: set[str] = set()
        for func in funcs:
            names = _raised_names(func)
            raised.update(names)
            if "ValueError" in names:
                bad.append(
                    f"{path}:{func.lineno} 的 {func.name} 抛了裸 ValueError——草稿契约失败要抛 "
                    "jc.workflow.DraftError（带 retryable_for_case），否则管线单发、白丢样本"
                )
        if "DraftError" not in raised:
            bad.append(f"{path} 的 check_* 一处 DraftError 都没有——管线不会重渲染")
    assert not bad, bad


def test_workflows_that_need_a_backend_have_a_contract_check() -> None:
    """输入：无（注册表现场列）。输出：无。

    预期行为：真调后端的 workflow（`needs` 非空）必须有 `check_*` 函数——模型写出来的文本
    没人逐条查，契约就只是提示词里的一句话。
    """
    bad: list[str] = []
    for name in WORKFLOWS:
        if not load(name).needs:
            continue
        path = Path("workflows") / name / "render.py"
        if not path.exists():
            bad.append(f"{name} 需要后端，却没有 {path}")
        elif not _check_functions(path):
            bad.append(f"{name} 需要后端，但 {path} 里一个 check_* 函数都没有")
    assert not bad, bad
