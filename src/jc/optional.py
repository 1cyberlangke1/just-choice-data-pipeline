"""可选依赖（`[llm]` extra）的名字与「怎么装」的提示。

输入：import 失败时抛出的异常。输出：一句人话（指名缺哪个包、怎么装），或者 None（缺的不是
可选依赖）。
预期行为：核心层与纯构造 workflow 只用标准库，所以**没装 `[llm]` 不该让整条 CLI 起不来**；
真要走到需要它的地方（`build_backend`、某个走 LLM 的 workflow），报的必须是人话——
`pip install -e ".[llm]"`，而不是裸的 `ModuleNotFoundError: No module named 'instructor'`。

为什么单独一个模块：`jc.backend`（后端实现）与 `jc.workflow`（注册表扫描）都要说这句话，
而注册表不该 import 后端——分层规矩是「通用层不静态依赖任何具体实现」。
"""

from __future__ import annotations

from typing import Final

#: `[llm]` extra 装的东西：`instructor`（结构化输出）、`openai`（端点）、`pydantic`
#: （响应模型）、`pyrate-limiter`（限速）、`tenacity`（退避）。核心层与纯构造 workflow 一个都不碰。
LLM_PACKAGES: Final[frozenset[str]] = frozenset(
    {"instructor", "openai", "pydantic", "pyrate_limiter", "tenacity"}
)

#: 装法：写进提示里，让人能直接照抄（Windows 上 `.venv\Scripts\python.exe -m pip ...` 同理）。
INSTALL_HINT: Final[str] = 'python -m pip install -e ".[llm]"'


def missing_llm_package(exc: ModuleNotFoundError) -> str | None:
    """输入：`ModuleNotFoundError`。输出：缺的那个可选包名，或 None（缺的不是可选依赖）。

    预期行为：只看**顶层**包名——`instructor.core` 这种子模块导入失败也算 `instructor` 缺了；
    名字不在 `LLM_PACKAGES` 里就给 None，免得把别人的 `ModuleNotFoundError` 也认成自己的。
    """
    top = (exc.name or "").split(".")[0]
    return top if top in LLM_PACKAGES else None


def llm_extra_hint(exc: ModuleNotFoundError) -> str | None:
    """输入：`ModuleNotFoundError`。输出：人话提示，或 None（缺的不是可选依赖）。

    预期行为：只在缺的确实是 `[llm]` 里的包时才给提示——别的 `ModuleNotFoundError`
    （比如 workflow 名字写错了）不该被这句提示盖住，否则会指错方向。
    """
    package = missing_llm_package(exc)
    if package is None:
        return None
    return f"缺可选依赖 {package}（LLM 后端要的 [llm] extra）；装法：{INSTALL_HINT}"
