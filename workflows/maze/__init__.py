"""迷宫 workflow：把迷宫渲染成多种 state 格式，并派生导航类选择题。

导入本包即自注册（`jc.workflow.load("maze")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
