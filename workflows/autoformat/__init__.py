# 结构恢复（autoformat）workflow 包。
"""结构恢复（autoformat）workflow：块的顺序与类型由代码写，每一块的文字由后端写。

导入本包即自注册（`jc.workflow.load("autoformat")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
