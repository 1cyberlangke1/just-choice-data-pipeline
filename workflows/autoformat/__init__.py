"""结构恢复（autoformat）workflow：构造式（文档与全部判断都由规则算出来，零 token）。

导入本包即自注册（`jc.workflow.load("autoformat")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
