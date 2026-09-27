"""游戏与模拟控制 workflow：构造式（一屏局面与全部判断都由规则算出来，零 token）。

导入本包即自注册（`jc.workflow.load("sim")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
