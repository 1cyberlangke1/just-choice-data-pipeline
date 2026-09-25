"""会议日程 workflow：混合式（构造骨架 + 模型补表述）。

导入本包即自注册（`jc.workflow.load("hybrid")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
