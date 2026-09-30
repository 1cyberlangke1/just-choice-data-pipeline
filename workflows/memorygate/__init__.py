"""记忆写入门控（memorygate）workflow：判据表由代码写，候选记忆由后端写，命中与否按字面算。

导入本包即自注册（`jc.workflow.load("memorygate")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
