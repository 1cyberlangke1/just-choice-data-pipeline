"""模型路由与降级 workflow：候选目录与门槛由代码写，用户请求由后端写。

导入本包即自注册（`jc.workflow.load("modelroute")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
