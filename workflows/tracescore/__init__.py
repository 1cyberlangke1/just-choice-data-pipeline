"""代理轨迹自评（tracescore）workflow：信号表由代码写，执行轨迹由后端写，命中与否按字面算。

导入本包即自注册（`jc.workflow.load("tracescore")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
