"""模型路由与降级 workflow：构造式（画像与全部判断都由规则表算出来，零 token）。

导入本包即自注册（`jc.workflow.load("modelroute")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
