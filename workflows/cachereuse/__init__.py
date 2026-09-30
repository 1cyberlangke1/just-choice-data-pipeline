"""缓存复用判定（cachereuse）workflow：构造式（两条操作轨迹与全部判断都由代码算出来，零 token）。

导入本包即自注册（`jc.workflow.load("cachereuse")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
