"""评测与基准 workflow：构造式（一份评测记录与全部判断都由规则算出来，零 token）。

导入本包即自注册（`jc.workflow.load("bench")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
