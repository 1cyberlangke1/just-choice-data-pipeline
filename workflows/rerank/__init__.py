"""检索重排（rerank）workflow：构造式（查询、候选短名单与全部判断都由画像表算出来，零 token）。

导入本包即自注册（`jc.workflow.load("rerank")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
