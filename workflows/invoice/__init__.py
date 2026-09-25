"""单据字段抽取 workflow：抽取式（答案从素材里抽出来，程序可校验）。

导入本包即自注册（`jc.workflow.load("invoice")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
