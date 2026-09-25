"""邮件安全分诊 workflow：特征注入式（标签先行，模型只写文本外壳）。

导入本包即自注册（`jc.workflow.load("mail")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
