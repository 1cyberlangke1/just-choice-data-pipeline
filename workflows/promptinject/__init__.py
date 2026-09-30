"""提示注入（promptinject）workflow：注入判据由代码写，不可信内容由后端写，命中与否由代码按字面算。

导入本包即自注册（`jc.workflow.load("promptinject")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
