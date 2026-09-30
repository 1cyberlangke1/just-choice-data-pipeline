"""测试影响选择（testimpact）workflow：覆盖表由代码写，改动由后端写，该跑哪些套件按字面算。

导入本包即自注册（`jc.workflow.load("testimpact")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
