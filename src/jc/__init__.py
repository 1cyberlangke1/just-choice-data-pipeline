"""jc：JEV 风格选择题数据合成管线（**通用层**）。

分层：contract（契约常量）→ jsonio（严格读写）→ validate（契约校验）
→ workflow（workflow 协议与注册表）→ assemble（组装）→ judge（判别器协议）
→ pipeline（编排）→ cli。

本包**不认识任何具体 workflow**：workflow 放在仓库根的 `workflows/` 下，自注册后由
`jc.workflow.load(name)` 动态加载。领域词表只允许出现在 `workflows/` 里。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
