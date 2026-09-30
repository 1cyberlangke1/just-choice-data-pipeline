"""聊天回复门控（chatgate）workflow：消息形状判据表、候选毛病表与处置档位表由代码写，一条消息与三条候选回复由后端写。

导入本包即自注册（`jc.workflow.load("chatgate")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
