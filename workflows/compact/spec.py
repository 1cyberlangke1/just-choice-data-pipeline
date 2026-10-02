"""会话上下文压缩的真值规格：一组注入特征 + 这条样本的问法编号。

输入/输出：只有不可变数据类与纯函数；不含领域词表（词表在 `locales/`）。
"""

from __future__ import annotations

from dataclasses import dataclass

from jc.jsontypes import JsonValue
from jc.spec_dict import tagged_spec_dict

from .features import Labels, TagSet


@dataclass(frozen=True)
class CompactSpec:
    """一条样本的真值规格。

    字段：`tags` 注入特征；`phrasing` 问法编号；`split` 切分名；`seq` 样本序号。
    """

    tags: TagSet
    phrasing: int
    split: str
    seq: int

    @property
    def labels(self) -> Labels:
        """输入：无。输出：这一组特征对应的全部金标签。"""
        return self.tags.label()

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；
        进度日志按 run 目录分文件，所以不同 run_seed 之间不会串。
        """
        return f"compact-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"compact-{self.split}-{self.seq:07d}-trace"


def spec_to_dict(spec: CompactSpec) -> dict[str, JsonValue]:
    """输入：规格。输出：可 JSON 序列化的字典（写进 manifest 与调试，不进样本行）。"""
    return tagged_spec_dict(tags=spec.tags, phrasing=spec.phrasing, split=spec.split, seq=spec.seq)
