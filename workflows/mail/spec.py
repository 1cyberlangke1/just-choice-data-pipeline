"""邮件 workflow 的真值规格：一组注入特征 + 这条样本的写法编号。

输入/输出：只有不可变数据类与纯函数；不含领域词表（词表在 `locales/`）。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from dataclasses import dataclass


from . import features
from .features import Labels, TagSet


@dataclass(frozen=True)
class MailSpec:
    """一条样本的真值规格。

    字段：`tags` 注入特征；`phrasing` 问法 / 写法编号；`split` 切分名；`seq` 样本序号。
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
    def link_path(self) -> str:
        """输入：无。输出：这一条的链接地址要用哪个路径段（见 `features.link_path_at`）。

        预期行为：纯派生、不进构造函数——它是**渲染期变量**，不是数据维度；按样本序号取，所以
        同一批里逐条不同。清单说「没有链接」时这个值不会出现在提示词里。
        """
        return features.link_path_at(self.seq)

    @property
    def link_ip(self) -> str:
        """输入：无。输出：这一条的链接地址要用哪个 IP（见 `features.link_ip_at`）。

        预期行为：同上——纯派生、按样本序号取，池里全是文档网段。
        """
        return features.link_ip_at(self.seq)

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；
        进度日志按 run 目录分文件，所以不同 run_seed 之间不会串。
        """
        return f"mail-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"mail-{self.split}-{self.seq:07d}-trace"


def spec_to_dict(spec: MailSpec) -> dict[str, JsonValue]:
    """输入：规格。输出：可 JSON 序列化的字典（写进 manifest 与调试，不进样本行）。"""
    return {
        "tags": spec.tags.as_dict(),
        "link_target": spec.tags.link_target,
        "anchor": spec.tags.anchor,
        "phrasing": spec.phrasing,
        "split": spec.split,
        "seq": spec.seq,
    }
