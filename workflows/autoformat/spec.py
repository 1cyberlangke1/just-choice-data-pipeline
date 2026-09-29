"""结构恢复的真值规格：文档组合 + 块计划 + 切分信息。

输入/输出：只有不可变数据类与纯函数；不含领域词表（文案全在 `locales/`，正文由后端写）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jc.jsontypes import JsonValue

from .features import DocCase, DocLabels, DocPlan, label_for


@dataclass
class DocSpec:
    """一条样本的真值规格。

    字段：`case` 文档组合；`split` 切分名；`seq` 样本序号；`phrasing` 问法编号；`plan` 块计划；
    `texts` 后端写的那几段文字。

    **为什么这个类不是 frozen 的**：`texts` 只有渲染阶段才拿得到（`sample` 时还不知道模型会写
    什么），而「并段」那一族的题面要引用**行号**——行号由折行结果决定，也就由 `texts` 决定。协议里
    `ask(spec, ...)` 拿不到 state，所以让 `render` 把文字**回填**进同一份 spec（同一条样本的
    `render` 与 `ask` 由管线在同一个线程里前后调用，不存在跨线程共享）。
    """

    case: DocCase
    split: str
    seq: int
    phrasing: int
    plan: DocPlan
    #: 渲染阶段回填：按块顺序的一段段文字（`sample` 时是空表）。
    texts: list[str] = field(default_factory=list)

    @property
    def labels(self) -> DocLabels:
        """输入：无。输出：这一条的全部金标签（由块计划推出，后端碰不到）。"""
        return label_for(self.plan)

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；进度日志按 run 目录分文件，
        所以不同 run_seed 之间不会串。
        """
        return f"autoformat-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"autoformat-{self.split}-{self.seq:07d}-trace"


def spec_to_dict(spec: DocSpec) -> dict[str, JsonValue]:
    """输入：规格。输出：可 JSON 序列化的字典（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.as_dict(),
        "split": spec.split,
        "seq": spec.seq,
        "phrasing": spec.phrasing,
        "kinds": list(spec.plan.kinds()),
        "callout_kinds": list(spec.plan.callout_kinds()),
        "list_flags": list(spec.plan.list_flags()),
    }
