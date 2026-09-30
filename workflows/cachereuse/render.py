"""缓存复用判定（cachereuse）workflow 的渲染：把真值规格写成一屏「两条操作轨迹」。

输入：`ReuseSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「抬头（请求号）→ 来源一行 → 缓存轨迹一块 → 当前轨迹一块」，每块是「抬头 + 若干
带编号的步骤行」。

关键约束：**等价步在两条轨迹里必须是同一个键**——`verbatim` 两侧都取第 0 种写法（逐字相同），
`synonym` 缓存侧取第 0 种、当前侧取第 1 种（写法不同但归一化后相同）。反解侧的判据是
「归一化后动作与元素都对得上」，所以两种写法都算等价。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from collections.abc import Sequence

from . import spec as spec_mod
from .spec import ReuseSpec, Step


def spelling_index(spelling: str) -> int:
    """输入：写法名（`verbatim` / `synonym`）。输出：当前轨迹该取第几种写法（0 / 1）。

    预期行为：未知写法当场报错——悄悄退回 0 会把「同义写法」这一类样本渲染成逐字相同，
    判据那边看不出差别，整条维就白设了。
    """
    if spelling == spec_mod.VERBATIM:
        return 0
    if spelling == spec_mod.SYNONYM:
        return 1
    raise ValueError(f"未知写法：{spelling!r}（可用：{spec_mod.SPELLINGS}）")


def step_text(loc: dict[str, JsonValue], action: str, target: str, spelling: int) -> str:
    """输入：语言字典、动作键、元素键、写法下标。输出：这一步的可读原文。

    预期行为：动作与元素各查各的写法表，中间用该语言的引号对包住元素名（引号是**结构**，
    反解按它切分，所以它进语言文件而不是硬编码在反解里）。
    """
    meta = loc["meta"]
    verb = loc["action"][action][spelling]
    noun = loc["target"][target][spelling]
    return f"{verb}{meta['quote_open']}{noun}{meta['quote_close']}"


def side_lines(steps: Sequence[Step], loc: dict[str, JsonValue], spelling: int) -> tuple[str, ...]:
    """输入：步骤序列、语言字典、这一侧用的写法下标。输出：带编号与缩进的行。

    预期行为：编号从 1 起、点号是**结构**（中英都用 `N.`），缩进两格只是给人看的——
    反解按 `strip()` 后的内容认编号。
    """
    return tuple(
        f"  {index}. {step_text(loc, action, target, spelling)}"
        for index, (action, target) in enumerate(steps, start=1)
    )


def build(spec: ReuseSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：`state` 全文。

    预期行为：两块轨迹的抬头用方括号包住；来源那一行只写这一条样本的来源名（它只改文本、
    不影响任何标签）。
    """
    meta = loc["meta"]
    case = spec.case
    cached, current = spec_mod.traces_of(case)
    lines = [
        f"{meta['request']} {spec.query_id}",
        f"{meta['source']}{meta['sep']}{loc['flavor'][case.flavor]}",
        f"[{meta['cached']}]",
        *side_lines(cached, loc, 0),
        f"[{meta['current']}]",
        *side_lines(current, loc, spelling_index(case.spelling)),
    ]
    return "\n".join(lines) + "\n"
