"""函数调用 workflow 的读法：从请求文本把工具与参数读回来。

输入：state 文本 + 语言字典。输出：`CallPlan`（读出来的绑定）。`alias` 与 `phrasing` 读不出来，
给 `ticker` / `0`——它们只影响措辞，不影响答案。
预期行为：只按文本读——工具靠**意图短语**认（必须恰好命中一个），参数靠**值短语 / flag 短语**认，
认不出就是「没提到」。所以判别器能把答案从 state 独立重算；构造写歪与渲染写歪会变成同一类错误，
不会两边各错一半还互相印证。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue



from .render import as_phrases, flag_phrases, value_phrases
from .spec import (
    KIND_FLAG,
    KIND_SET,
    TOOLS,
    Binding,
    CallPlan,
    phrase_key_of,
)


def tool_of(state: str, loc: dict[str, JsonValue]) -> str:
    """输入：请求文本、语言字典。输出：工具名。

    预期行为：意图短语必须**恰好命中一个**——命中 0 个（文本没说清要干什么）或 2 个以上
    （意图短语不互斥）都直接报错，不猜一个。
    """
    hits = [
        tool
        for tool, intent in loc["intent"].items()
        if any(phrase in state for phrase in as_phrases(intent))
    ]
    if len(hits) != 1:
        raise ValueError(f"意图短语命中 {len(hits)} 个工具：{sorted(hits)}")
    return hits[0]


def stated_values(state: str, name: str, values: tuple[str, ...], loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：文本、参数名、取值表、语言字典。输出：文本里命中的取值（0 个或多个，按取值表顺序）。

    预期行为：**两种别名写法都认**——请求里写代码（`NVDA`）还是写中文名（`英伟达`）都要读得出来。
    """
    key = phrase_key_of(name)
    return tuple(
        value
        for value in values
        if any(phrase in state for phrase in value_phrases(loc, key, value))
    )


def read_request(state: str, loc: dict[str, JsonValue]) -> CallPlan:
    """输入：请求文本、语言字典。输出：从文本读出来的绑定计划。

    预期行为：逐参数读——`choice` 命中多个取值直接报错（短语表不互斥就是构造 bug），
    `set` 读出一组成员，`flag` 看它的出现短语在不在。`stated` 就是「读出来了没有」。
    """
    tool = tool_of(state, loc)
    bindings: list[Binding] = []
    for arg in TOOLS[tool]:
        if arg.kind == KIND_FLAG:
            stated = any(phrase in state for phrase in flag_phrases(loc, arg.name))
            bindings.append(Binding(arg.name, arg.kind, stated, value="on" if stated else ""))
            continue
        found = stated_values(state, arg.name, arg.values, loc)
        if arg.kind == KIND_SET:
            bindings.append(Binding(arg.name, arg.kind, bool(found), members=found))
            continue
        if len(found) > 1:
            raise ValueError(f"参数 {arg.name} 命中多个取值：{sorted(found)}")
        bindings.append(
            Binding(arg.name, arg.kind, bool(found), value=found[0] if found else "")
        )
    return CallPlan(tool, tuple(bindings), "ticker", 0)
