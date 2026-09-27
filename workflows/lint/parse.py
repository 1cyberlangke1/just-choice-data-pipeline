"""行级规则检查 workflow 的读法：从 state 把规则与片段读回来。

输入：state 文本 + 语言字典。输出：`LintReading`（读出来的规则 + 逐行原文 + 逐行命中标记）。
预期行为：只按文本读——规则靠**规则短语**认（必须恰好命中一个），片段靠 `LINE_PREFIX`
逐行剥掉行号前缀，命中与否用 `spec.matches` 重算。所以判别器能把答案从 state 独立重算，
`reparse` 也不需要语言文件（它读合并表）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .render import as_phrases
from .spec import LINE_PREFIX, RULES, matches


@dataclass(frozen=True)
class LintReading:
    """从 state 读回来的一份片段：规则 + 逐行原文 + 逐行命中标记。"""

    rule: str
    lines: tuple[str, ...]
    matched: tuple[bool, ...]

    @property
    def hits(self) -> int:
        """输入：无。输出：命中行数。"""
        return sum(1 for flag in self.matched if flag)

    def matched_lines(self) -> tuple[str, ...]:
        """输入：无。输出：命中的那些行的原文（按片段顺序）。"""
        return tuple(line for line, flag in zip(self.lines, self.matched, strict=True) if flag)


def rule_of(state: str, loc: dict[str, Any]) -> str:
    """输入：state 文本、语言字典。输出：规则名。

    预期行为：规则短语必须**恰好命中一个**——命中 0 个（文本没说清按哪条规则查）或
    2 个以上（规则短语不互斥）都直接报错，不猜一个。
    """
    hits = [
        rule
        for rule in RULES
        if any(phrase in state for phrase in as_phrases(loc["rule"][rule]))
    ]
    if len(hits) != 1:
        raise ValueError(f"规则短语命中 {len(hits)} 条规则：{sorted(hits)}")
    return hits[0]


def read_state(state: str, loc: dict[str, Any]) -> LintReading:
    """输入：state 文本、语言字典。输出：读回来的片段。

    预期行为：前两行是查询句与抬头（不参与片段），从第三行起每行都必须有行号前缀，
    且行号连续（从 1 起）。前缀对不上就报错——不静默跳过，否则片段会悄悄少一行，
    `line` 那一族题的数量与答案跟着错。
    """
    if not isinstance(state, str):
        raise TypeError(f"state 必须是字符串，收到 {type(state).__name__}")
    rows = state.split("\n")
    if len(rows) < 3:
        raise ValueError(f"state 至少要有三行（查询句 + 抬头 + 片段），收到 {len(rows)} 行")
    rule = rule_of(state, loc)
    lines: list[str] = []
    for index, row in enumerate(rows[2:], start=3):
        found = LINE_PREFIX.match(row)
        if found is None:
            raise ValueError(f"第 {index} 行没有行号前缀：{row!r}")
        expected = index - 2
        if int(found.group(1)) != expected:
            raise ValueError(f"第 {index} 行的行号是 {found.group(1)}，应为 {expected}")
        lines.append(found.group(2))
    return LintReading(rule, tuple(lines), tuple(matches(rule, line) for line in lines))
