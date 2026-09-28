"""评测与基准 workflow 的渲染：把真值规格写成一屏评测记录。

输入：`BenchSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「标题 → 集 / 门槛 → 空行 → 五行记录」，每行四格：第一格是参考答案，
后面每个候选一格，格子里是「候选名 + 答案字母」（名字与字母之间用一个固定的分隔符连，
不是冒号——冒号在中英两种语言里写法不同，而这一格是**两种语言都要反解**的）。
**抬头两格与五行四格就是判据的全部输入**。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue



from .spec import BenchRow, BenchSpec


def field_line(label: str, value: str, loc: dict[str, JsonValue]) -> str:
    """输入：标签、值、语言字典。输出：`标签：值` 一行。

    预期行为：分隔符统一从 `meta.colon` 取——反解要求标签后面紧跟分隔符，
    换语言时不会因为冒号写法不同而认不出来。
    """
    return f"{label}{loc['meta']['colon']}{value}"


def cell_text(name: str, letter: str, loc: dict[str, JsonValue]) -> str:
    """输入：名字（候选名或参考格标签）、字母、语言字典。输出：一格。

    预期行为：格内用 `meta.pair_sep` 把「名字」与「字母」分开。用独立的分隔符而不是冒号，
    是因为冒号在中英两种语言里写法不同（`：` / `: `），而这一格要能**跨语言反解**；
    候选名里不许出现这个分隔符（`test_locales.py` 锁住）。
    """
    return f"{name}{loc['meta']['pair_sep']}{letter}"


def row_line(index: int, row: BenchRow, loc: dict[str, JsonValue]) -> str:
    """输入：序号（从 1 起）、一行记录、语言字典。输出：记录那一行。

    预期行为：第一格是参考答案，后面按**列序**排三个候选格。这四格就是 `hit` / `count` /
    `pass` / `best` / `gap` 五族的全部输入，所以一行里必须都写出来。
    """
    cells = [cell_text(str(loc["cell"]["ref"]), row.ref, loc)]
    cells += [cell_text(str(loc["cand"][cell.cand]), cell.letter, loc) for cell in row.cells]
    return f"[{index}] " + str(loc["meta"]["cell_sep"]).join(cells)


def build(spec: BenchSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：整屏记录文本。

    预期行为：每一格都只从**规格里已经算好的东西**取（集号、门槛、五行四格），渲染器不做
    任何判断——判断全在 `spec` 的规则函数里。
    """
    meta = loc["meta"]
    labels = loc["label"]
    lines = [
        str(meta["title"]),
        field_line(str(labels["request"]), spec.request_id, loc),
        field_line(str(labels["bar"]), str(spec.bar), loc),
        "",
    ]
    for index, row in enumerate(spec.rows, start=1):
        lines.append(row_line(index, row, loc))
    return "\n".join(lines)
