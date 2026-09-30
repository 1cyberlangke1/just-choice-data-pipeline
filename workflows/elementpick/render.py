"""界面元素定位 workflow 的渲染：把真值规格写成一屏元素清单。

输入：`ElementSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「标题 → 屏号 → 动作 → 空行 → 六行元素表」，每行四格：
角色 · 可见性 · 可点性 · 文本。**动作那一行的值与元素表四格是判据的全部输入**。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from .spec import SHADOW_PREFIX, ElementRow, ElementSpec


def field_line(label: str, value: str, loc: dict[str, JsonValue]) -> str:
    """输入：标签、值、语言字典。输出：`标签：值` 一行。

    预期行为：分隔符统一从 `meta.colon` 取——反解要求标签后面紧跟分隔符，
    换语言时不会因为冒号写法不同而认不出来。
    """
    return f"{label}{loc['meta']['colon']}{value}"


def text_of(text_key: str, loc: dict[str, JsonValue]) -> str:
    """输入：文本键（标签键或 `shadow:<标签键>`）、语言字典。输出：那一格要写的原文。

    预期行为：近似文本 = 标签原文 + 后缀（`meta.shadow_suffix`），所以那段文本**真的**
    把标签原文当子串含在里面——`parse.decode_text` 是这条的镜像（按同一后缀剥回去）。
    渲染写歪会变成「反解认不出来」，而不是静默认成别的键。
    """
    if text_key.startswith(SHADOW_PREFIX):
        base = text_key[len(SHADOW_PREFIX) :]
        return f"{loc['label'][base]}{loc['meta']['shadow_suffix']}"
    return str(loc["label"][text_key])


def row_line(index: int, row: ElementRow, loc: dict[str, JsonValue]) -> str:
    """输入：序号（从 1 起）、一行元素、语言字典。输出：元素表那一行。

    预期行为：四格按固定顺序拼——角色 / 可见性 / 可点性 / 文本，用语言文件里的分隔符
    连起来。这四格就是全部四族条件题（落点 / 角色类 / 挡住的条数 / 近似）的输入。
    """
    cells = (
        loc["role"][row.role],
        loc["visibility"][row.visibility],
        loc["clickable"][row.clickable],
        text_of(row.text, loc),
    )
    return f"[{index}] " + str(loc["meta"]["cell_sep"]).join(str(cell) for cell in cells)


def build(spec: ElementSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：整屏清单文本。

    预期行为：每一格都只从**规格里已经算好的东西**取（屏号、动作标签、六行元素），
    渲染器不做任何判断——判断全在 `spec` 的规则函数里。
    """
    meta = loc["meta"]
    lines = [
        str(meta["title"]),
        field_line(str(meta["screen_label"]), spec.screen_id, loc),
        field_line(str(meta["action_label"]), text_of(spec.case.label, loc), loc),
        "",
    ]
    for index, row in enumerate(spec.rows, start=1):
        lines.append(row_line(index, row, loc))
    return "\n".join(lines)
