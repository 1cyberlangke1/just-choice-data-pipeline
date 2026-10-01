"""反解提示词的选项名，必须**逐字**等于反解草稿收得下的那组取值（全库尺子）。

为什么要有：题族在数据集里可能是 noul，`criteria` 的 key 是 `true` / `false`，而反解草稿要的
是取值本身（`yes` / `no`、`clean` / `spam`）。提示词照 criteria 列，模型就得在两套词之间自己
翻译——实测就是翻译错：`search` 的 en 侧 6 条里 5 条把 `clean` 读成 `spam`（好数据被拒），
`review` 的 `secret` / `debug` 拒稿理由全是「注入 yes，反解成 no」。两处都在真机上量过
（`tmp/probe_trace.py` / `tmp/probe_schema.py` 的做法：同提示词同 key，只换一处变量）。

输入：每个带 `TRACE_FAMILIES` 的 workflow 的 `judges` 模块 + 它每种语言的 locale。
输出：无（断言）。
预期行为：逐族比对 `jc.llmjudge.trace_options()` 列出来的选项名与 `TraceDraft` 那个字段的
取值域（`Literal[...]` 取实参；`bool` 按提示词里那两行记 `true` / `false`）。字段不是枚举
（例如 `browser` 的 `row: int` / `label: str` 是自由填的）就跳过——那种没有「两套词」的问题。
"""

from __future__ import annotations

import importlib
import typing
from pathlib import Path

from pydantic import BaseModel

from jc import llmjudge
from jc.jsontypes import JsonValue

ROOT = Path(__file__).resolve().parents[2]

#: 题族写在哪一节：多数 workflow 写在 `fam`，`rubric` 的反解口径单独写在 `trace`。
SECTIONS: tuple[str, ...] = ("fam", "trace")


def trace_workflows() -> list[str]:
    """输入：无。输出：带 `TRACE_FAMILIES` 的 workflow 名（按名字排序）。"""
    names: list[str] = []
    for path in sorted((ROOT / "workflows").glob("*/judges.py")):
        module = importlib.import_module(f"workflows.{path.parent.name}.judges")
        if getattr(module, "TRACE_FAMILIES", None):
            names.append(path.parent.name)
    return names


def trace_draft(module: object) -> type[BaseModel]:
    """输入：`judges` 模块。输出：它自己定义的反解草稿模型。

    预期行为：按类名以 `Trace` 开头找；找到 0 个或多个都直接报错——尺子认不出就该红。
    """
    found = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value.__module__ == getattr(module, "__name__", "")
        and value.__name__.startswith("Trace")
    ]
    assert len(found) == 1, f"反解草稿模型认不出唯一一个：{[item.__name__ for item in found]}"
    return found[0]


def accepted_values(annotation: object) -> set[str] | None:
    """输入：字段注解。输出：它收得下的取值；不是枚举/布尔就返回 None（这一族跳过）。"""
    if annotation is bool:
        return {"true", "false"}
    args = typing.get_args(annotation)
    if args and all(isinstance(item, str) for item in args):
        return {str(item) for item in args}
    return None


def section_with(loc: dict[str, JsonValue], family: str) -> str:
    """输入：语言字典、题族名。输出：这个题族写在 `SECTIONS` 里的哪一节。"""
    for section in SECTIONS:
        node = loc.get(section)
        if isinstance(node, dict) and family in node:
            return section
    raise AssertionError(f"{family} 在 {SECTIONS} 哪一节里都找不到")


def test_trace_option_names_match_the_draft_values() -> None:
    """输入：无。输出：无。预期行为：每一族、每种语言，两套词都逐字对上。"""
    problems: list[str] = []
    checked = 0
    for name in trace_workflows():
        module = importlib.import_module(f"workflows.{name}.judges")
        draft = trace_draft(module)
        locales = importlib.import_module(f"workflows.{name}.locales")
        for locale in locales.available():
            loc = locales.load(locale)
            for family in module.TRACE_FAMILIES:
                field = draft.model_fields.get(family)
                if field is None:
                    continue
                expected = accepted_values(field.annotation)
                if expected is None:
                    continue
                names = {
                    option
                    for option, _ in llmjudge.trace_options(
                        family, loc, section=section_with(loc, family)
                    )
                }
                checked += 1
                if names != expected:
                    problems.append(
                        f"{name}/{locale} 的 {family}：提示词列 {sorted(names)}，"
                        f"草稿收 {sorted(expected)}"
                    )
    assert checked >= 30, f"只比到 {checked} 族，扫法是不是坏了？"
    assert not problems, "反解提示词的选项名与草稿取值不是一套词：\n  " + "\n  ".join(problems)


def test_ruler_bites() -> None:
    """输入：无。输出：无。预期行为：尺子自己带一条反例——取值域不同就必须被抓出来。"""
    assert accepted_values(bool) == {"true", "false"}
    assert accepted_values(typing.Literal["a", "b"]) == {"a", "b"}
    assert accepted_values(int) is None
    assert accepted_values(str) is None
