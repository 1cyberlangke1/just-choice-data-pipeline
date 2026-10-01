"""草稿模型（pydantic）**不许给字符串加长度约束**——`minLength` 会把换行压平。

为什么要有：真机上 `review` 的草稿 10 条里 7 条被拒，zh 与 en **一样多**，理由几乎全是
「改动内容只有 1 行」。变量一个个拆开（`tmp/probe_real.py` / `tmp/probe_schema.py` /
`tmp/probe_items.py`，探针原文记在 `可靠合成数据范式.md`）：同一条提示词、同一把 key、
同一个模型——**裸 openai 客户端 4/4 有换行，走 `instructor` 那条路 4/4 被压成一行**；
再逐条把 instructor 那份 schema 里的东西加回去，只有 `"minLength": 1` 一加进去换行就
归零（2/2 → 0），其余（`title`、`name`、schema 级 description、去掉
`additionalProperties`）都不影响。列表的 `minItems` 不触发（2/2 有换行），gemini 那条路
也不触发（3/3 有换行）——所以**只拆字符串长度约束**，列表约束照留。

后果不只是少几行：约束档下模型还会写出 `ヒ` / `character 1` / `: noqa` 这类碎片，它被逼着
把多行内容塞进一行。空串本来就有各家的 `check_draft` 拦（报「草稿的…是空的」再重渲染），
所以这些约束是**冗余**的，删掉不丢任何把关能力。

输入：`src/jc/llmjudge.py` 与 `workflows/*/{render,judges}.py` 里**自己定义**的 pydantic 模型。
输出：无（断言）。
预期行为：把这些模型自己的 JSON schema 摊开，逐层找 `type: string` 且带 `minLength` /
`maxLength` 的节点，一个都不许有；`minItems` / `maxItems` 不在此列（实测不触发，而且
那是真校验——`citation` / `rerank` / `lint` 正靠它对齐句子数 / 候选数 / 行数）。
"""

from __future__ import annotations

import importlib
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]

#: 关掉闸门的写法本身也要被拦：`Field(min_length=…)` 与 `constr(min_length=…)` 都会进 schema，
#: 这条测试看的是**算出来的 schema**，所以两种写法都躲不过——不用再列黑名单。
STRING_LENGTH_KEYS: tuple[str, ...] = ("minLength", "maxLength")


def draft_modules() -> list[str]:
    """输入：无。输出：可能定义草稿模型的模块名（通用层一处 + 每个 workflow 的两处）。"""
    names = ["jc.llmjudge"]
    names += [f"workflows.{path.parent.name}.render" for path in sorted((ROOT / "workflows").glob("*/render.py"))]
    names += [f"workflows.{path.parent.name}.judges" for path in sorted((ROOT / "workflows").glob("*/judges.py"))]
    return names


def draft_models(module: object) -> list[type[BaseModel]]:
    """输入：模块。输出：**它自己定义**的 pydantic 模型类。

    预期行为：只看 `__module__` 指向该模块的类——`from .render import XxxDraft` 这种
    转手引用不算，否则同一个模型会被报好几遍。
    """
    found: list[type[BaseModel]] = []
    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, BaseModel) and value is not BaseModel:
            if value.__module__ == module.__name__:
                found.append(value)
    return found


def string_length_constraints(node: object, where: str) -> list[str]:
    """输入：JSON schema 的一个节点、它在 schema 里的路径。输出：带长度约束的字符串位置。

    预期行为：递归整个 schema（`$defs` / `anyOf` / `items` 都进得去），只在
    `type: string` 的节点上认那两个键——列表上的 `minItems` 不是它管的。
    """
    found: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "string":
            found.extend(f"{where}.{key}={node[key]}" for key in STRING_LENGTH_KEYS if key in node)
        for key, value in node.items():
            found.extend(string_length_constraints(value, f"{where}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(string_length_constraints(value, f"{where}[{index}]"))
    return found


def test_no_draft_model_constrains_string_length() -> None:
    """输入：无。输出：无。预期行为：全库草稿模型的 schema 里没有字符串长度约束。"""
    offenders: list[str] = []
    checked = 0
    for name in draft_modules():
        module = importlib.import_module(name)
        for model in draft_models(module):
            checked += 1
            offenders.extend(
                f"{name}.{model.__name__}{item}"
                for item in string_length_constraints(model.model_json_schema(), "")
            )
    assert checked >= 40, f"只扫到 {checked} 个草稿模型，扫法是不是坏了？"
    assert not offenders, (
        "这些草稿模型给字符串加了长度约束，英伟达那条路上换行会被压平（见本文件文档串）：\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_ruler_bites() -> None:
    """输入：无。输出：无。预期行为：尺子自己带一条反例——加回约束就必须被扫出来。"""

    class RulerProbe(BaseModel):
        """测试用的假草稿模型（故意违规）。"""

        body: str = Field(min_length=1, description="故意加一条长度约束")

    found = string_length_constraints(RulerProbe.model_json_schema(), "")
    assert found, "尺子扫不出自己写的反例，等于没在量"


def test_list_constraints_are_left_alone() -> None:
    """输入：无。输出：无。预期行为：列表长度约束不在禁用之列（那是真校验，实测不触发压平）。"""
    module = importlib.import_module("workflows.rerank.render")
    schemas = [model.model_json_schema() for model in draft_models(module)]
    assert any("minItems" in str(schema) for schema in schemas), (
        "rerank 的候选数约束不见了？列表约束是允许的，别顺手删掉"
    )
