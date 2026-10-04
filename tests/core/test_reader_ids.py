"""全库尺子：反解草稿里「编号」字段的取值域必须写进类型（`Literal[...]`），不能是裸 `str`。

为什么要有这条：2026-10-02 真机复验，读者模型会把**正文里的词**（HTML 标签名 `div` / `img`、
字段名 `criteria`、中文词「踩中」）当成判据编号报回来。裸 `str` 时判别器的口径是「认不出来
只算没判成」（`skipped=True`）——那一条等于没判，**闸门静默少一道**。写成 `Literal[*features.X]`
会把取值域放进 JSON schema 的 `enum`，支持结构化输出的端点会约束生成。判据见范式 §9.96。

口径：只看**判别器自己真的拿来卡取值的那个字段**——`unknown = [x for x, _ in pairs if x not in
features.CONST]`（或 `if x not in features.CONST:`）里的变量名与常量名就是取值域，再回 AST 找
那个字段。没有这类检查的文件不管。
"""

from __future__ import annotations

import ast
import pathlib
import re
from typing import Final


#: 找「取值域」的两条写法（判别器里现成的那两行）。
PAIRS_RE: Final[re.Pattern[str]] = re.compile(
    r"unknown = \[(\w+) for \w+, _ in pairs if \1 not in features\.([A-Z_]+)\]"
)
IF_RE: Final[re.Pattern[str]] = re.compile(r"if (\w+) not in features\.([A-Z_]+):")
#: 循环变量名与草稿字段名对不上的两家：`cls` 那个字段叫 `class_id`、`path` 那个叫 `file`。
RENAME: Final[dict[str, str]] = {"cls": "class_id", "path": "file"}


def _targets() -> list[tuple[pathlib.Path, str, str]]:
    """输入：无。输出：[(judges.py, 字段名, 常量名), …]。预期行为：每个文件至多一个字段。"""
    out: list[tuple[pathlib.Path, str, str]] = []
    for path in sorted(pathlib.Path("workflows").rglob("judges.py")):
        text = path.read_text(encoding="utf-8")
        found = set(PAIRS_RE.findall(text)) | set(IF_RE.findall(text))
        if len(found) != 1:
            continue
        var, const = next(iter(found))
        out.append((path, RENAME.get(var, var), const))
    return out


def _alias_value(path: pathlib.Path, alias: str) -> str | None:
    """输入：文件、别名。输出：该模块级赋值的源码；没有这个赋值就 None。"""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == alias for target in node.targets
        ):
            return ast.unparse(node.value)
    return None


def _annotation(path: pathlib.Path, name: str) -> str | None:
    """输入：文件、字段名。输出：该字段的标注源码；没有这个字段就 None。"""
    hits = [
        ast.unparse(item.annotation)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
        for item in node.body
        if isinstance(item, ast.AnnAssign)
        and isinstance(item.target, ast.Name)
        and item.target.id == name
    ]
    return hits[0] if len(hits) == 1 else None


def test_reader_id_fields_are_literal() -> None:
    """输入：无。输出：无。预期行为：判别器拿来卡取值域的字段，标注不能是裸 `str`。"""
    targets = _targets()
    assert len(targets) >= 17, f"只找到 {len(targets)} 家，尺子本身可能坏了"
    bare = [
        f"{path.parent.name}.{name}"
        for path, name, _const in targets
        if _annotation(path, name) in (None, "str")
    ]
    assert bare == [], f"这些反解草稿的编号字段还是裸 str：{bare}（取值域要写进 Literal）"


def test_literal_values_come_from_features() -> None:
    """输入：一家 judges.py、它的字段名与常量名。输出：无。

    预期行为：标注是 `Literal[*features.X]`，**不是手抄的字符串表**——手抄会与 features 漂开。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for path, name, const in _targets():
        try:
            annotation = _annotation(path, name)
            assert annotation is not None
            # `ast.unparse` 会给单元素下标补一个尾逗号（`Literal[*features.TAG_IDS,]`），抹掉再比。
            value = (_alias_value(path, annotation) or "").replace(",]", "]")
            assert value == f"Literal[*features.{const}]", f"{path.parent.name}: {annotation} = {value}"
        except AssertionError as exc:
            bad.append(f"{(path, name, const)!r}: {exc}")
    assert not bad, "\n".join(bad)
