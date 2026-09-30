"""全仓库禁用 `Any` 及其等价物（显式 Any + 隐式 Any）。

为什么要有：`Any` 会让类型检查器在该处彻底闭嘴——等于把「没标注」伪装成「标注了」。
用户原话：「类型标注全是 any 和没标注有什么区别」，接着一句是「如果能禁用 any 就禁用」。
所以判据不是「有没有写出 `Any` 这个词」，而是「这里是不是**真的**有类型」——凡是静态
查得到的 Any 等价物都算，防止它换个写法再长回来。

输入：`src/` `workflows/` `tests/` 下的全部 `*.py` 加仓库根那一层的全部 `*.py`。
输出：无（断言）。
预期行为：用 **AST** 扫，不用 grep——grep 会把注释、文档、字符串里的 `Any` 也算进去，
而 AST 只看真正的 import 与注解。想放宽时改 `EXEMPT` 并写清理由，不许整目录豁免。

禁用范围（漏一条就等于没禁）：

1. `from typing import Any`、`from typing_extensions import Any`（含 `as` 改名）；
2. `from typing import *` / `from typing_extensions import *`——星号导入会把 `Any`
   悄悄带进作用域，没法证明它没被用；
3. 裸名字 `Any`，以及属性 `typing.Any` / `t.Any` / `typing_extensions.Any`；
4. **字符串注解**（`x: "list[Any]"`）与 `# type:` 注释里的 `Any`——它们以字符串存在，
   AST 里看不见名字，只能把字符串再解析一遍；
5. `getattr(typing, "Any")` 这种字面量取属性。

6. **裸泛型标注**：`x: dict` 就是 `dict[Any, Any]`，`list` / `set` / `tuple` / `type` /
   `Counter` / `Callable` / `Sequence` / `Mapping` … 同理；`Optional` / `Union` / `Annotated` /
   `Final` / `ClassVar` 这些 typing 的**特化形式**也一样（`x: Optional` = `Optional[Any]`）。
   判据只有一条：带方括号的放行（参数写全了），不带方括号的报错。清单见 `BARE_GENERICS`，
   它自己也被一条测试钉住（清单里每个名字都必须是标准库里真存在的泛型类，否则规则会腐烂成
   误报源）；
7. **`Callable[..., X]`**：`...` 就是「参数表是 Any」。写清参数类型，或者用 `Protocol`
   （本仓库的测试夹具就是这么写的）。注意 `tuple[int, ...]` **不算**——那里的 `...` 是
   「变长」，是有效类型，不是 Any；
8. **空容器赋值不写元素类型**：`seen = set()` / `rows = []` / `texts = {}`——元素类型推不
   出来时它就是 `set[Any]`。写成 `seen: set[str] = set()`。只认单个 `名字` / `属性` 目标，
   且目标自带注解或 `# type:` 注释的放行；
9. **把闸门关掉的注释**：`# noqa: ANN…` 把标注检查关掉，等于「这里的类型随它去」；裸
   `# noqa`（不带码）连 ANN 一起关，同样算。`# type: ignore` 同理，但只报**光秃秃**那种：
  不带 `[错误码]` 就是把整行的类型检查全关，比 `Any` 还彻底；写清了码
   （`# type: ignore[assignment]`）算「知道自己在压哪一处」，放行——**但码里写着 `no-any-` /
   `disallow-any-` 的除外**：那正是「我在压一个 Any」，等于把 Any 藏进注释里。要豁免就在
   代码里把类型写出来，不是在这里关闸门；
10. **生产代码直接 `json.loads` / `json.load`**：typeshed 里它们返回 `Any`，正是「悄悄拿
    回一个 Any」。JSON 一律走 `jc.jsonio`（它给 `JsonValue`，还顺带拒掉重复 key 与 NaN）。
    `tests/` 不在此列——测试要拿标准库 json 做「这份载荷标准库读得回来」的对照实验，那是
    **故意**的；`src/jc/jsonio.py` 自己当然也不在此列（它就是那个唯一的收口）。

11. **`# type:` 注释里的裸泛型**：注解换个位置写还是注解——`rows = []  # type: list` 就是
    `list[Any]`，`def f(x):  # type: (list) -> None` 同理。第 6 / 7 条的尺子长在**代码注解**
    上，够不到注释，这里补一遍；
12. **「类型参数位」上的裸泛型**：`cast(list, x)` / `TypeVar("T", bound=dict)` /
    `NewType("X", dict)`——这几格实参写的是**类型表达式**而不是普通值，裸泛型放在那里同样
    等于 Any；它们不是注解，第 6 条的遍历也够不到。

不在此列的：文档串、普通注释、测试里当**数据**用的字符串（AST 看不见它们）。
动态拼出来的名字（`getattr(m, "An" + "y")`）静态查不到，属于明知漏网，不假装能查。
第三方无桩库返回的隐式 Any（typeshed 覆盖不到的地方）也一样：那要 mypy / pyright 全量
迁移才管得住，本仓库还没到那一步，理由与实测数字写在 `pyproject.toml` 那段注释里。
"""

from __future__ import annotations

import ast
import builtins
import collections
import collections.abc
import io
import re
import tokenize
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

#: 扫描入口：这几个目录递归扫 `*.py`。
ROOTS: tuple[str, ...] = ("src", "workflows", "tests")

#: 仓库根那一层的散装脚本（`conftest.py`，以及将来可能出现的 `setup.py` / `noxfile.py`）。
#: 用 glob 而不是写死文件名：写死的话，新加一个根脚本就等于在规则外开了个口子。
ROOT_GLOB: str = "*.py"

#: 豁免文件：键是相对路径，值是理由。空 = 一个都不豁免。
EXEMPT: dict[str, str] = {}

#: 会提供 `Any` 的模块——星号导入与 `Any` 名字都从这两处进来。
_ANY_MODULES = frozenset({"typing", "typing_extensions"})

#: 要找的名字本身。
_BANNED = "Any"

#: 裸泛型的名字清单：这些名字**不带方括号**时，类型参数就是隐式 Any。
#: 只收本仓库真会用到的（内建容器 / collections / 常用 abc / typing 的旧写法）。
#: `test_bare_generics_are_real_generics` 会逐个验证它们确实是标准库里的泛型类。
BARE_GENERICS: Final[frozenset[str]] = frozenset(
    {
        # 内建
        "dict", "list", "set", "frozenset", "tuple", "type",
        # collections / collections.abc
        "Counter", "defaultdict", "deque", "OrderedDict", "ChainMap",
        "Callable", "Iterable", "Iterator", "Sequence", "Mapping", "MutableMapping",
        "AbstractSet", "MutableSet", "Generator", "Coroutine", "Awaitable",
        "AsyncIterator", "AsyncIterable", "AsyncGenerator", "Reversible",
        "Container", "KeysView", "ValuesView", "ItemsView",
        # re
        "Pattern", "Match",
        # typing 里的旧写法（大写别名）
        "Dict", "List", "Set", "FrozenSet", "Tuple", "Type",
        "DefaultDict", "Deque", "OrderedDict",
        # typing 的特化形式：裸用时类型参数同样是隐式 Any（`x: Optional` = `Optional[Any]`）
        "Optional", "Union", "Annotated", "Final", "ClassVar",
    }
)

#: 「空容器」的构造器名字：`x = set()` 这种没写元素类型的赋值，元素类型推不出来时是 Any。
EMPTY_CONTAINERS: Final[frozenset[str]] = frozenset(
    {"set", "dict", "list", "tuple", "frozenset", "Counter", "deque", "defaultdict", "OrderedDict"}
)

#: `noqa` 注释的形态：光秃秃的 `noqa`，或者 `noqa` 冒号后跟规则码（如 ANN202 / E501），
#: 码后面允许跟人话注释。
_NOQA: Final[re.Pattern[str]] = re.compile(r"#\s*noqa\b(?P<rest>.*)$")

#: 从 `noqa` 冒号后面揪出规则码（形如 ANN202 / E501 / F821）。
_RULE_CODE: Final[re.Pattern[str]] = re.compile(r"\b[A-Z]{1,4}\d{2,4}\b")

#: `# type: ignore` 的形态：注释**一开头**就是它（所以文档里引它不算），后面可以跟 `[错误码]`。
_TYPE_IGNORE: Final[re.Pattern[str]] = re.compile(r"^#\s*type:\s*ignore\b(?P<rest>.*)$")

#: `# type: ignore[...]` 方括号里的 mypy 错误码（小写、可带连字符、逗号分隔）。
_MYPY_CODES: Final[re.Pattern[str]] = re.compile(r"\[\s*[a-z][a-z0-9,\- ]*\]")

#: 返回 `Any` 的标准库 JSON 读入口（写入口 `json.dumps` 返回 str，不在列）。
_ANY_JSON_READERS: Final[frozenset[str]] = frozenset({"loads", "load"})

#: 唯一允许直接调 `json.loads` 的文件——严格读写的收口。
_JSON_GATE: Final[str] = "src/jc/jsonio.py"


@dataclass(frozen=True)
class Finding:
    """一处「Any 或其等价物」：行号 + 形态（报错时说人话用）。"""

    line: int
    shape: str


def _dotted_name(node: ast.AST) -> str:
    """输入：任意表达式节点。输出：它的名字（`a.b.C` 取最后一段 `C`）；不是名字给空串。

    预期行为：只看 `Name` / `Attribute`；下标的**基名**也走这里（`dict[str, int]` 的 `dict`）。
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_production(rel: str) -> bool:
    """输入：相对路径。输出：它是不是「生产代码」（`src/` `workflows/`，除了 jsonio 自己）。

    预期行为：路径分隔符先归一成 `/`；测试与仓库根脚本一律不算生产代码（理由见模块文档第 10 条）。
    """
    posix = rel.replace("\\", "/")
    if posix == _JSON_GATE:
        return False
    return posix.startswith("src/") or posix.startswith("workflows/")


def _sources() -> list[str]:
    """输入：无。输出：待检查文件的相对路径（排序后）。

    预期行为：目录递归取 `*.py` 并跳过 `__pycache__`；仓库根那一层的 `*.py` 也全收
    （写死 `conftest.py` 会让新加的根脚本漏网）。
    """
    found: list[str] = []
    for entry in ROOTS:
        path = Path(entry)
        for child in path.rglob("*.py"):
            if "__pycache__" in child.parts:
                continue
            found.append(child.as_posix())
    found.extend(path.as_posix() for path in Path().glob(ROOT_GLOB))
    return sorted(set(found))


def _hits_in(node: ast.AST) -> list[int]:
    """输入：任意 AST 子树。输出：其中裸 `Any` / `.Any` 的行号。

    预期行为：只看名字与属性节点；字符串常量不算（字符串由 `_string_sites` 单独处理）。
    """
    found: list[int] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == _BANNED:
            found.append(sub.lineno)
        elif isinstance(sub, ast.Attribute) and sub.attr == _BANNED:
            found.append(sub.lineno)
    return found


def _parse_annotation_text(text: str) -> ast.expr | None:
    """输入：注解的字符串写法（如 `list[Any]`）。输出：解析后的表达式；解析不了给 None。

    预期行为：用 `eval` 模式解析——注解本来就是个表达式。解析失败说明它压根不是
    合法注解（比如 `# type:` 后面写了句人话），按「查不到」处理，不抛异常。
    """
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError:
        return None


def _parse_func_type_text(text: str) -> ast.FunctionType | None:
    """输入：函数上的 `# type:` 注释文本（如 `(list, int) -> None`）。输出：解析后的签名。

    预期行为：函数类型注释不是**一个表达式**，`eval` 模式解析不了，得用 `func_type` 模式；
    解析不了（注释里写的是人话）按「查不到」处理，不抛异常。
    """
    try:
        return ast.parse(text, mode="func_type")
    except SyntaxError:
        return None


def _parse_module(source: str, filename: str) -> ast.Module:
    """输入：源码文本与文件名。输出：模块 AST。

    预期行为：先带 `type_comments=True` 解析——不带这个开关时 `# type:` 注释根本不会
    挂到 `type_comment` 字段上，第 4 类写法就查不出来。注释写歪了（`# type:` 后面写人话）
    会直接抛 SyntaxError，那就退回不带类型注释再解析一遍：歪注释里的 `Any` 认不出来，
    但绝不因此崩掉整条测试。
    """
    try:
        return ast.parse(source, filename=filename, type_comments=True)
    except SyntaxError:
        return ast.parse(source, filename=filename)


def _string_annotation_hits(annotation: ast.AST) -> list[int]:
    """输入：一个注解子树。输出：其中字符串注解里出现 `Any` 的行号。

    预期行为：字符串注解在 AST 里只是 `Constant`，得把它的值再解析一遍才看得见 `Any`。
    行号取那个字符串常量本身的行号——内层解析出来的行号是相对字符串自己的，没法用。
    """
    found: list[int] = []
    for sub in ast.walk(annotation):
        if not isinstance(sub, ast.Constant) or not isinstance(sub.value, str):
            continue
        parsed = _parse_annotation_text(sub.value)
        if parsed is not None and _hits_in(parsed):
            found.append(sub.lineno)
    return found


def _string_sites(tree: ast.AST) -> list[int]:
    """输入：整棵 AST。输出：字符串注解与 `# type:` 注释里出现 `Any` 的行号。

    预期行为：遍历所有「能挂注解」的节点取注解子树；`# type:` 注释不挂在注解字段上，
    而是躺在语句节点的 `type_comment` 字符串字段里，单独解析，行号用语句自己的行号。
    """
    found: list[int] = []
    for node in ast.walk(tree):
        annotation: ast.AST | None = None
        if isinstance(node, ast.arg):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        elif isinstance(node, ast.AnnAssign):
            annotation = node.annotation
        if annotation is not None:
            found.extend(_string_annotation_hits(annotation))
        comment = getattr(node, "type_comment", None)
        if isinstance(comment, str):
            parsed = _parse_annotation_text(comment)
            if parsed is not None and _hits_in(parsed):
                found.append(node.lineno)
    return found


def _getattr_literal_hits(tree: ast.AST) -> list[int]:
    """输入：整棵 AST。输出：`getattr(..., "Any")` 这种字面量取属性的行号。

    预期行为：只认第二个实参是字面量 `"Any"` 的调用；动态名字静态查不到，不假装能查。
    """
    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "getattr"):
            continue
        if len(node.args) >= 2:
            name = node.args[1]
            if isinstance(name, ast.Constant) and name.value == _BANNED:
                found.append(node.lineno)
    return found


def _rebase(found: list[Finding], line: int) -> list[Finding]:
    """输入：一批发现 + 统一行号。输出：行号全换成 `line` 的那批发现。

    预期行为：字符串注解里再解析出来的那棵树，行号是相对**那个字符串自己**的，报出去
    指不到真位置；统一换成外层字符串常量所在的行。
    """
    return [Finding(line, item.shape) for item in found]


def _shape_findings(annotation: ast.AST) -> list[Finding]:
    """输入：一个注解子树。输出：里面的裸泛型与 `Callable[..., X]`（行号取节点自己的）。

    预期行为：`dict[str, int]` 里的 `dict` 是被下标吃掉的**基名**，不算裸用——先把所有
    `Subscript.value` 的 id 记下来再从结果里剔掉。`tuple[int, ...]` 的 `...` 是「变长」
    标记（有效类型），只有出现在 `Callable` 的**第一格**时才当「参数表是 Any」。
    """
    bases = {id(sub.value) for sub in ast.walk(annotation) if isinstance(sub, ast.Subscript)}
    found: list[Finding] = []
    for sub in ast.walk(annotation):
        if isinstance(sub, ast.Subscript) and _dotted_name(sub.value) == "Callable":
            parts = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else []
            if parts and isinstance(parts[0], ast.Constant) and parts[0].value is Ellipsis:
                found.append(Finding(sub.lineno, "Callable[..., X]（参数表是 Any）"))
        elif isinstance(sub, (ast.Name, ast.Attribute)) and id(sub) not in bases:
            name = _dotted_name(sub)
            if name in BARE_GENERICS:
                found.append(Finding(sub.lineno, f"裸泛型 {name}（= {name}[Any, …]）"))
    return found


def _annotation_shape_sites(tree: ast.AST) -> list[Finding]:
    """输入：整棵 AST。输出：所有注解里的裸泛型 / `Callable[..., X]`。

    预期行为：遍历所有「能挂注解」的节点（参数、返回值、变量标注）各跑一遍形态检查；
    注解写成字符串时（`x: "list[dict]"`）把字符串再解析一遍，行号统一换成字符串那行。
    """
    found: list[Finding] = []
    for node in ast.walk(tree):
        annotation: ast.AST | None = None
        if isinstance(node, ast.arg):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        elif isinstance(node, ast.AnnAssign):
            annotation = node.annotation
        if annotation is None:
            continue
        found.extend(_shape_findings(annotation))
        for sub in ast.walk(annotation):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                parsed = _parse_annotation_text(sub.value)
                if parsed is not None:
                    found.extend(_rebase(_shape_findings(parsed), sub.lineno))
    return found


def _type_comment_shape_sites(tree: ast.AST) -> list[Finding]:
    """输入：整棵 AST。输出：`# type:` 注释里写成裸泛型 / `Callable[..., X]` 的位置。

    预期行为：注释里的类型和代码里的类型是同一件事，只是换了个地方写，所以形态检查也要
    跑一遍。普通类型注释（赋值 / `for` / `with`）是**一个表达式**，用 `eval` 模式解析；
    函数头上的是 `(list, int) -> None` 这种**签名**，用 `func_type` 模式解析。行号统一
    换成语句自己的行——解析出来的那棵小树，行号是相对注释自己的，报出去指不到真位置。
    """
    found: list[Finding] = []
    for node in ast.walk(tree):
        comment = getattr(node, "type_comment", None)
        if not isinstance(comment, str):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signature = _parse_func_type_text(comment)
            if signature is None:
                continue
            parts = [*signature.argtypes, signature.returns]
        else:
            expression = _parse_annotation_text(comment)
            if expression is None:
                continue
            parts = [expression]
        for part in parts:
            found.extend(_rebase(_shape_findings(part), node.lineno))
    return found


def _type_argument_sites(tree: ast.AST) -> list[Finding]:
    """输入：整棵 AST。输出：`cast` / `TypeVar` / `NewType` 里写成裸泛型的类型实参。

    预期行为：这三个调用的实参里有一格是**类型表达式**（不是普通值），写在注解里要报的
    裸泛型，写在这里同样等于 Any。口径：`cast(T, x)` 只看第 1 格（第 2 格是值）；
    `NewType("N", T)` 只看第 2 格（第 1 格是字符串名字）；`TypeVar("T", …)` 看第 1 格
    之后的全部位置参数（约束）与 `bound=` 关键字。名字里没有的类型实参（`cast(int, x)`）
    本来就不是泛型，不报。
    """
    found: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        positions: list[ast.expr] = []
        if name == "cast":
            positions = list(node.args[:1])
        elif name == "NewType":
            positions = list(node.args[1:2])
        elif name == "TypeVar":
            positions = list(node.args[1:])
            positions.extend(item.value for item in node.keywords if item.arg == "bound")
        for position in positions:
            found.extend(_shape_findings(position))
    return found


def _container_sites(tree: ast.AST) -> list[Finding]:
    """输入：整棵 AST。输出：没写元素类型的空容器赋值（`seen = set()` / `rows = []`）。

    预期行为：只认**单目标、无注解、无 `# type:` 注释**的赋值——注解写全了放行；
    `x = [1, 2]` 非空，元素类型看得见，也放行；`a, b = [], []` 这种多目标不在此列
    （属于明知漏网，规则要短）。**重新赋值也照报**：规则不去追「这个作用域里别处有没有
    带类型的绑定」——那要写一整套作用域分析，钝一点但零歧义；真想清空已有容器就写
    `.clear()`，比重新绑定更贴意图。
    """
    found: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or node.type_comment is not None:
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], (ast.Name, ast.Attribute)):
            continue
        value = node.value
        empty_literal = (isinstance(value, ast.List) and not value.elts) or (
            isinstance(value, ast.Dict) and not value.keys
        )
        empty_call = (
            isinstance(value, ast.Call)
            and not value.args
            and not value.keywords
            and _dotted_name(value.func) in EMPTY_CONTAINERS
        )
        if empty_literal or empty_call:
            found.append(Finding(node.lineno, "空容器没写元素类型（推不出来时是 X[Any]）"))
    return found


def _muting_comment_sites(source: str) -> list[Finding]:
    """输入：源码文本。输出：把类型检查关掉的注释（`# noqa` / `# type: ignore`）所在行。

    预期行为：用 `tokenize` 拿**真注释**——字符串里写的 `# noqa: ANN202` 是数据，不算；
    `noqa` 带码的只看码（含 `ANN` 才报），裸 `# noqa` 连 ANN 一起关，同样报；
    `# type: ignore` **没写** `[错误码]` 时报（整行闸门都关了）；写了码但码名里带 `any`
    （`no-any-return` / `disallow-any-expr`…）的也报——那正是「我在压一个 Any」。tokenize
    遇到解析不了的源码就放弃这一项：一个手滑的注释不该把整条测试炸掉。
    """
    found: list[Finding] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return found
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        match = _NOQA.search(token.string)
        if match is not None:
            rest = match.group("rest")
            codes = _RULE_CODE.findall(rest) if ":" in rest else []
            if not codes or any(code.startswith("ANN") for code in codes):
                found.append(Finding(token.start[0], "noqa 关掉了标注检查（ANN）"))
        ignore = _TYPE_IGNORE.search(token.string)
        if ignore is None:
            continue
        codes = _MYPY_CODES.search(ignore.group("rest"))
        if codes is None:
            found.append(Finding(token.start[0], "type: ignore 没写错误码（整行闸门都关了）"))
        elif "any" in codes.group(0).lower():
            found.append(Finding(token.start[0], "type: ignore 压的正是 Any 那类错误（no-any-…）"))
    return found


def _json_reader_sites(tree: ast.AST, rel: str) -> list[Finding]:
    """输入：整棵 AST 与相对路径。输出：生产代码里直接调 `json.loads` / `json.load` 的位置。

    预期行为：只扫 `src/` `workflows/`（`src/jc/jsonio.py` 除外，它就是那个收口）；
    测试与仓库根脚本放行（理由见模块文档第 10 条）。`import json as j` 这种改名认不出来，
    属于明知漏网。
    """
    if not _is_production(rel):
        return []
    found: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _ANY_JSON_READERS
            and _dotted_name(func.value) == "json"
        ):
            found.append(Finding(node.lineno, "json.loads/load 直接拿回 Any（该走 jc.jsonio）"))
    return found


def any_findings_in_source(
    source: str, filename: str = "<source>", rel: str = ""
) -> list[Finding]:
    """输入：Python 源码文本、文件名、相对路径。输出：全部发现（按行号、形态排序）。

    预期行为：把十二类写法合到一处——显式 `Any`（import / 名字 / 属性 / 字符串注解 /
    `# type:` 注释 / `getattr` 字面量）、星号导入、裸泛型、`Callable[..., X]`、空容器、
    关闸门的注释（`# noqa: ANN…` / 裸 `# type: ignore` / `[no-any-…]`）、生产代码的
    `json.loads`、`# type:` 注释里的裸泛型、「类型参数位」（`cast` / `TypeVar` / `NewType`）
    上的裸泛型。`rel` 为空表示「不在仓库里」（最后一类不测）。
    """
    tree = _parse_module(source, filename)
    found: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _ANY_MODULES:
            for alias in node.names:
                if alias.name == "*":
                    found.append(Finding(node.lineno, "星号导入（会把 Any 带进作用域）"))
                elif alias.name == _BANNED:
                    found.append(Finding(node.lineno, "显式 Any"))
        elif isinstance(node, (ast.Name, ast.Attribute)) and _dotted_name(node) == _BANNED:
            found.append(Finding(node.lineno, "显式 Any"))
    found.extend(Finding(line, "显式 Any（字符串注解 / # type: 注释）") for line in _string_sites(tree))
    found.extend(Finding(line, "显式 Any（getattr 字面量）") for line in _getattr_literal_hits(tree))
    found.extend(_annotation_shape_sites(tree))
    found.extend(_type_comment_shape_sites(tree))
    found.extend(_type_argument_sites(tree))
    found.extend(_container_sites(tree))
    found.extend(_muting_comment_sites(source))
    found.extend(_json_reader_sites(tree, rel))
    return sorted(found, key=lambda item: (item.line, item.shape))


def any_sites_in_source(source: str, filename: str = "<source>", rel: str = "") -> list[int]:
    """输入：Python 源码文本与其文件名。输出：出现 Any 等价物的行号（去重升序）。

    预期行为：文档串、普通注释、当数据用的字符串一律不算；十类写法全算（见模块文档）。
    返回行号而不是布尔，方便报错时直接指到行。
    """
    return sorted({item.line for item in any_findings_in_source(source, filename, rel)})


def any_sites(path: Path) -> list[int]:
    """输入：一个 `*.py` 路径。输出：出现 Any 等价物的行号（去重升序）。

    预期行为：按 UTF-8 读文件再交给 `any_sites_in_source`；文件读不了就抛（不吞错）。
    """
    return any_sites_in_source(
        path.read_text(encoding="utf-8"), filename=str(path), rel=path.as_posix()
    )


#: 干净源码：文档串和注释里提 `Any` 不算，因为 AST 看不见它们。
_CLEAN_SOURCE = '''
"""这个文档串里写着 Any，但它不该被算进去。"""
# 这行注释里也写着 Any。


def add(a: int, b: int) -> int:
    """返回两数之和。"""
    return a + b
'''

#: 当数据用的字符串里有 `Any`，也不该算——它不是注解。
_DATA_ONLY_SOURCE = 'NAME = "Any"\nTEXT = "from typing import Any"\n'


def test_clean_source_has_no_sites() -> None:
    """输入：文档串与注释里提 `Any` 的源码。输出：无。预期行为：一个行号都不报。"""
    assert any_sites_in_source(_CLEAN_SOURCE) == []


def test_data_only_strings_are_ignored() -> None:
    """输入：把 `Any` 当数据字符串用的源码。输出：无。预期行为：不当成注解，不报。"""
    assert any_sites_in_source(_DATA_ONLY_SOURCE) == []


def test_broken_type_comment_does_not_crash() -> None:
    """输入：`# type:` 后面写了句人话的源码。输出：无。

    预期行为：解析不了就跳过，不抛异常——否则一个手滑的注释会把整条测试炸掉。
    """
    assert any_sites_in_source("x = 1  # type: 这里写的不是类型\n") == []


@pytest.mark.parametrize(
    ("source", "expect"),
    [
        ("from typing import Any\n", [1]),
        ("from typing import Any as Whatever\n", [1]),
        ("from typing_extensions import Any\n", [1]),
        ("from typing import *\n", [1]),
        ("from typing_extensions import *\n", [1]),
        ("import typing\n\ndef f(x: typing.Any) -> None: ...\n", [3]),
        ("import typing as t\n\nx: t.Any = 1\n", [3]),
        ("def f(x: Any) -> None: ...\n", [1]),
        ("def f(x: 'Any') -> None: ...\n", [1]),
        ("def f(x: list['Any']) -> None: ...\n", [1]),
        ("def f(x: 'list[typing.Any]') -> None: ...\n", [1]),
        ("x = []  # type: list[Any]\n", [1]),
        ("import typing\n\nT = getattr(typing, 'Any')\n", [3]),
        ("import typing\n\nAny = 1\n", [3]),
    ],
)
def test_banned_shapes_are_found(source: str, expect: list[int]) -> None:
    """输入：一段含 `Any` 的源码与期望行号。输出：无。预期行为：报出的行号与期望一致。"""
    assert any_sites_in_source(source) == expect


def test_any_sites_reads_file(tmp_path: Path) -> None:
    """输入：临时目录里一个含 `Any` 的文件。输出：无。预期行为：路径版与文本版结果一致。"""
    target = tmp_path / "sample.py"
    target.write_text("import typing\n\nAlias = typing.Any\n", encoding="utf-8")
    assert any_sites(target) == [3]


@pytest.mark.parametrize("rel", _sources())
def test_no_any_in_the_repo(rel: str) -> None:
    """输入：一个相对路径。输出：无。

    预期行为：文件里不许出现 `Any` 或其等价物；报了就把**行号 + 形态**一起说出来，
    形态直接指向模块文档里对应那条修法，不用回头猜是哪种写法。
    """
    if rel in EXEMPT:
        pytest.skip(EXEMPT[rel])
    text = Path(rel).read_text(encoding="utf-8")
    found = any_findings_in_source(text, filename=rel, rel=rel)
    assert not found, "\n".join(f"{rel} 第 {item.line} 行：{item.shape}" for item in found)


def test_sources_are_actually_found() -> None:
    """输入：无。输出：无。

    预期行为：扫到的文件数够多。尺子最怕的不是漏报而是**空转**——路径写错、目录改名，
    参数化用例数变成 0，pytest 一声不吭地全绿。这条把那种情况钉住。
    """
    assert len(_sources()) > 100, f"只扫到 {len(_sources())} 个文件，尺子可能空转了"


def _accepts_type_args(obj: object) -> bool:
    """输入：任意对象。输出：它接不接方括号（`X[int]` / `X[int, int]` / `X[[int], int]`）。

    预期行为：三种形态都试一遍——`Callable` 只吃「两格」写法（参数表 + 返回值），
    只试 `X[int]` 会把它误判成「不是泛型」。不抛 TypeError 就算接得住。
    """
    for key in (int, (int, int), ([int], int)):
        try:
            obj[key]
        except TypeError:
            continue
        return True
    return False


def test_bare_generics_are_real_generics() -> None:
    """输入：无。输出：无。

    预期行为：`BARE_GENERICS` 里每个名字都能在标准库里解析成一个**真的能下标**的东西。
    清单写错一个名字，规则就从「禁 Any」变成「禁某个正常类」，而且是**静默**误伤——
    所以清单自己也得有尺子，不靠人记。判据用运行时试下标，不用 `__class_getitem__`：
    后者对 `type` / `typing.Dict` 这类是假阴性（它们能下标但没那个属性）。
    """
    namespaces = (vars(builtins), vars(collections), vars(collections.abc), vars(typing), vars(re))
    resolved = {
        name: next((ns[name] for ns in namespaces if name in ns), None) for name in BARE_GENERICS
    }
    unknown = [name for name, obj in sorted(resolved.items()) if not _accepts_type_args(obj)]
    assert not unknown, f"这些名字不是标准库里的泛型（要么删掉，要么改对）：{unknown}"


#: 裸泛型的正例：(源码, 期望行号)。每种写法都要报，行号要对得上。
_BARE_GENERIC_CASES: tuple[tuple[str, list[int]], ...] = (
    ("def f(x: dict) -> None: ...\n", [1]),
    ("def f() -> list: ...\n", [1]),
    ("x: tuple = ()\n", [1]),
    ("import typing\n\nx: typing.Dict = {}\n", [3]),
    ("def f(x: list[dict]) -> None: ...\n", [1]),
    ("def f(x: 'list[set]') -> None: ...\n", [1]),
    ("from collections import Counter\n\nx: Counter = Counter()\n", [3]),
    ("def f(x: Optional) -> None: ...\n", [1]),
    ("def f() -> Union: ...\n", [1]),
    ("x: Final = 1\n", [1]),
    ("x: Annotated = 1\n", [1]),
    ("class C:\n    x: ClassVar = 1\n", [2]),
    ("def f(x: Optional[int]) -> None: ...\n", []),
    ("def f(x: Union[int, str]) -> None: ...\n", []),
)


@pytest.mark.parametrize(("source", "expect"), _BARE_GENERIC_CASES)
def test_bare_generics_are_found(source: str, expect: list[int]) -> None:
    """输入：裸泛型标注的源码与期望行号。输出：无。预期行为：报出的行号与期望一致。"""
    assert any_sites_in_source(source) == expect


#: 负例：这些是**真标注**，一个都不许误伤。误伤等于逼人把类型删掉，比漏报更糟。
_REAL_ANNOTATION_CASES: tuple[str, ...] = (
    "def f(x: dict[str, int]) -> None: ...\n",
    "def f(x: tuple[int, ...]) -> None: ...\n",
    "def f(x: Callable[[int], str]) -> None: ...\n",
    "def f(x: Sequence[int]) -> None: ...\n",
    "def f(x: list[list[int]]) -> None: ...\n",
    "def f(x: 'dict[str, list[int]]') -> None: ...\n",
    "if isinstance(x, dict):\n    pass\n",
    "rows: list[str] = []\n",
    "seen: set[str] = set()\n",
    "texts: dict[str, str] = {}\n",
    "def f() -> None:\n    rows = [1]\n",
    "def f() -> None:\n    rows = []  # type: list[int]\n",
    "import json\n\ntext = json.dumps({'a': 1})\n",
)


@pytest.mark.parametrize("source", _REAL_ANNOTATION_CASES)
def test_real_annotations_are_not_flagged(source: str) -> None:
    """输入：类型写全了的源码。输出：无。预期行为：一个行号都不报（尺子的下限）。"""
    assert any_sites_in_source(source) == []


@pytest.mark.parametrize(
    ("source", "expect"),
    [
        ("def f(x: Callable[..., int]) -> None: ...\n", [1]),
        ("def f(x: Callable[..., int] | None) -> None: ...\n", [1]),
        ("def f(x: Callable[[int], int]) -> None: ...\n", []),
        ("def f(x: tuple[int, ...]) -> None: ...\n", []),
    ],
)
def test_callable_with_ellipsis_is_banned(source: str, expect: list[int]) -> None:
    """输入：`Callable[..., X]` 的几种写法。输出：无。

    预期行为：参数表写成 `...` 就报（那就是 Any）；`tuple[int, ...]` 的 `...` 是「变长」，
    是有效类型，不许报——两者长得像，判据是它有没有落在 `Callable` 的第一格。
    """
    assert any_sites_in_source(source) == expect


@pytest.mark.parametrize(
    ("source", "expect"),
    [
        ("def f() -> None:\n    rows = []\n", [2]),
        ("def f() -> None:\n    texts = {}\n", [2]),
        ("def f() -> None:\n    seen = set()\n", [2]),
        ("def f() -> None:\n    seen = set([1])\n", []),
        ("def f() -> None:\n    texts = {'a': 1}\n", []),
        ("class C:\n    def __init__(self) -> None:\n        self.rows = []\n", [3]),
        ("a, b = [], []\n", []),
    ],
)
def test_empty_containers_without_element_type_are_banned(source: str, expect: list[int]) -> None:
    """输入：空容器赋值的几种写法。输出：无。

    预期行为：没写元素类型的空容器就报；带注解、带 `# type:`、非空、多目标的一律不报。
    """
    assert any_sites_in_source(source) == expect


@pytest.mark.parametrize(
    ("source", "expect"),
    [
        ("def f():\n    return 1  # noqa: ANN202\n", [2]),
        ("def f():\n    return 1  # noqa: ANN202 —— 测试内部小工具\n", [2]),
        ("def f():\n    return 1  # noqa\n", [2]),
        ("def f():\n    return 1  # noqa: E501\n", []),
        ("TEXT = '# noqa: ANN202'\n", []),
    ],
)
def test_noqa_that_mutes_annotations_is_banned(source: str, expect: list[int]) -> None:
    """输入：几种 `noqa` 写法。输出：无。

    预期行为：码里带 ANN（或干脆不带码，连 ANN 一起关）就报；别的码放行；字符串里写的
    `noqa` 是数据，不算。
    """
    assert any_sites_in_source(source) == expect


#: `# type: ignore` 的几种写法：(源码, 期望行号)。
_TYPE_IGNORE_CASES: tuple[tuple[str, list[int]], ...] = (
    ("def f() -> int:\n    return 1  # type: ignore\n", [2]),
    ("def f() -> int:\n    return 1  # type:ignore\n", [2]),
    ("def f() -> int:\n    return 1  # type: ignore  # 上游桩子不对\n", [2]),
    ("def f() -> int:\n    return 1  # type: ignore[assignment]\n", []),
    ("def f() -> int:\n    return 1  # type: ignore[arg-type, return-value]\n", []),
    ("def f() -> int:\n    return 1  # type: ignore[attr-defined]  # 动态挂的属性\n", []),
    ("def f() -> int:\n    return 1  # type: ignore[no-any-return]\n", [2]),
    ("def f() -> int:\n    return 1  # type: ignore[no-any-unimported]  # 上游没桩\n", [2]),
    ("def f() -> int:\n    return 1  # type: ignore[disallow-any-expr, assignment]\n", [2]),
    ("TEXT = '# type: ignore'\n", []),
    ("x = 1  # type: int\n", []),
)


@pytest.mark.parametrize(("source", "expect"), _TYPE_IGNORE_CASES)
def test_bare_type_ignore_is_banned(source: str, expect: list[int]) -> None:
    """输入：几种 `# type: ignore` 写法。输出：无。

    预期行为：**不带** `[错误码]` 的 `# type: ignore` 报——它把整行的类型检查全关，比
    `Any` 还彻底；写清了码的放行（本仓库现有那几处都是带码的），**但码名里带 `any` 的
    除外**（`no-any-return` / `disallow-any-expr`…）：那正是「我在压一个 Any」；`# type: int`
    这种普通类型注释不是 ignore，不报；字符串里写的 `# type: ignore` 是数据，不报。
    """
    assert any_sites_in_source(source) == expect


#: `# type:` 注释里写成裸泛型的几种写法：(源码, 期望行号)。
_TYPE_COMMENT_SHAPE_CASES: tuple[tuple[str, list[int]], ...] = (
    ("rows = []  # type: list\n", [1]),
    ("rows = []  # type: dict\n", [1]),
    ("rows = []  # type: Callable[..., int]\n", [1]),
    ("rows = []  # type: list[int]\n", []),
    ("rows = []  # type: dict[str, int]\n", []),
    ("def f(x):  # type: (list) -> None\n    return None\n", [1]),
    ("def f(x):  # type: (list[int]) -> dict\n    return {}\n", [1]),
    ("def f(x):  # type: (list[int]) -> dict[str, int]\n    return {}\n", []),
    ("x = 1  # type: 这里写的不是类型\n", []),
)


@pytest.mark.parametrize(("source", "expect"), _TYPE_COMMENT_SHAPE_CASES)
def test_type_comment_shapes_are_banned(source: str, expect: list[int]) -> None:
    """输入：`# type:` 注释的几种写法。输出：无。

    预期行为：注释里的裸泛型报（它就是隐式 Any）；带方括号的放行；函数头上那种
    `(参数表) -> 返回值` 两段都要查；注释里写人话（解析不了）不报、不崩。
    """
    assert any_sites_in_source(source) == expect


#: 「类型参数位」上写成裸泛型的几种写法：(源码, 期望行号)。
_TYPE_ARGUMENT_CASES: tuple[tuple[str, list[int]], ...] = (
    ("from typing import cast\n\nx = cast(list, y)\n", [3]),
    ("from typing import cast\n\nx = cast(Callable[..., int], y)\n", [3]),
    ("from typing import cast\n\nx = cast(list[int], y)\n", []),
    ("from typing import cast\n\nx = cast(SomeClass, y)\n", []),
    ("from typing import TypeVar\n\nT = TypeVar('T', bound=dict)\n", [3]),
    ("from typing import TypeVar\n\nT = TypeVar('T', list, set)\n", [3]),
    ("from typing import TypeVar\n\nT = TypeVar('T', bound=dict[str, int])\n", []),
    ("from typing import NewType\n\nUid = NewType('Uid', dict)\n", [3]),
    ("from typing import NewType\n\nUid = NewType('Uid', dict[str, int])\n", []),
    ("from typing import NewType\n\nUid = NewType('Uid', str)\n", []),
)


@pytest.mark.parametrize(("source", "expect"), _TYPE_ARGUMENT_CASES)
def test_type_argument_positions_are_banned(source: str, expect: list[int]) -> None:
    """输入：`cast` / `TypeVar` / `NewType` 的几种写法。输出：无。

    预期行为：类型实参位置上的裸泛型报；写全了的、以及压根不是泛型的类型（`str` /
    `SomeClass`）不报——误伤等于逼人把类型删掉，比漏报更糟。
    """
    assert any_sites_in_source(source) == expect


@pytest.mark.parametrize(
    ("source", "rel", "expect"),
    [
        ("import json\n\nx = json.loads('{}')\n", "src/jc/cli.py", [3]),
        ("import json\n\nx = json.load(handle)\n", "workflows/mail/render.py", [3]),
        ("import json\n\nx = json.dumps({'a': 1})\n", "src/jc/cli.py", []),
        ("import json\n\nx = json.loads('{}')\n", "tests/core/test_cli.py", []),
        ("import json\n\nx = json.loads('{}')\n", "src/jc/jsonio.py", []),
        ("import json\n\nx = json.loads('{}')\n", "conftest.py", []),
    ],
)
def test_production_json_readers_are_banned(source: str, rel: str, expect: list[int]) -> None:
    """输入：源码 + 它所在的相对路径 + 期望行号。输出：无。

    预期行为：生产代码（`src/` `workflows/`）里直接读 JSON 就报；测试、仓库根脚本、
    `jc/jsonio.py` 自己放行；`json.dumps` 返回 str，任何时候都不报。
    """
    assert any_sites_in_source(source, rel=rel) == expect
