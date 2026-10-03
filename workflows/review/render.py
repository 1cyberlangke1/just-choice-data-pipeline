"""改动审查的文本外壳：改动清单与三张表摊平，改动正文由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **改动清单、审查规则表、严重度表、门控表由代码渲染**——它们是结构化事实（这次改动落在
  什么文件上、以及三张判据表），让模型去写只会写歪；
- 后端只写**改动正文**（真实改动里那几行代码是长尾的，模板拼的脱离现实）；
- 那段正文必须体现注入的 `secret` 与 `debug` 两维，交给 `judges.TraceJudge` 反解一次
  再比对。

与 routing 不同的是提示词**五维都给**：正文是一段代码改动，它的形状（是不是测试文件、
新增还是删除、几行）本来就必须与清单那一行对得上。泄底风险由两处挡：正文里不许出现表名
（locale 的 `banned_intent_phrases`），也不许把段落名再抄一遍（`label_leak`）。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
**本模块不执行任何代码**——它只把文本当字符串读写，改动内容里的 `print(` 只是几个字符。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc import locale_files
from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.state_parse import split_sections
from jc.workflow import DraftError

from . import features, locales
from .features import TagSet

#: 提示词里特征的固定顺序：五维全给（见模块文档）。顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = tuple(features.DIMS)

#: Markdown 围栏残留：模型偶尔把整段包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 段落名在 locale 里的键（渲染与解析共用同一份清单）。
SECTION_KEYS: Final[tuple[str, ...]] = (
    "body_label",
    "manifest_label",
    "rules_label",
    "severity_label",
    "gate_label",
)

#: 「改动清单」那一行的格数（文件类型 | 变更类型 | 规模）。
MANIFEST_CELLS: Final[int] = 3

#: 审查规则表的格数（规则名 | 条件）。
RULE_CELLS: Final[int] = 2

#: 严重度表的格数（规则名 | 源码 | 测试 | 配置 | 文档）。
SEVERITY_CELLS: Final[int] = 5

#: 门控表的格数（档位 | 新增 | 修改 | 删除）。
GATE_CELLS: Final[int] = 4

#: 「改动清单」三格的顺序（与 `features.DIMS` 里那三维的声明顺序一致）。
MANIFEST_ORDER: Final[tuple[str, ...]] = ("file_kind", "change_kind", "size")

#: 严重度表的列（文件类型）与门控表的列（变更类型）。
SEVERITY_COLUMNS: Final[tuple[str, ...]] = features.FILE_KINDS
GATE_COLUMNS: Final[tuple[str, ...]] = features.CHANGE_KINDS


class ReviewDraft(BaseModel):
    """后端要填的草稿：**只有改动正文**。

    刻意只留一个字段——清单与三张表全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    body: str = Field(
        description="这次改动的那几行：每行以 +（新增）/ -（删除）/ 空格（上下文）开头",
    )


def build_prompt(tags: TagSet, loc: dict[str, JsonValue], phrasing: int) -> str:
    """组装提示词。

    输入：一组注入特征、语言字典、问法编号。
    输出：完整提示词（五维都描述，不含答案）。
    预期行为：特征按 `DIM_ORDER` 固定顺序逐行列出；规则整段附在后面，每条前面加短横线
    并把 `{language}` 填成语言名——留着占位符会把 `{language}` 原样发给模型。
    """
    return locales.tagged_prompt(loc, tags.as_dict(), DIM_ORDER, phrasing)


def build_task(tags: TagSet, loc: dict[str, JsonValue], task_id: str) -> GenerationTask:
    """输入：注入特征、语言字典、任务 id。输出：`GenerationTask`。

    预期行为：`response_model` 是 `ReviewDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="code-review",
        prompt=build_prompt(tags, loc, 0),
        response_model=ReviewDraft,
        seeds=(),
        spec={"tags": tags.as_dict()},
    )


def label_leak(text: str, loc: dict[str, JsonValue]) -> str:
    """输入：一段文本、语言字典。输出：文本开头误写的段落名（没有就空串）。

    预期行为：只认「段落名 + 冒号」开头这一种写法——正文里正常提到这些词不算
    （模型很爱在开头再抄一遍「改动内容：」，抄进去就是一行重复标记）。查段落名的机制只有
    一份，住在 `jc.locale_files.label_leak`（大小写不敏感的口径也在那里）；本家只声明键表。
    """
    return locale_files.label_leak(text, loc, SECTION_KEYS)



#: 转出去：实现只有一份，住在 `jc.locale_files`（7 家逐字同构，只留那一份）。
banned_echo = locale_files.banned_echo


def split_body_lines(raw: str) -> list[str]:
    """输入：草稿正文。输出：非空行（只去尾部空白，**保留行首的前缀字符**）。

    预期行为：行首那一个字符就是 `+` / `-` / 空格，去掉它就分不出新增、删除与上下文了，
    所以这里只 `rstrip`；判「空行」时再看 `strip()`。
    """
    out: list[str] = []
    for line in raw.splitlines():
        text = line.rstrip()
        if text.strip():
            out.append(text)
    return out


def normalize_body_lines(lines: list[str]) -> list[str]:
    """输入：改动正文的每一行。输出：前缀补齐成规范 diff 行的每一行。

    预期行为：真机上模型有两种常见写法要收下——把整段缩进两格（Markdown 列表那种），
    标记落在缩进后面（`  - foo`）；以及上下文行忘写那个空格前缀（`foo` 而不是 ` foo`）。
    前者把标记拉回行首，后者补一个空格。**只动行首那个前缀**，代码本身一个字符都不改。
    """
    out: list[str] = []
    for line in lines:
        if features.diff_mark(line) == " ":
            out.append(line if line[:1] == " " else " " + line)
        else:
            out.append(line.lstrip())
    return out


def check_draft(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、注入特征、语言字典。输出：无；不合规就抛 `DraftError`（管线据此重渲染）。

    预期行为：正文会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查（前一条
    `body.strip()` 非空就保证了 `split_body_lines` 至少给一行，所以后面不再查空）：

    - 非空、不含 Markdown 围栏；
    - 开头不许再写一遍段落名；不许出现表名（这两条先查：它们的诊断比「前缀不对」具体）；
    - 行数与 `features.size_line_bounds(规模)` 对得上（上界是宽容过的，见 `features`）；
    - 行前缀与变更类型对得上：新增全是新增行（不许有删除行、且至少有一行 `+`）、
      删除全是删除行（不许有新增行、且至少有一行 `-`）、修改至少有一行变了
      （既没有 `+` 也没有 `-` 的「修改」等于没改）；

    前缀的读法走 `features.diff_mark`（剥掉行首空白再认标记），所以模型把整段缩进两格、
    或者上下文行忘写空格前缀，都不算坏数据——真机上这两种各占了一批（见范式 §9.19）。
    """
    body = str(data.get("body", ""))
    if not body.strip():
        raise DraftError("草稿的「改动内容」是空的")
    if FENCE_PATTERN.search(body):
        raise DraftError("草稿的「改动内容」里混进了 Markdown 围栏")
    lines = normalize_body_lines(split_body_lines(body))
    leaked = label_leak(body, loc)
    if leaked:
        raise DraftError(f"草稿的「改动内容」把段落名「{leaked}」也写进去了")
    echoed = banned_echo(body, loc)
    if echoed:
        raise DraftError(f"草稿的「改动内容」把表名写进去了：{'、'.join(echoed)}")
    low, high = features.size_line_bounds(tags.size)
    if not low <= len(lines) <= high:
        raise DraftError(
            f"草稿的「改动内容」有 {len(lines)} 行，与规模 {tags.size!r} 要求的 {low}~{high} 行对不上"
        )
    marks = {features.diff_mark(line) for line in lines}
    if tags.change_kind == "added":
        if "-" in marks:
            raise DraftError("草稿的「改动内容」是「新增」，却出现了删除行（- 前缀）")
        if "+" not in marks:
            raise DraftError("草稿的「改动内容」是「新增」，却一行 + 都没有")
    elif tags.change_kind == "deleted":
        if "+" in marks:
            raise DraftError("草稿的「改动内容」是「删除」，却出现了新增行（+ 前缀）")
        if "-" not in marks:
            raise DraftError("草稿的「改动内容」是「删除」，却一行 - 都没有")
    elif marks == {" "}:
        raise DraftError("草稿的「改动内容」是「修改」，却一行都没变（既没有 + 也没有 -）")


def manifest_row(tags: TagSet) -> tuple[str, str, str]:
    """输入：注入特征。输出：「改动清单」那一行的三格（文件类型 | 变更类型 | 规模）。"""
    return (tags.file_kind, tags.change_kind, tags.size)


def manifest_lines(tags: TagSet, loc: dict[str, JsonValue]) -> list[str]:
    """输入：注入特征、语言字典。输出：清单行（只有一行，三格都是 ASCII 词元）。"""
    return [locales.row_line(loc, manifest_row(tags))]


def rule_rows(loc: dict[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    """输入：语言字典。输出：审查规则表的规范形态 `((规则名, 条件), ...)`。

    预期行为：顺序严格按 `features.REVIEW_RULES`——这张表是**从上往下第一条命中**的
    语义，顺序变了语义就变了。条件文案取自 locale，规则名保持 ASCII。
    """
    return tuple(
        (name, str(loc["rules"][name]["condition"])) for name in features.REVIEW_RULES
    )


def rule_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：审查规则表的每一行（规则名 | 条件）。"""
    return [locales.row_line(loc, cells) for cells in rule_rows(loc)]


def rule_row_variants() -> frozenset[tuple[tuple[str, str], ...]]:
    """输入：无。输出：各语言下审查规则表规范形态的集合。

    预期行为：一次 run 里语言按样本轮转（`locales[seq % len]`），所以审计拿到的 `locale`
    未必是产物那一种；「条件」那一列是随语言渲染的文案，判定必须写「等于**任一**语言的
    规范表」，拿调用方给的那一种去比会把另一种语言的产物误判成「规则表与规范表对不上」。
    """
    return frozenset(rule_rows(locales.load(lang)) for lang in locales.available())


def severity_rows(loc: dict[str, JsonValue]) -> tuple[tuple[str, ...], ...]:
    """输入：语言字典。输出：严重度表的规范形态 `((规则名, 源码, 测试, 配置, 文档), ...)`。

    预期行为：四行（每条规则一行），四列文件类型按 `SEVERITY_COLUMNS` 的顺序排；
    档位取自 `features.SEVERITY_TABLE`，不适用的格子（文件类型不在该规则适用范围里）
    写 0 档。兜底规则那一行全是 0 档。
    """
    del loc
    out: list[tuple[str, ...]] = []
    for rule in features.REVIEW_RULES:
        cells = tuple(
            str(features.SEVERITY_TABLE[(rule, kind)])
            if kind in features.RULE_SCOPES[rule] and rule != "none"
            else str(features.NOT_RAISED)
            for kind in SEVERITY_COLUMNS
        )
        out.append((rule, *cells))
    return tuple(out)


def severity_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：严重度表的每一行（规则名 | 源码 | 测试 | 配置 | 文档）。"""
    return [locales.row_line(loc, cells) for cells in severity_rows(loc)]


def gate_rows(loc: dict[str, JsonValue]) -> tuple[tuple[str, ...], ...]:
    """输入：语言字典。输出：门控表的规范形态 `((档位, 新增, 修改, 删除), ...)`。

    预期行为：四行按档位从重到轻（3 → 0）；三列变更类型按 `GATE_COLUMNS` 的顺序排；
    动作取自 `features.GATE_TABLE`，表里缺格会在这里当场抛 KeyError。
    """
    del loc
    return tuple(
        (str(level), *(features.GATE_TABLE[(level, kind)] for kind in GATE_COLUMNS))
        for level in range(features.SEVERITY_LEVELS - 1, -1, -1)
    )


def gate_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：门控表的每一行（档位 | 新增 | 修改 | 删除）。"""
    return [locales.row_line(loc, cells) for cells in gate_rows(loc)]


def to_state(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`body`）、注入特征、语言字典。
    输出：五段文本——改动内容 / 改动清单 / 审查规则 / 严重度表 / 门控表。
    预期行为：正文先过 `normalize_body_lines` 把行首前缀补齐（只动那一个字符，代码本身
    一个字符都不改，反解校验看的是代码里的标记，不受影响）；正文独占段落头下面那几行
    （一行一段代码，拼在段头后面会读成一整行）；
    后四段**由代码写**（它们是注入特征与固定表，不是模型的发挥），所以从 `tags` 推出来。
    """
    meta = loc["meta"]
    header = locales.header
    lines = [header(loc, meta["body_label"])]
    lines.extend(normalize_body_lines(split_body_lines(str(data["body"]))))
    lines.append(
        f"{header(loc, meta['manifest_label'])}{meta['paren'].format(text=meta['manifest_columns'])}"
    )
    lines.extend(manifest_lines(tags, loc))
    lines.append(
        f"{header(loc, meta['rules_label'])}{meta['paren'].format(text=meta['rules_columns'])}"
    )
    lines.extend(rule_lines(loc))
    lines.append(
        f"{header(loc, meta['severity_label'])}{meta['paren'].format(text=meta['severity_columns'])}"
    )
    lines.extend(severity_lines(loc))
    lines.append(
        f"{header(loc, meta['gate_label'])}{meta['paren'].format(text=meta['gate_columns'])}"
    )
    lines.extend(gate_lines(loc))
    return "\n".join(lines)


@dataclass(frozen=True)
class ManifestState:
    """从「改动清单」那一行里反解出来的三格。"""

    file_kind: str
    change_kind: str
    size: str


@dataclass(frozen=True)
class ParsedReview:
    """从 state 里纯代码反解出来的事实。"""

    body_lines: tuple[str, ...]
    manifest: ManifestState | None
    rules: tuple[tuple[str, str], ...]
    severity: tuple[tuple[str, ...], ...]
    gate: tuple[tuple[str, ...], ...]

    @property
    def body(self) -> str:
        """输入：无。输出：改动内容那一段的原文（行之间用换行拼起来）。"""
        return "\n".join(self.body_lines)

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：改动内容那一段是不是空的。"""
        return not self.body.strip()


#: 转出去：切段机制只有一份，住在 `jc.state_parse.split_sections`（11 家逐字同构，只留那一份）。
def _sections(state: str, loc: dict[str, JsonValue]) -> dict[str, list[str]]:
    """输入：state、语言字典。输出：`{段落名: 该段的行列表}`。

    预期行为：段落头是**前缀匹配**（后面可能跟着括号里的列名），段头那一行的括号说明
    单独收进该段、不当数据行；其余行只去尾部空白——正文是代码，行首那个 `+` / `-` / 空格
    就是它的前缀，去掉就解不出来了。
    """
    return split_sections(state, loc, SECTION_KEYS, keep_indent=True)


def _cells(line: str, loc: dict[str, JsonValue]) -> tuple[str, ...] | None:
    """输入：一行、语言字典。输出：拆好的格子；不是数据行就 None。

    预期行为：只认 `row_prefix` 开头的行，按 `CELL_SEP` 拆并把每格去空白。
    """
    prefix = str(loc["meta"]["row_prefix"])
    text = line.strip()
    if not text.startswith(prefix):
        return None
    rest = text[len(prefix) :].strip()
    return tuple(cell.strip() for cell in rest.split(locales.CELL_SEP))


def body_lines(state: object, loc: dict[str, JsonValue]) -> list[str]:
    """输入：state、语言字典。输出：改动内容那一段的每一行（非文本 state 给空列表）。"""
    if not isinstance(state, str):
        return []
    return list(_sections(state, loc).get(str(loc["meta"]["body_label"]), []))


def body_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：改动内容那一段的原文（没有就空串）。"""
    return "\n".join(body_lines(state, loc))


def manifest_of(state: object, loc: dict[str, JsonValue]) -> ManifestState | None:
    """输入：state、语言字典。输出：解析出来的清单行（解不出就 None）。

    预期行为：只认形如 `- source | modified | medium` 的三格行；词元不在取值表里就当
    解不出来——判别器与审计会拿它跟注入特征逐格比对，串味会当场暴露。
    """
    if not isinstance(state, str):
        return None
    for line in _sections(state, loc).get(str(loc["meta"]["manifest_label"]), []):
        cells = _cells(line, loc)
        if cells is None:
            continue  # 段落头那一行（列名）不是数据行，跳过
        if len(cells) != MANIFEST_CELLS:
            return None
        file_kind, change_kind, size = cells
        if (
            file_kind in features.FILE_KINDS
            and change_kind in features.CHANGE_KINDS
            and size in features.SIZES
        ):
            return ManifestState(file_kind=file_kind, change_kind=change_kind, size=size)
        return None
    return None


def rules_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    """输入：state、语言字典。输出：解析出来的审查规则表（解不出就空元组）。"""
    if not isinstance(state, str):
        return ()
    out: list[tuple[str, str]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["rules_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != RULE_CELLS:
            continue
        name, condition = cells
        if name in features.REVIEW_RULES:
            out.append((name, condition))
    return tuple(out)


def severity_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, ...], ...]:
    """输入：state、语言字典。输出：解析出来的严重度表（解不出就空元组）。

    预期行为：只认五格行且首格是规则名、后四格都是 0~9 的单个数字；一行对不上就跳过。
    """
    if not isinstance(state, str):
        return ()
    out: list[tuple[str, ...]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["severity_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != SEVERITY_CELLS:
            continue
        if cells[0] not in features.REVIEW_RULES:
            continue
        if not all(cell.isdigit() and len(cell) == 1 for cell in cells[1:]):
            continue
        out.append(tuple(cells))
    return tuple(out)


def gate_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, ...], ...]:
    """输入：state、语言字典。输出：解析出来的门控表（解不出就空元组）。

    预期行为：只认四格行且首格是**档位范围内**的单个数字、后三格都是门控动作词元；
    一行对不上就跳过（`- 5 | …` 那种越界档位也算对不上，否则审计会把假表当好表）。
    """
    if not isinstance(state, str):
        return ()
    out: list[tuple[str, ...]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["gate_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != GATE_CELLS:
            continue
        if not (cells[0].isdigit() and len(cells[0]) == 1):
            continue
        if not 0 <= int(cells[0]) < features.SEVERITY_LEVELS:
            continue
        if not all(cell in features.GATES for cell in cells[1:]):
            continue
        out.append(tuple(cells))
    return tuple(out)


def parse_state(state: object) -> ParsedReview:
    """输入：state。输出：纯代码能反解的部分（正文 + 清单 + 三张表）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    empty = ParsedReview(body_lines=(), manifest=None, rules=(), severity=(), gate=())
    if not isinstance(state, str):
        return empty
    for lang in locales.available():
        loc = locales.load(lang)
        body = body_lines(state, loc)
        manifest = manifest_of(state, loc)
        if body or manifest is not None:
            return ParsedReview(
                body_lines=tuple(body),
                manifest=manifest,
                rules=rules_of(state, loc),
                severity=severity_of(state, loc),
                gate=gate_of(state, loc),
            )
    return empty
