"""命令审批 workflow 的文本外壳：把特征清单交给后端写成一条待审命令。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- 提示词**只描述注入特征**，不含任何答案、题号或标签；
- 后端回三个可见部分（用户目标 / 命令 / 备注），拼成 state；
- 文本是否真的体现了这些特征，交给 `judges.TraceJudge` 反解一次再比对——反解不出来
  就不猜，直接丢（`docs_local/可靠合成数据范式.md` §9 的纪律）。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
**本模块绝不执行任何命令**——它只把命令当字符串读。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc import locale_files
from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from . import locales
from .features import TagSet

#: 提示词里特征的固定顺序：顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("dialect", "goal", "target", "elevation")

#: Markdown 围栏残留：模型偶尔把整条命令包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 提权写法（bash / PowerShell 各一套）。只认小写比对，调用方先 `.lower()`。
ELEVATION_MARKERS: Final[tuple[str, ...]] = (
    "sudo",
    "doas",
    "pkexec",
    "runas",
    "start-process -verb runas",
    "管理员",
)

#: PowerShell 的 `动词-名词` cmdlet：动词表 + **后面必须真的接一个字母**。
#: 用正则而不是裸子串：`grep -rn "test-"` 这种命令里带引号的词会被子串匹配误伤，
#: 而误判会直接拒掉好数据。
POWERSHELL_CMDLET: Final[re.Pattern[str]] = re.compile(
    r"\b(?:get|set|remove|new|copy|move|select|invoke|restart|stop|start|clear|test|out)"
    r"-[a-z][a-z0-9]*",
    re.IGNORECASE,
)

#: 与方言无关的外部工具：这些命令在 bash 与 PowerShell 里写法一样，
#: 不能因为它们「没有 cmdlet」就判成方言不符（`git reset`、`docker rm` 都是这样）。
EXTERNAL_TOOLS: Final[tuple[str, ...]] = (
    "git",
    "docker",
    "podman",
    "psql",
    "mysql",
    "redis-cli",
    "ssh",
    "scp",
    "kubectl",
    "aws",
    "gcloud",
    "curl",
    "wget",
)


class ShellDraft(BaseModel):
    """后端要填的草稿：一条待审命令的三个可见部分。

    字段刻意全是字符串——后端**只写文本**，编号、题号与 `type` 一律由组装器产生
    （PLAN 3.1 不变量①）。

    「命令」单独占一个字段而不是塞进备注里：提示词里写十遍「单行」，不如让它少一个
    可发挥的字段；`check_draft` 会强制三个字段都非空、单行、且不带格式残留。
    """

    goal_line: str = Field(description="用户想干什么：一句话，带具体的路径或对象名")
    command: str = Field(description="要执行的命令：**单行**，可直接粘进终端")
    note: str = Field(description="这条命令会做什么：一句话，客观描述，不要评价该不该批准")


@dataclass(frozen=True)
class ParsedCommand:
    """从 state 里纯代码反解出来的事实。"""

    command: str

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：命令那一行是不是空的。"""
        return not self.command.strip()


def command_label(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：「拟执行命令」那一行的行首标记（含冒号）。

    预期行为：只此一处定义解析口径——渲染器与判别器共用它，两处各写一份迟早会漂移。
    """
    return f"{loc['meta']['command_label']}{loc['meta']['colon']}"


def label_prefixes(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：四类行首标记（环境 / 目标 / 命令 / 备注）。

    预期行为：只此一处列出「哪些行首标记算新字段」——判别器要判「命令后面那一行
    是不是命令的续行」，就得先知道这一行是不是别的字段。
    """
    meta = loc["meta"]
    return tuple(
        f"{meta[key]}{meta['colon']}"
        for key in ("env_label", "goal_label", "command_label", "note_label")
    )


def build_prompt(tags: TagSet, loc: dict[str, JsonValue], phrasing: int) -> str:
    """组装提示词。

    输入：一组注入特征、语言字典、问法编号。
    输出：完整提示词（只描述特征，不含答案）。
    预期行为：特征按 `DIM_ORDER` 固定顺序逐行列出；规则整段附在后面。
    """
    values = tags.as_dict()
    lines = [loc["prompt"]["intro"], ""]
    lines.extend(locales.feature_line(loc, dim, values[dim]) for dim in DIM_ORDER)
    lines.append("")
    rules = loc["prompt"]["rules"]
    offset = phrasing % len(rules)
    rotated = list(rules[offset:]) + list(rules[:offset])
    lines.extend(f"- {rule.format(language=loc['language_name'])}" for rule in rotated)
    return "\n".join(lines)


def build_task(tags: TagSet, loc: dict[str, JsonValue], task_id: str) -> GenerationTask:
    """输入：注入特征、语言字典、任务 id。输出：`GenerationTask`。

    预期行为：`response_model` 是 `ShellDraft`（instructor 要 pydantic 模型）；
    `seeds` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="shell-approval",
        prompt=build_prompt(tags, loc, 0),
        response_model=ShellDraft,
        seeds=(),
        spec={"tags": tags.as_dict()},
    )


def check_command(data: dict[str, JsonValue]) -> None:
    """输入：草稿字典。输出：无；不合规就抛 `DraftError`（管线据此重渲染）。

    预期行为：命令必须非空、**单行**、不含 Markdown 围栏。这三条不能只靠提示词——
    实跑里模型爱把命令包进 ``` 或写成多行脚本，那样的 state 没法当「一条命令」审。
    """
    command = str(data.get("command", "")).strip()
    if not command:
        raise DraftError("草稿的「拟执行命令」是空的")
    # 先判围栏：模型爱把命令整条包进 ``` 里，围栏里的换行是它的**结果**而不是独立毛病，
    # 先报「混进围栏」才能指向真正的病因（否则只会看到一句「多行」去瞎猜）。
    if FENCE_PATTERN.search(command):
        raise DraftError("草稿的「拟执行命令」里混进了 Markdown 围栏")
    if "\n" in command.strip():
        raise DraftError("草稿的「拟执行命令」是多行——待审命令必须是单行")
    if command.startswith("#!"):
        raise DraftError("草稿的「拟执行命令」以 shebang 开头——那是脚本，不是一条命令")
    # 实跑里模型把 PowerShell 的格式化表达式当成命令交回来过（`{0:N2} MB -f (…)`）：
    # 那以 `{` 开头，是表达式 / 脚本块，任何合法命令都不会这么起头。只认首字符，
    # 不做别的语法判断——没有解析器时，宽口径的猜测会把好命令拒掉。
    if command.startswith(("{", "}")):
        raise DraftError("草稿的「拟执行命令」以 `{` 或 `}` 开头——那是表达式或脚本块，不是一条命令")
    # 实跑里模型把「一句话描述」写进命令字段过（`查看 $HOME/… 目录下的文件，获取列表信息。`）：
    # 合法命令都以 ASCII 字符起头（命令名 / 路径 / 引号 / 重定向符），首字符是汉字就说明它写的
    # 不是命令。同样只认首字符——没有解析器时，宽口径的猜测会把好命令拒掉。
    if not command[0].isascii():
        raise DraftError("草稿的「拟执行命令」以非 ASCII 字符开头——那是描述，不是一条命令")


#: 本家 `label_leak` 查的键表：四个字段的行首标记都由代码写，模型再抄一遍就是重复标记。
LEAK_KEYS: Final[tuple[str, ...]] = ("env_label", "goal_label", "command_label", "note_label")


def label_leak(text: str, loc: dict[str, JsonValue]) -> str:
    """输入：一段文本、语言字典。输出：文本开头误写的字段名（没有就空串）。

    预期行为：模型很爱把字段名自己写进内容里（`用户目标：用户目标：查看…`），
    只认「字段名 + 冒号」开头这一种写法——正文里正常提到「环境」这类词不算。查标记名的
    机制只有一份，住在 `jc.locale_files.label_leak`（大小写不敏感的口径也在那里）。
    """
    return locale_files.label_leak(text, loc, LEAK_KEYS)


def check_draft(data: dict[str, JsonValue], loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、语言字典。输出：无；不合规就抛 `DraftError`（管线据此重渲染）。

    预期行为：三个可见字段都会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：
    - 「用户目标」「备注」非空、单行，且开头不许再写一遍字段名；
    - 「用户目标」不许写成一整条命令——判据是「与命令同一个词起头」。实跑 10 条里有
      4 条把目标行写成了命令原文，这类样本的「目标说明」是假的（`docs_local/可靠合成数据范式.md`
      的纪律：宁可让重渲染多花一次配额，也不收一条文本与意图对不上的数据）；
    - 「拟执行命令」见 `check_command`：非空、单行、无围栏、无 shebang。
    """
    check_command(data)
    goal = str(data.get("goal_line", "")).strip()
    note = str(data.get("note", "")).strip()
    for name, text in (("用户目标", goal), ("备注", note)):
        if not text:
            raise DraftError(f"草稿的「{name}」是空的")
        if "\n" in text:
            raise DraftError(f"草稿的「{name}」是多行——它必须是单行")
        leaked = label_leak(text, loc)
        if leaked:
            raise DraftError(f"草稿的「{name}」把字段名「{leaked}」也写进去了")
    if first_token(goal) == first_token(str(data.get("command", ""))):
        raise DraftError("草稿的「用户目标」写成了命令（与命令同一个词起头）")


def elevated_in(command: str) -> bool:
    """输入：一条命令。输出：命令里有没有提权写法。

    预期行为：大小写不敏感；只认 `ELEVATION_MARKERS` 里的写法，不做模糊匹配
    （误判会直接拒掉好数据）。
    """
    lowered = command.lower()
    return any(marker in lowered for marker in ELEVATION_MARKERS)


def powershell_verb_in(command: str) -> bool:
    """输入：一条命令。输出：命令里有没有 PowerShell 的 `动词-名词` cmdlet 写法。"""
    return POWERSHELL_CMDLET.search(command) is not None


def first_token(command: str) -> str:
    """输入：一条命令。输出：第一个词（小写；空命令 → 空串）。

    预期行为：用来判「这条命令是不是一个与方言无关的外部工具调用」。
    """
    parts = command.strip().split()
    return parts[0].lower() if parts else ""


def external_tool_in(command: str) -> bool:
    """输入：一条命令。输出：第一个词是不是与方言无关的外部工具。"""
    return first_token(command) in EXTERNAL_TOOLS


def to_state(data: dict[str, JsonValue], loc: dict[str, JsonValue], dialect: str) -> str:
    """把草稿拼成 state。

    输入：草稿字典（`goal_line` / `command` / `note`）、语言字典、方言。
    输出：四行文本——环境 / 用户目标 / 拟执行命令 / 备注。
    预期行为：只拼接与去首尾空白，不改写模型写的字（反解校验要靠原文）；
    「环境」那一行由渲染器写（方言是注入特征，不是模型的发挥，所以从 `tags` 传进来）。
    """
    meta = loc["meta"]
    return "\n".join(
        [
            f"{meta['env_label']}{meta['colon']}{dialect}",
            f"{meta['goal_label']}{meta['colon']}{str(data['goal_line']).strip()}",
            f"{command_label(loc)}{str(data['command']).strip()}",
            f"{meta['note_label']}{meta['colon']}{str(data['note']).strip()}",
        ]
    )


def command_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：「拟执行命令」那一行的内容（没有那一行 → 空串）。

    预期行为：只认行首标记、只取标记之后的文本（去空白）——命令归渲染器写，
    判别器读它就够了，不需要重新理解整段文本。
    """
    if not isinstance(state, str):
        return ""
    prefix = command_label(loc)
    for line in state.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return ""


def command_of_any(state: object) -> str:
    """输入：state。输出：命令那一行（把各语言的标记都试一遍）。

    预期行为：`reparse` 的协议签名没有语言参数，而命令那一行的标记随语言变
    （「拟执行命令：」/「Proposed command: 」），所以这里逐语言试；命中即返回。
    标记互不重叠（中文标点 vs ASCII 冒号），不会串味。
    """
    if not isinstance(state, str):
        return ""
    for lang in locales.available():
        found = command_of(state, locales.load(lang))
        if found:
            return found
    return ""


#: 「模型写的自然语言正文」那两行：结构键 → 语言文件里的行首标记名。
#: 命令那一行**不在这里**——命令、路径与标识符按约定保持 ASCII，喂给语言判据只会把
#: 好数据当坏数据。
PROSE_LINES: Final[tuple[tuple[str, str], ...]] = (("用户目标", "goal_label"), ("备注", "note_label"))


def prose_of(state: object, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：state、语言字典。输出：{行名: 那一行的文本}（缺哪一行就不放哪个键）。

    预期行为：只认行首标记、只取标记之后的文本，与 `command_of` 同一套路。
    """
    out: dict[str, str] = {}
    if not isinstance(state, str):
        return out
    meta = loc["meta"]
    for name, label in PROSE_LINES:
        prefix = f"{meta[label]}{meta['colon']}"
        for line in state.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                out[name] = stripped[len(prefix) :].strip()
                break
    return out


def prose_of_any(state: object) -> dict[str, str]:
    """输入：state。输出：两格自然语言正文（把各语言的标记都试一遍）。

    预期行为：`language_texts` 是**只读样本**的纯函数、拿不到调用方的语言，而这两行的
    标记随语言变（「用户目标：」/「User goal: 」），所以逐语言试；命中即返回。
    标记互不重叠（中文标点 vs ASCII 冒号），不会串味——与 `command_of_any` 同理。
    """
    if not isinstance(state, str):
        return {}
    for lang in locales.available():
        found = prose_of(state, locales.load(lang))
        if found:
            return found
    return {}


def parse_state(state: object) -> ParsedCommand:
    """输入：state。输出：纯代码能反解的部分（命令那一行）。预期行为：与语言无关。"""
    return ParsedCommand(command=command_of_any(state))
