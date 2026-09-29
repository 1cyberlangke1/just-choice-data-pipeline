"""命令审批 workflow 的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `DialectJudge`（零 token）：命令必须与注入的方言一致——bash 里不许出现 PowerShell 的
  `动词-名词` cmdlet。只做**单向**检查（PowerShell 侧只要求至少出现一个 cmdlet）：
  误判会直接拒掉好数据，而漏检只是少拒几条；
- `DestructiveJudge`（零 token）：`destructive` 那一族的真值，纯代码就能反解——
  命令里有没有破坏性写法，与注入标签比；
- `ElevatedJudge`（零 token）：`elevated` 那一族的真值，同理，认提权写法；
- `HygieneJudge`（零 token）：state 里不许出现 Markdown 围栏、HTML 标签、多行命令；
- `TraceJudge`（一次后端调用）：把待审命令**交回后端**，把目标位置那一族原样再问一遍，
  答案必须与注入特征一致。这是「渲染必须可反解」的落地（PLAN 3.4 第 ③ 档）。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，
也不要一条标签和文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后
还拿不到草稿，只能记成 `skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict
from jc.jsontypes import JsonValue

from . import features, locales, render
from .spec import ShellSpec

#: 破坏性写法（大小写不敏感）。**用词边界匹配**：`rm ` 直接子串匹配会把 `firm ` 也算进去。
DESTRUCTIVE_PATTERNS: Final[tuple[str, ...]] = (
    r"\brm\b",
    r"\brmdir\b",
    r"\bshred\b",
    r"\btruncate\b",
    r"\bclear-content\b",
    r"\bgit\s+reset\b",
    r"\bdrop\s+table\b",
    r"\bdocker\s+(container\s+)?rm\b",
    r"\bremove-item\b",
    r"\bdel\s+/[sfq]\b",
    r"\berase\b",
    r"\bunlink\b",
)

#: 破坏性写法之外的其它「会毁掉数据」的信号：把文件截成 0 字节的两种 bash 写法。
#: `> file` 单独成一条命令就是截断（实跑里模型就这么写 `truncate_file`，被漏检过一次）；
#: 用 `^>\s+` 而不是 `^>`，`>> file` 是**追加**、不毁数据，不该算进来。
TRUNCATE_PATTERNS: Final[tuple[str, ...]] = (r":\s*>\s*\S+", r"^>\s+\S+")

#: 格式残留：单独成行的围栏 / HTML 换行标签。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset({"```", "```bash", "```powershell", "```sh"})
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")

#: 反解时重问的原子题族（题族名与数据集里完全一致，选项也用同一份 locale）。
TRACE_FAMILIES: Final[tuple[str, ...]] = ("target_owner",)


def destructive_in(command: str) -> bool:
    """输入：一条命令。输出：命令里有没有破坏性写法。

    预期行为：只认 `DESTRUCTIVE_PATTERNS` 与 `TRUNCATE_PATTERNS`，用词边界匹配。
    """
    lowered = command.lower()
    if any(re.search(pattern, lowered) for pattern in DESTRUCTIVE_PATTERNS):
        return True
    return any(re.search(pattern, lowered) for pattern in TRUNCATE_PATTERNS)


def command_continues(state: str, loc: dict[str, JsonValue]) -> bool:
    """输入：state、语言字典。输出：命令那一行后面是不是还跟着命令的续行。

    预期行为：`render.command_of` 只取一行，多行命令的**第二行起**会掉在它后面；那些行
    若不是任何已知字段的行首标记，就说明模型把命令写成了多行脚本（实跑里确实会这么干，
    所以 `render.check_command` 在渲染时就先拒一次，这里是产物侧的兜底）。
    """
    prefix = render.command_label(loc)
    lines = [line.strip() for line in state.splitlines()]
    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        return bool(following) and not following.startswith(render.label_prefixes(loc))
    return False


class TraceDraft(BaseModel):
    """反解草稿：反解要问的原子特征取值。

    取值域必须与 `features.TARGETS` 以及 locale 里 `criteria` 的 key 一致；
    这条由 `tests/workflows/shell/test_locales.py` 锁住，不靠人记。
    """

    target_owner: Literal["workdir", "system_path", "production", "container", "remote"]


def build_verify_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的待审命令文本、语言字典。
    输出：提示词。预期行为：把 `TRACE_FAMILIES` 里的原子题连同选项原样再问一遍——
    反解用的选项与数据集里的一致，否则「答对了」与「标签对了」不是一回事。
    """
    lines = [
        loc["verify"]["intro"],
        "",
        loc["verify"]["state_header"],
        state,
        "",
        loc["verify"]["questions_header"],
    ]
    for index, family in enumerate(TRACE_FAMILIES, start=1):
        lines.append(f"{index}. {loc['fam'][family]['q'][0]}")
        for name, desc in loc["fam"][family]["criteria"].items():
            lines.append(loc["verify"]["option_line"].format(name=name, desc=desc))
    return "\n".join(lines)


@dataclass(frozen=True)
class DialectJudge:
    """纯规则判别器：命令的写法必须与注入的方言一致。

    字段：`name`。
    预期行为：**只做单向检查**——bash 侧不许出现 PowerShell 的 `动词-名词` cmdlet；
    PowerShell 侧只要求至少出现一个 cmdlet（别名 `ls` / `rm` 在 PowerShell 里也合法，
    拿 bash 词表去禁会误伤好数据）。
    """

    name: str = "dialect"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只看写法，不看语义。"""
        if not isinstance(spec, ShellSpec):
            return Verdict(True, reason="没有 spec，跳过方言判别", skipped=True)
        # 命令那一行的标记**随语言变**（「拟执行命令：」/「Proposed command: 」），
        # 写死一种语言会让另一种语言的样本解析出空串，所以逐语言试。
        command = render.command_of_any(sample["state"])
        has_cmdlet = render.powershell_verb_in(command)
        if spec.tags.dialect == "bash":
            if has_cmdlet:
                return Verdict(
                    False, reason=f"注入的是 bash，命令里却出现了 PowerShell cmdlet：{command!r}"
                )
            return Verdict(True, reason="bash 写法一致")
        if has_cmdlet or render.external_tool_in(command):
            return Verdict(True, reason="PowerShell 写法一致")
        return Verdict(
            False,
            reason=f"注入的是 PowerShell，命令里既没有 cmdlet 也不是外部工具调用：{command!r}",
        )


@dataclass(frozen=True)
class DestructiveJudge:
    """零 token 判别器：`destructive` 那一族的真值，纯代码就能反解。"""

    name: str = "destructive"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：数命令里的破坏性写法。"""
        if not isinstance(spec, ShellSpec):
            return Verdict(True, reason="没有 spec，跳过破坏性判别", skipped=True)
        command = render.command_of_any(sample["state"])
        found = destructive_in(command)
        expected = features.is_destructive(spec.tags.goal)
        if found == expected:
            return Verdict(True, reason="破坏性与注入特征一致")
        return Verdict(
            False,
            reason=(
                f"注入的动作 {spec.tags.goal!r} 应当{'是' if expected else '不是'}破坏性的，"
                f"命令 {command!r} 读出来却是{'破坏性' if found else '非破坏性'}"
            ),
            raw={"command": command},
        )


@dataclass(frozen=True)
class ElevatedJudge:
    """零 token 判别器：`elevated` 那一族的真值，纯代码就能反解。"""

    name: str = "elevated"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：认提权写法。"""
        if not isinstance(spec, ShellSpec):
            return Verdict(True, reason="没有 spec，跳过提权判别", skipped=True)
        command = render.command_of_any(sample["state"])
        found = render.elevated_in(command)
        expected = spec.tags.elevation == "elevated"
        if found == expected:
            return Verdict(True, reason="提权与注入特征一致")
        return Verdict(
            False,
            reason=(
                f"注入的提权档是 {spec.tags.elevation!r}，"
                f"命令 {command!r} 读出来却是{'带提权' if found else '不带提权'}"
            ),
            raw={"command": command},
        )


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：state 里不许出现格式残留。

    查三类：**单独成行的 Markdown 围栏**、**HTML 换行标签**、**命令那一行是空的或多行的**。
    口径压到最窄——宁可漏检也不误判，因为误判会直接拒掉好数据。

    字段：`locale` 语言（要按语言认「拟执行命令」那一行的行首标记）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只看残留与形状，不看语义。"""
        del spec
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过卫生检查")
        leftovers = sorted(
            {line.strip() for line in state.splitlines() if line.strip() in LEFTOVER_LINES}
        )
        lowered = state.lower()
        tags = [tag for tag in LEFTOVER_TAGS if tag in lowered]
        loc = locales.load(self.locale)
        command = render.command_of(state, loc)
        problems: list[str] = []
        if not command:
            problems.append("「拟执行命令」那一行是空的")
        if command_continues(state, loc):
            problems.append("「拟执行命令」是多行")
        if not leftovers and not tags and not problems:
            return Verdict(True, reason="没有格式残留")
        return Verdict(
            False,
            reason="state 里混进了格式残留：" + "、".join([*leftovers, *tags, *problems]),
            raw={"leftovers": leftovers, "tags": tags, "problems": problems},
        )


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把待审命令交回后端反解，目标位置必须与注入的一致。

    字段：`backend` 生成后端（由 `bind_backend` 注入）；`locale` 语言。
    """

    backend: GenerationBackend
    locale: str
    name: str = "trace"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：不一致时把「注入的」与「反解出来的」列在理由里，方便人看是哪一维写歪了。
        端点故障与草稿缺字段都只算「没判成」（`skipped=True`）——记成「不通过」会把
        端点抖动写成数据质量问题。
        """
        if not isinstance(spec, ShellSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        if self.backend is None:
            return Verdict(True, reason="没有后端，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="text",
            domain="shell-approval",
            prompt=build_verify_prompt(sample["state"], loc),
            response_model=TraceDraft,
            spec={"tags": spec.tags.as_dict()},
        )
        try:
            draft = self.backend.generate(task)
        except BackendError as exc:
            if getattr(exc, "fatal_for_run", False):
                raise  # 全部 provider 都挂了：整批该停下（PLAN 3.1.4）
            return Verdict(
                True,
                reason=f"反解没跑成（端点故障，不是数据问题）：{exc}",
                raw={"error": type(exc).__name__},
                skipped=True,
            )
        data = draft.data if isinstance(draft.data, dict) else {}
        broken = [name for name in TRACE_FAMILIES if name not in data]
        if broken:
            return Verdict(
                True,
                reason=f"反解草稿缺字段，判不了：{'、'.join(broken)}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        labels = spec.labels
        problems = [
            f"{family}: 注入 {getattr(labels, family)!r}，反解成 {data[family]!r}"
            for family in TRACE_FAMILIES
            if data[family] != getattr(labels, family)
        ]
        return Verdict(
            not problems,
            reason="; ".join(problems),
            raw={"traced": {family: data[family] for family in TRACE_FAMILIES}},
        )
