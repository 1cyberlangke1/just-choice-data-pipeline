"""浏览器 / GUI 操作的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：那句话必须存在、单行、不含围栏 / HTML 标签 / 段落名残留；
- `FrameJudge`（零 token）：元素表必须逐格等于按注入特征算出来的规范表，门控表与意图表
  必须逐行等于代码写的那两张——渲染串味（挂错 tags、漏行、错格）全在这一关拦下；
- `TraceJudge`（一次后端调用）：把整屏快照交回后端反解**用户想干什么**、**指的是第几行**、
  **那个元素叫什么**，三个答案都必须与注入特征一致。这是「渲染必须可反解」的落地
  （PLAN 3.4 第 ③ 档）：那句话是模型写的，唯一由它承载的金标签就是意图，而「指的是谁」
  必须能从那句话里读回来，否则这道题没人答得对。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，
也不要一条标签和文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后
还拿不到草稿，只能记成 `skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, Field

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict
from jc.jsontypes import JsonValue

from . import features, locales, render
from .spec import BrowserSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签（围栏在那句话里由 `render.check_draft` 与
#: 卫生判别器各查一次；这里再按「单独成行」查一遍整段 state——模型偶尔会把整段回答
#: 包进 ``` 里，那样的行会落在那句话那一段之外）。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")

#: 反解时重问的原子题族（题族名与数据集里完全一致，选项也用同一份 locale）。
TRACE_FAMILIES: Final[tuple[str, ...]] = ("intent", "row", "label")


class TraceDraft(BaseModel):
    """反解草稿：反解「用户想干什么」「指的是第几行」「那个元素叫什么」。

    取值域必须与 `features.INTENTS` 以及 locale 里 `fam.intent.criteria` 的 key 一致；
    这条由 `tests/workflows/browser/test_locales.py` 锁住，不靠人记。
    """

    intent: Literal["fill", "open", "switch", "submit"]
    row: int = Field(ge=1, description="用户指的是「界面元素」表里的第几行（从 1 数起）")
    label: str = Field(min_length=1, description="用户指的那个元素名（原样照抄）")


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。
    输出：提示词。预期行为：把 `TRACE_FAMILIES` 里的原子题连同选项原样再问一遍——
    反解用的选项与数据集里的一致，否则「答对了」与「标签对了」不是一回事。
    """
    lines = [
        str(loc["verify"]["intro"]),
        "",
        str(loc["verify"]["state_header"]),
        state,
        "",
        str(loc["verify"]["questions_header"]),
        f"1. {loc['verify']['q_intent']}",
    ]
    for name, desc in loc["fam"]["intent"]["criteria"].items():
        lines.append(str(loc["verify"]["option_line"]).format(name=name, desc=desc))
    lines.append(f"2. {loc['verify']['q_row']}")
    lines.append(f"3. {loc['verify']['q_label']}")
    return "\n".join(lines)


def _has_leftover(state: str) -> str:
    """输入：state 文本。输出：命中的残留（没有就空串）。"""
    for line in state.splitlines():
        if line.strip() in LEFTOVER_LINES:
            return line.strip()
    for tag in LEFTOVER_TAGS:
        if tag in state:
            return tag
    return ""


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：state 里不许有格式残留，那句话还得有内容、是单行。

    查四类：**单独成行的围栏 / HTML 换行标签**、**那句话是空的**、**那句话是多行**、
    **那句话开头又写了一遍段落名**。口径压到最窄——宁可漏检也不误判，
    因为误判会直接拒掉好数据。

    字段：`locale` 语言（要按语言认那一段的段落名）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只看残留与形状，不看语义。"""
        del spec
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过卫生检查")
        loc = locales.load(self.locale)
        leftover = _has_leftover(state)
        if leftover:
            return Verdict(False, reason=f"state 里有格式残留：{leftover!r}")
        lines = render.request_lines(state, loc)
        if not lines:
            return Verdict(False, reason="state 里没有「用户请求」那一段，或者那句话是空的")
        if len(lines) > 1:
            return Verdict(False, reason=f"用户那句话是多行（{len(lines)} 行）——它必须是一行")
        leaked = render.label_leak(lines[0], loc)
        if leaked:
            return Verdict(False, reason=f"用户那句话把段落名「{leaked}」也写进去了")
        return Verdict(True, reason="state 干净：那句话单行、无围栏、无段落名残留")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：三张表必须与代码算出来的一模一样。

    字段：`locale` 语言。预期行为：逐格比对元素表（行号 / 角色 / 元素名 / 状态），
    再逐行比对门控表与意图表；任何一处对不上都判不通过——那是渲染挂错了 tags，
    标签会跟着一起错。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做结构比对。"""
        if not isinstance(spec, BrowserSpec):
            return Verdict(True, reason="没有 spec，跳过框架比对", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过框架比对", skipped=True)
        loc = locales.load(self.locale)
        problems: list[str] = []
        elements = render.elements_of(state, loc)
        expected_elements = render.canonical_elements(spec.tags)
        if elements != expected_elements:
            problems.append(
                f"元素表与注入特征对不上：解析出 {len(elements or ())} 行，"
                f"应为 {len(expected_elements)} 行"
            )
        gates = render.gates_of(state, loc)
        if gates != render.gate_rows(loc):
            problems.append(f"门控表与规范表对不上：解析出 {len(gates or ())} 行，应为 4 行")
        criteria = render.criteria_of(state, loc)
        if criteria != render.criteria_rows():
            problems.append(
                f"意图表与规范表对不上：解析出 {len(criteria or ())} 行，应为 4 行"
            )
        if not problems:
            return Verdict(True, reason="元素表与两张判据表都与注入特征一致")
        return Verdict(False, reason="；".join(problems), raw={"elements": str(elements)})


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整屏快照交回后端反解三件事，必须与注入的一致。

    字段：`backend` 生成后端（由 `bind_backend` 注入）；`locale` 语言。
    """

    backend: GenerationBackend
    locale: str
    name: str = "trace"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：不一致时把「注入的」与「反解出来的」列在理由里，方便人看是哪里写歪了。
        端点故障与草稿缺字段都只算「没判成」（`skipped=True`）——记成「不通过」会把
        端点抖动写成数据质量问题。
        """
        if not isinstance(spec, BrowserSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        if self.backend is None:
            return Verdict(True, reason="没有后端，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="text",
            domain="browser-gui-action",
            prompt=build_trace_prompt(str(sample["state"]), loc),
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
        expected = {
            "intent": spec.tags.intent,
            "row": features.wanted_index(spec.tags),
            "label": features.wanted_label(spec.tags),
        }
        got = {
            "intent": data["intent"],
            "row": data["row"],
            "label": str(data["label"]).strip(),
        }
        problems = [
            f"{name}: 注入 {expected[name]!r}，反解成 {got[name]!r}"
            for name in TRACE_FAMILIES
            if got[name] != expected[name]
        ]
        return Verdict(not problems, reason="; ".join(problems), raw={"traced": got})


def option_names(question: dict[str, JsonValue]) -> list[str]:
    """输入：一道 choice 题。输出：按**产物里写的顺序**排好的选项名。

    预期行为：答案编号就是在这个顺序里的下标（见 `jc.assemble`）——审计要拿它把编号
    换回选项名，才能与纯代码重算的结果比对。
    """
    criteria = question.get("criteria")
    return list(criteria) if isinstance(criteria, dict) else []