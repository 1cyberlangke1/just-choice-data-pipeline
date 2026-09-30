"""工具调用门控的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：意图那一行必须存在、单行、不含围栏 / HTML 标签 / 段落名残留；
- `FrameJudge`（零 token）：状态行必须逐格等于注入特征（作用范围 / 策略档位 / 审批状态），
  策略表必须逐行等于规范表——渲染串味（挂错 tags、漏行、错格）全在这一关拦下；
- `TraceJudge`（一次后端调用）：把整段请求交回后端反解**工具**与**内容性质**，两个答案
  都必须与注入特征一致。这是「渲染必须可反解」的落地（PLAN 3.4 第 ③ 档）：意图是模型
  写的，唯一由文本承载的两个标签就是这两维，它们必须能从文本里读回来，否则这道题
  没人答得对。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，
也不要一条标签和文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后
还拿不到草稿，只能记成 `skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict
from jc.jsontypes import JsonValue

from . import locales, render
from .spec import ToolGateSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签（围栏在意图行里由 `render.check_draft` 与
#: 卫生判别器各查一次；这里再按「单独成行」查一遍整段 state——模型偶尔会把整段回答
#: 包进 ``` 里，那样的行会落在意图段之外）。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")

#: 反解时重问的原子题族（题族名与数据集里完全一致，选项也用同一份 locale）。
TRACE_FAMILIES: Final[tuple[str, ...]] = ("tool", "payload")


class TraceDraft(BaseModel):
    """反解草稿：反解「工具」与「内容性质」两维。

    取值域必须与 `features.TOOLS` / `features.PAYLOADS` 以及 locale 里 `criteria`
    的 key 一致；这条由 `tests/workflows/toolgate/test_locales.py` 锁住，不靠人记。
    """

    tool: Literal[
        "read_file",
        "write_file",
        "run_command",
        "http_request",
        "send_message",
    ]
    payload: Literal["benign", "sensitive", "harmful"]


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
    ]
    for index, family in enumerate(TRACE_FAMILIES, start=1):
        lines.append(f"{index}. {loc['fam'][family]['q'][0]}")
        for name, desc in loc["fam"][family]["criteria"].items():
            lines.append(str(loc["verify"]["option_line"]).format(name=name, desc=desc))
    return "\n".join(lines)


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：state 里不许有格式残留，意图那一行还得有内容、是单行。

    查三类：**单独成行的围栏**、**HTML 换行标签**、**意图那一行是空的 / 多行 / 带段落名**。
    口径压到最窄——宁可漏检也不误判，因为误判会直接拒掉好数据。

    字段：`locale` 语言（要按语言认「调用意图」那一段的段落名）。
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
        lines = render.intent_lines(state, loc)
        lowered = state.lower()
        leftovers = sorted(
            {line.strip() for line in state.splitlines() if line.strip() in LEFTOVER_LINES}
        )
        tags = [tag for tag in LEFTOVER_TAGS if tag in lowered]
        problems: list[str] = []
        if not lines or not lines[0].strip():
            problems.append("「调用意图」那一行是空的")
        else:
            if len(lines) > 1:
                problems.append("「调用意图」不止一行")
            if render.label_leak(lines[0], loc):
                problems.append("「调用意图」开头又写了一遍段落名")
        if leftovers or tags or problems:
            return Verdict(
                False,
                reason="state 里有格式残留：" + "、".join([*leftovers, *tags, *problems]),
                raw={"leftovers": leftovers, "tags": tags, "problems": problems},
            )
        return Verdict(True, reason="意图单行、无围栏、无残留")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：状态行与策略表必须与代码算出来的一模一样。

    字段：`locale` 语言。预期行为：逐格比对状态行（作用范围 / 策略档位 / 审批状态），
    再逐行比对策略表（规则名 + 处置）；任何一处对不上都判不通过——那是渲染挂错了 tags，
    标签会跟着一起错。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做结构比对。"""
        if not isinstance(spec, ToolGateSpec):
            return Verdict(True, reason="没有 spec，跳过框架比对", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过框架比对", skipped=True)
        loc = locales.load(self.locale)
        expected = render.CallState(
            reach=spec.tags.reach, policy=spec.tags.policy, approval=spec.tags.approval
        )
        problems: list[str] = []
        parsed = render.call_of(state, loc)
        if parsed != expected:
            problems.append(f"状态行与注入特征对不上：解析出 {parsed}，应为 {expected}")
        rules = render.rules_of(state, loc)
        canonical = render.canonical_rules(loc)
        if rules != canonical:
            problems.append(f"策略表与规范表对不上：解析出 {len(rules)} 行，应为 {len(canonical)} 行")
        if not problems:
            return Verdict(True, reason="状态行与策略表都与注入特征一致")
        return Verdict(False, reason="；".join(problems), raw={"call": str(parsed)})


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整段请求交回后端反解两维，必须与注入的一致。

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
        if not isinstance(spec, ToolGateSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        if self.backend is None:
            return Verdict(True, reason="没有后端，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="text",
            domain="tool-call-gating",
            prompt=build_trace_prompt(sample["state"], loc),
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
