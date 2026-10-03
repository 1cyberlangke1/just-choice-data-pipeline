"""检索结果判读的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：三段文本必须存在、单行、不含围栏 / HTML 标签 / 段落名残留；
- `FrameJudge`（零 token）：「来源与时效」那一行必须逐格等于注入特征（来源类型 / 时效），
  规则表必须逐行等于规范表——渲染串味（挂错 tags、漏行、错格）全在这一关拦下；
- `TraceJudge`（一次后端调用）：把整段结果页交回后端反解**查询意图 / 是否切题 / 是不是
  垃圾**，三个答案都必须与注入特征一致。这是「渲染必须可反解」的落地（PLAN 3.4 第 ③ 档）：
  三段文本是模型写的，唯一由文本承载的三个标签就是这三维，它们必须能从文本里读回来，
  否则这道题没人答得对。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，
也不要一条标签和文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后
还拿不到草稿，只能记成 `skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict, leftover_in
from jc.jsontypes import JsonValue
from jc import llmjudge

from . import locales, render
from .spec import SearchSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签（围栏在三段文本里由 `render.check_draft` 与
#: 卫生判别器各查一次；这里再按「单独成行」查一遍整段 state——模型偶尔会把整段回答
#: 包进 ``` 里，那样的行会落在文本段之外）。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")

#: 反解时重问的原子题族（题族名与数据集里完全一致，选项也用同一份 locale）。
TRACE_FAMILIES: Final[tuple[str, ...]] = ("query_kind", "topicality", "spam")


class TraceDraft(BaseModel):
    """反解草稿：反解「查询意图 / 是否切题 / 是不是垃圾」三维。

    取值域必须与 `features.QUERY_KINDS` / `TOPICALITIES` / `SPAMS` 以及 locale 里
    `criteria` 的 key 一致；这条由 `tests/workflows/search/test_locales.py` 锁住，不靠人记。
    """

    query_kind: Literal["navigational", "informational", "transactional"]
    topicality: Literal["on_topic", "off_topic"]
    spam: Literal["clean", "spam"]


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """输入：渲染好的 state、语言字典。输出：反解提示词（`TRACE_FAMILIES` 逐族再问一遍）。"""
    return llmjudge.trace_prompt(state, loc, TRACE_FAMILIES)


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：state 里不许有格式残留，三段文本还得有内容、是单行。

    查三类：**单独成行的围栏 / HTML 换行标签**、**三段文本是空的**、**三段文本是多行**。
    口径压到最窄——宁可漏检也不误判，因为误判会直接拒掉好数据。

    字段：`locale` 语言（要按语言认那几段的段落名）。
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
        leftover = leftover_in(state, LEFTOVER_LINES, LEFTOVER_TAGS)
        if leftover:
            return Verdict(False, reason=f"state 里有格式残留：{leftover!r}")
        problems: list[str] = []
        for key, label in (
            ("query_label", "查询"),
            ("title_label", "标题"),
            ("snippet_label", "摘要"),
        ):
            lines = render.field_lines(state, loc, key)
            if not lines:
                problems.append(f"「{label}」是空的")
            elif len(lines) > 1:
                problems.append(f"「{label}」是多行（{len(lines)} 行）")
            else:
                leaked = render.label_leak(lines[0], loc)
                if leaked:
                    problems.append(f"「{label}」里混进了段落名「{leaked}」")
        if not problems:
            return Verdict(True, reason="三段文本都是单行、非空，没有格式残留")
        return Verdict(False, reason="；".join(problems))


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：来源行与规则表必须与代码算出来的一模一样。

    字段：`locale` 语言。预期行为：逐格比对「来源与时效」（来源类型 / 时效），
    再逐行比对规则表（规则名 + 处置）；任何一处对不上都判不通过——那是渲染挂错了 tags，
    标签会跟着一起错。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做结构比对。"""
        if not isinstance(spec, SearchSpec):
            return Verdict(True, reason="没有 spec，跳过框架比对", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过框架比对", skipped=True)
        loc = locales.load(self.locale)
        expected = render.SourceState(
            authority=spec.tags.authority, freshness=spec.tags.freshness
        )
        problems: list[str] = []
        parsed = render.source_of(state, loc)
        if parsed != expected:
            problems.append(f"来源行与注入特征对不上：解析出 {parsed}，应为 {expected}")
        rules = render.rules_of(state, loc)
        canonical = render.canonical_rules(loc)
        if rules != canonical:
            problems.append(f"规则表与规范表对不上：解析出 {len(rules)} 行，应为 {len(canonical)} 行")
        if not problems:
            return Verdict(True, reason="来源行与规则表都与注入特征一致")
        return Verdict(False, reason="；".join(problems), raw={"source": str(parsed)})


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整段结果页交回后端反解三维，必须与注入的一致。

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
        if not isinstance(spec, SearchSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        if self.backend is None:
            return Verdict(True, reason="没有后端，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="text",
            domain="search-result-reading",
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
