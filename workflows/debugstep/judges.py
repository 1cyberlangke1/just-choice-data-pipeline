"""调试器下一步动作的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：会话得**兑现注入特征**——非空、不带围栏、长度落在区间内、五个结构
  键都在，按字面算出来的命中组合与计划逐条相同，而且**没有把判据原文整段贴进去**；整段 state
  不许有格式残留（围栏行）；
- `FrameJudge`（零 token）：state 的骨架必须完整且规范——判据表里 D1…D4 一个不少、顺序与
  `features.TAG_IDS` 一致，会话那一块解析得出来且五行结构键齐全。任何一处对不上都说明渲染挂错了
  case、或者语言文件缺了判据，标签会跟着一起错；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端**读一遍**，答「这屏会话踩了哪几条」，必须与
  计划算出来的结论一致。这是「渲染必须可反解」的落地：题面上的答案就是从 state 里读出来的，
  所以这段 state 必须真的读得出那个结论。

为什么「踩了哪几条」既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此重试），模型
那一遍是**读者视角的复核**——机械口径只认字符，会话要是写成「本会话踩的判据是：……」这种自指
句子，机械口径照样通过，而读者读出来的东西没有意义，这种样本该丢。两遍问的不是同一件事。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，也不要一条标签和文本
对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后还拿不到草稿，只能记成
`skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.judge import quote_hit as _quote_hit
from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict, verify_prompt
from jc.jsontypes import JsonValue

from . import features, locales, render
from .spec import DebugStepSpec

#: 格式残留：单独成行的围栏。会话里不会出现反引号围栏，所以这一份清单是安全的。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown", "```md", "```yaml", "```python"}
)


class TraceHit(BaseModel):
    """反解草稿里的一条命中：判据编号 + 让它命中的那段原文。

    为什么要连原文一起要：模型报「踩了 D2」却抄不出那段话，是**误报**，不是数据错——会话里到底
    有没有那个形状是机械可查的（`features.PATTERNS`），所以让它把证据抄出来，代码再拿正则复核
    一遍，抄不出来的那条直接不算。
    """

    tag: str = Field(description="命中的判据编号（D1 / D2 / D3 / D4）")
    quote: str = Field(description="从会话里**原样抄出来**的那一段——就是它满足了这条判据")


class TraceDraft(BaseModel):
    """反解草稿：只回答「这屏会话踩了哪几条判据」。

    取值域由 `features.TAG_IDS` 定，认不出来的编号一律算「没判成」（`skipped=True`）而不是
    「不通过」——那是反解格式没对齐，不是数据本身错。
    """

    tags: list[TraceHit] = Field(description="命中的判据（每条带编号与引文）；一条都没踩就给空数组")


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词（口径见 `jc.judge.verify_prompt`）。"""
    return verify_prompt(state, loc)


def quote_holds(tag: str, quote: str, body: str) -> bool:
    """输入：判据编号、反解抄出来的引文、一屏会话。输出：这条命中站不站得住。

    预期行为：三件事同时成立才算数——引文**原样在会话里**（空白差异不算抄错）、引文本身满足该判据
    的机械判据、**整屏会话也满足那条判据**（`features.PATTERNS`）。抄不出原文、抄出来的那段按
    正则不算命中、或那条判据按整屏会话根本不算命中，一律不算数。

    为什么第三条不能省（范式 §9.47，`turntake` 上踩过）：判据是**整段文本**的性质，引文只是
    「你确实读到过这里」的凭据。只看引文的话，反解抄一个孤立的关键词就能「证明」命中，而整屏会话
    其实根本不算——那样会把一条本来合格的样本误拒。
    """
    hit = _quote_hit(quote, body)
    if not hit:
        return False
    pattern = features.PATTERNS.get(tag)
    return bool(pattern and pattern.search(hit) and pattern.search(body))


def _dropped_note(dropped: list[str]) -> str:
    """输入：被丢掉的误报编号。输出：附在理由后面的一句说明（没有误报就是空串）。"""
    if not dropped:
        return ""
    return (
        f"（另有 {len(dropped)} 条报出来的判据抄不出原文、引文不合判据，"
        f"或按整屏会话根本不算命中，已按没命中处理：{dropped}）"
    )


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：会话必须兑现注入特征。

    字段：`locale` 语言（判据文案按它取，报错信息才读得懂）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：没有规格或 state 不是字符串时记「没判成」；有规格就按字面重算命中组合，与计划
        逐条比。规格类型不对（串了别的 workflow）也算没判成。
        """
        locales.load(self.locale)
        if not isinstance(spec, DebugStepSpec):
            return Verdict(True, reason="没有 spec，跳过兑现检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过兑现检查", skipped=True)
        parsed = render.parse_state(state)
        if parsed.text is None or not parsed.text.strip():
            return Verdict(False, reason="会话那一块解析不出来或为空")
        missing = features.missing_keys(parsed.text)
        if missing:
            return Verdict(False, reason=f"会话缺这几行：{list(missing)}")
        found = features.hits_of(parsed.text)
        if found != spec.case.hits:
            return Verdict(
                False,
                reason=(
                    f"会话按字面算出来的命中组合是 {found or '无'}，"
                    f"计划要的是 {spec.case.hits or '无'}"
                ),
            )
        lang = render.detect_locale(state) or self.locale
        pasted = render.pasted_rules(parsed.text, locales.load(lang))
        if pasted:
            return Verdict(False, reason=f"会话把判据原文原样贴了进去：{pasted}")
        leftovers = [
            line
            for line in (raw.strip() for raw in state.splitlines())
            if line in LEFTOVER_LINES
        ]
        if leftovers:
            return Verdict(False, reason=f"state 里有格式残留：{sorted(set(leftovers))}")
        return Verdict(True, reason="会话兑现了计划，且没有格式残留")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：state 的骨架必须完整且规范。

    字段：`locale` 语言。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：判据表里的编号必须是 `D1…D4` 且顺序一致、会话那一块必须解析得出来、五个结构键
        必须齐全。
        """
        locales.load(self.locale)
        if not isinstance(spec, DebugStepSpec):
            return Verdict(True, reason="没有 spec，跳过骨架检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过骨架检查", skipped=True)
        parsed = render.parse_state(state)
        problems: list[str] = []
        if parsed.tags != features.TAG_IDS:
            problems.append(
                f"判据表列出的编号是 {list(parsed.tags)}，应为 {list(features.TAG_IDS)}"
            )
        if parsed.text is None or not parsed.text.strip():
            problems.append("会话那一块解析不出来或为空")
        else:
            missing = features.missing_keys(parsed.text)
            if missing:
                problems.append(f"会话缺这几行：{list(missing)}")
        if not problems:
            return Verdict(True, reason="判据表与会话骨架都与规范形态一致")
        return Verdict(False, reason="；".join(problems))


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整段 state 交回后端读一遍，读出来的结论必须与计划一致。

    字段：`backend` 生成后端（由 `bind_backend` 注入）；`locale` 语言。
    """

    backend: GenerationBackend
    locale: str
    name: str = "trace"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：不一致时把「计划的」与「反解出来的」列在理由里，方便人看是哪里写歪了。端点故障、
        草稿缺字段与不认识的编号都只算「没判成」（`skipped=True`）——记成「不通过」会把端点抖动或
        格式漂移写成数据质量问题。
        """
        if not isinstance(spec, DebugStepSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="choice",
            domain="debugstep",
            prompt=build_trace_prompt(str(sample["state"]), loc),
            response_model=TraceDraft,
            seeds=(),
            spec={
                "case": spec.case.as_dict(),
                "hits": list(spec.case.tags()),
                "runtime": spec.case.runtime,
            },
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
        labels = spec.labels
        seen = draft.data.get("tags")
        if not isinstance(seen, list):
            return Verdict(
                True,
                reason=f"反解草稿缺 tags 字段，判不了：{seen!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        pairs: list[tuple[str, str]] = []
        for item in seen:
            if not isinstance(item, dict):
                return Verdict(
                    True,
                    reason=f"反解草稿的 tags 项不是「编号 + 引文」对象，判不了：{item!r}",
                    raw={"raw_text": draft.raw_text[:500]},
                    skipped=True,
                )
            pairs.append((str(item.get("tag", "")), str(item.get("quote", ""))))
        unknown = [tag for tag, _ in pairs if tag not in features.TAG_IDS]
        if unknown:
            return Verdict(
                True,
                reason=f"反解草稿里出现不认识的判据编号 {unknown}，判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        body = render.parse_state(str(sample["state"])).text or ""
        reported: list[str] = []
        dropped: list[str] = []
        for tag, quote in pairs:
            (reported if quote_holds(tag, quote, body) else dropped).append(tag)
        if set(reported) != set(labels.hits):
            return Verdict(
                False,
                reason=(
                    f"计划里命中 {list(labels.hits) or '无'}，"
                    f"反解读成 {reported or '无'}{_dropped_note(dropped)}"
                ),
                raw={"traced_hits": reported},
            )
        return Verdict(
            True,
            reason=f"反解与计划一致：命中 {list(labels.hits) or '无'}{_dropped_note(dropped)}",
            raw={"traced_hits": reported},
        )
