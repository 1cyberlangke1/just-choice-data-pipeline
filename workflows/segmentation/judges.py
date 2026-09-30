"""文档包切分的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：这一包得**兑现注入特征**——页数对得上、每一页非空、不带围栏、
  单页长度落在区间内、**每一页按字面算出来的种类与计划逐页相同**、每一页都带来源痕迹，
  而且**没有把判据原文整段贴进去**；整段 state 不许有格式残留；
- `FrameJudge`（零 token）：state 的骨架必须完整且规范——种类表里四个编号一个不少、顺序与
  `features.CLASS_IDS` 一致，页序那一块解析得出来且页数对得上。任何一处对不上都说明渲染挂错了
  case、或者语言文件缺了种类，标签会跟着一起错；
- `ReadJudge`（一次后端调用）：把整段 state 交回后端**读一遍**，答「这一包里出现了哪几类」，
  必须与计划算出来的结论一致。这是「渲染必须可反解」的落地：题面上的答案就是从 state 里读出来
  的，所以这段 state 必须真的读得出那个结论。

为什么「出现了哪几类」既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此重试），
模型那一遍是**读者视角的复核**——机械口径只认字符，正文要是写成「本页属于收据类」这种自指句子，
机械口径照样通过，而读者读出来的东西没有意义，这种样本该丢。两遍问的不是同一件事。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，也不要一条标签和
文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后还拿不到草稿，只能记成
`skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.judge import squash as _squash
from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict
from jc.jsontypes import JsonValue

from . import features, locales, render
from .spec import SegmentationSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown", "```md", "```yaml"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")


class ClassHit(BaseModel):
    """反解草稿里的一条命中：种类编号 + 让它命中的那段原文。

    为什么要连原文一起要：模型报「出现了收据类」却抄不出那段话，是**误报**，不是数据错——
    页面里到底有没有那个抬头是机械可查的（`features.PATTERNS`），所以让它把证据抄出来，
    代码再拿正则复核一遍，抄不出来的那一条直接不算。
    """

    class_id: str = Field(description="命中的种类编号（form / letter / receipt / contract）")
    quote: str = Field(description="从某一页里**原样抄出来**的那一段——就是它满足了这一类的判据")


class ReadDraft(BaseModel):
    """反解草稿：只回答「这一包里出现了哪几类」。

    取值域由 `features.CLASS_IDS` 定，认不出来的编号一律算「没判成」（`skipped=True`）而不是
    「不通过」——那是反解格式没对齐，不是数据本身错。
    """

    classes: list[ClassHit] = Field(
        description="这一包里出现过的种类（每条带编号与引文）；一类都没有就给空数组"
    )


def build_read_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。输出：提示词。
    预期行为：把整段 state（含种类表、切分规则与逐页正文）原样交给后端，只问「出现了哪几类」——
    反解用的口径与渲染期用的机械口径是同一份（判据连同原文都印在 state 里），否则「答对了」与
    「标签对了」不是一回事。
    """
    return "\n".join(
        [
            str(loc["verify"]["intro"]),
            "",
            str(loc["verify"]["state_header"]),
            state,
            "",
            str(loc["verify"]["q"]),
        ]
    )


def quote_holds(cls: str, quote: str, body: str) -> bool:
    """输入：种类编号、反解抄出来的引文、全部页面正文。输出：这条命中站不站得住。

    预期行为：两件事同时成立才算数——引文**原样出现在正文里**（空白折叠后比对，模型重排空白
    不算抄错），且引文本身满足该种类的机械判据（`features.PATTERNS`）。抄不出原文、或抄出来
    的那段按正则根本不算命中，一律不算数。
    """
    if not quote or _squash(quote) not in _squash(body):
        return False
    pattern = features.PATTERNS.get(cls)
    return bool(pattern and pattern.search(quote))


def _dropped_note(dropped: list[str]) -> str:
    """输入：被丢掉的误报编号。输出：附在理由后面的一句说明（没有误报就是空串）。"""
    if not dropped:
        return ""
    return f"（另有 {len(dropped)} 条报出来的种类抄不出原文或引文不合判据，已按没出现处理：{dropped}）"


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：这一包必须兑现注入特征。

    字段：`locale` 语言（判据文案按它取，报错信息才读得懂）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：没有规格或 state 不是字符串时记「没判成」；有规格就按字面重算逐页种类，与计划
        逐页比。规格类型不对（串了别的 workflow）也算没判成。
        """
        locales.load(self.locale)
        if not isinstance(spec, SegmentationSpec):
            return Verdict(True, reason="没有 spec，跳过兑现检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过兑现检查", skipped=True)
        parsed = render.parse_state(state)
        if not parsed.pages:
            return Verdict(False, reason="页序那一块解析不出来或没有页")
        if len(parsed.pages) != len(spec.plan.shape):
            return Verdict(
                False,
                reason=f"页数是 {len(parsed.pages)}，计划要的是 {len(spec.plan.shape)}",
            )
        found = features.observed_labels(parsed.pages)
        if found.shape != spec.plan.shape:
            return Verdict(
                False,
                reason=(
                    f"逐页按字面算出来的种类是 {list(found.shape)}，"
                    f"计划要的是 {list(spec.plan.shape)}"
                ),
            )
        for index, text in enumerate(parsed.pages, start=1):
            if not features.source_trace(text, spec.case.source):
                return Verdict(False, reason=f"第 {index} 页没有带来源痕迹")
        lang = render.detect_locale(state) or self.locale
        pasted = render.pasted_rules("\n".join(parsed.pages), locales.load(lang))
        if pasted:
            return Verdict(False, reason=f"正文把判据原文原样贴了进去：{pasted}")
        leftovers = [
            line for line in (raw.strip() for raw in state.splitlines()) if line in LEFTOVER_LINES
        ]
        if leftovers:
            return Verdict(False, reason=f"state 里有格式残留：{sorted(set(leftovers))}")
        if any(tag in state for tag in LEFTOVER_TAGS):
            return Verdict(False, reason="state 里混进了 HTML 换行标签")
        return Verdict(True, reason="这一包兑现了计划，且没有格式残留")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：state 的骨架必须完整且规范。

    字段：`locale` 语言。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：种类表里的编号必须是 `CLASS_IDS` 且顺序一致、页序那一块必须解析得出来且页数
        与计划一致。
        """
        locales.load(self.locale)
        if not isinstance(spec, SegmentationSpec):
            return Verdict(True, reason="没有 spec，跳过骨架检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过骨架检查", skipped=True)
        parsed = render.parse_state(state)
        problems: list[str] = []
        if parsed.classes != features.CLASS_IDS:
            problems.append(
                f"种类表列出的编号是 {list(parsed.classes)}，应为 {list(features.CLASS_IDS)}"
            )
        if not parsed.pages:
            problems.append("页序那一块解析不出来或没有页")
        elif len(parsed.pages) != len(spec.plan.shape):
            problems.append(
                f"页数是 {len(parsed.pages)}，应为 {len(spec.plan.shape)}"
            )
        if not problems:
            return Verdict(True, reason="种类表与页序骨架都与规范形态一致")
        return Verdict(False, reason="；".join(problems))


@dataclass(frozen=True)
class ReadJudge:
    """模型判别器：把整段 state 交回后端读一遍，读出来的结论必须与计划一致。

    字段：`backend` 生成后端（由 `bind_backend` 注入）；`locale` 语言。
    """

    backend: GenerationBackend
    locale: str
    name: str = "read"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：不一致时把「计划的」与「反解出来的」列在理由里，方便人看是哪里写歪了。端点故障、
        草稿缺字段与不认识的编号都只算「没判成」（`skipped=True`）——记成「不通过」会把端点抖动或
        格式漂移写成数据质量问题。
        """
        if not isinstance(spec, SegmentationSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.read_task_id,
            kind="choice",
            domain="segmentation",
            prompt=build_read_prompt(str(sample["state"]), loc),
            response_model=ReadDraft,
            seeds=(),
            spec={
                "case": spec.case.as_dict(),
                "classes": list(spec.plan.classes),
                "source": spec.case.source,
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
        seen = draft.data.get("classes")
        if not isinstance(seen, list):
            return Verdict(
                True,
                reason=f"反解草稿缺 classes 字段，判不了：{seen!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        pairs: list[tuple[str, str]] = []
        for item in seen:
            if not isinstance(item, dict):
                return Verdict(
                    True,
                    reason=f"反解草稿的 classes 项不是「编号 + 引文」对象，判不了：{item!r}",
                    raw={"raw_text": draft.raw_text[:500]},
                    skipped=True,
                )
            pairs.append((str(item.get("class_id", "")), str(item.get("quote", ""))))
        unknown = [cls for cls, _ in pairs if cls not in features.CLASS_IDS]
        if unknown:
            return Verdict(
                True,
                reason=f"反解草稿里出现不认识的种类编号 {unknown}，判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        body = "\n".join(render.parse_state(str(sample["state"])).pages)
        reported: list[str] = []
        dropped: list[str] = []
        for cls, quote in pairs:
            (reported if quote_holds(cls, quote, body) else dropped).append(cls)
        if set(reported) != set(labels.classes):
            return Verdict(
                False,
                reason=(
                    f"计划里出现 {list(labels.classes) or '无'}，"
                    f"反解读成 {reported or '无'}{_dropped_note(dropped)}"
                ),
                raw={"read_classes": reported},
            )
        return Verdict(
            True,
            reason=f"反解与计划一致：出现 {list(labels.classes) or '无'}{_dropped_note(dropped)}",
            raw={"read_classes": reported},
        )
