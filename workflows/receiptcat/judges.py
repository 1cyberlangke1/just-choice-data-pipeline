"""报销收据归类的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：收据行得**兑现注入特征**——非空、不带围栏、长度落在区间内，
  按字面算出来的命中类别与金额与计划逐条相同，诱饵该有就有、该没有就没有，而且正文里没有
  类别名；整段 state 不许有格式残留；
- `FrameJudge`（零 token）：state 的骨架必须完整且规范——归类表里 C1…C4 一个不少、上限表
  同样一个不少、档位表正好四行、收据行那一块解析得出来。任何一处对不上都说明渲染挂错了
  case、或者语言文件缺了类别，标签会跟着一起错；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端**读一遍**，答「这条收据行归哪一类、
  该走哪一档」，必须与计划算出来的结论一致。这是「渲染必须可反解」的落地：题面上的答案
  就是从 state 里读出来的，所以这段 state 必须真的读得出那个结论。

为什么「归哪一类」既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此重试），
模型那一遍是**读者视角的复核**——机械口径只认字符，正文要是写成「本行包含的归类关键词是
打车」这种自指句子，机械口径照样通过，而读者读出来的东西没有意义，这种样本该丢。两遍问的
不是同一件事。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，也不要一条标签和
文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后还拿不到草稿，只能记成
`skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict, verify_prompt
from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from . import features, locales, questions, render
from .spec import ReceiptCatSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown", "```md", "```yaml"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")


class TraceDraft(BaseModel):
    """反解草稿：只回答「这条收据行归哪一类、该走哪一档」。

    `category` 填**归类表里那个类别名**（原样抄），`tier` 填档位表里的**下标**（从 0 开始）。
    对不上名字或下标越界都算「没判成」（`skipped=True`）而不是「不通过」——那是反解格式没
    对齐，不是数据本身错。
    """

    category: str = Field(description="这条收据行归到归类表里的哪一类（原样抄类别名）")
    tier: int = Field(description="按档位表该走哪一档，填档位下标（从 0 开始）")


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词（口径见 `jc.judge.verify_prompt`）。"""
    return verify_prompt(state, loc)


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：收据行必须兑现注入特征。字段：`locale` 语言。"""

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：没有 spec 就判不了（`skipped=True`）；有 spec 就按 `render.check_draft`
        逐条复核，并顺带查一遍格式残留。`check_draft` 抛的是 `DraftError`，它带人话理由，
        直接拿来当判词。
        """
        if not isinstance(spec, ReceiptCatSpec):
            return Verdict(True, reason="没有 spec，跳过兑现检查", skipped=True)
        state = sample.get("state")
        parsed = render.parse_state(state)
        if parsed.text is None or not parsed.text.strip():
            return Verdict(False, reason="state 里认不出收据行那一块")
        leftovers = _leftovers(state)
        if leftovers:
            return Verdict(False, reason=f"state 里有格式残留：{leftovers}")
        try:
            render.check_draft(render.ContentDraft(text=parsed.text), spec.case, locales.load(self.locale))
        except DraftError as exc:
            return Verdict(False, reason=f"收据行没兑现注入特征：{exc}")
        return Verdict(True, reason="收据行兑现了注入特征，state 也没有格式残留")


def _leftovers(state: object) -> list[str]:
    """输入：state。输出：格式残留（单独成行的围栏 / 换行标签）。预期行为：没有就给空表。"""
    if not isinstance(state, str):
        return ["state 不是字符串"]
    found: list[str] = []
    for raw in state.splitlines():
        text = raw.strip()
        if text in LEFTOVER_LINES:
            found.append(text)
        found.extend(tag for tag in LEFTOVER_TAGS if tag in text)
    return found


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：state 的骨架必须完整且规范。字段：`locale` 语言。"""

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格（不用）。输出：Verdict。

        预期行为：三张表与收据行一处不缺才算通过。归类表 / 上限表都按**类别编号**比——
        编号是结构键，与语言无关，所以换语言不会让这条判据翻车。
        """
        del spec
        parsed = render.parse_state(sample.get("state"))
        problems: list[str] = []
        if parsed.categories != features.CATEGORY_IDS:
            problems.append(
                f"归类表列出的类别是 {list(parsed.categories)}，应为 {list(features.CATEGORY_IDS)}"
            )
        if parsed.caps != features.CATEGORY_IDS:
            problems.append(
                f"上限表列出的类别是 {list(parsed.caps)}，应为 {list(features.CATEGORY_IDS)}"
            )
        if parsed.tiers != len(features.TIERS):
            problems.append(f"档位表有 {parsed.tiers} 行，应为 {len(features.TIERS)} 行")
        if parsed.text is None or not parsed.text.strip():
            problems.append("收据行那一块解析不出来或为空")
        if not problems:
            return Verdict(True, reason="三张表与收据行的骨架都与规范形态一致")
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

        预期行为：不一致时把「计划的」与「反解出来的」列在理由里，方便人看是哪里写歪了。
        端点故障、草稿缺字段与不认识的名字 / 越界的下标都只算「没判成」（`skipped=True`）
        ——记成「不通过」会把端点抖动或格式漂移写成数据质量问题。

        期望的那一份文案按 **state 自己的语言**取（`render.detect_locale`）：一次 run 里混了
        zh / en 时，拿调用方给的那一种去比会平白把对的判成错的（范式 §9.6x 那一类坑）。
        """
        if not isinstance(spec, ReceiptCatSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        state = str(sample["state"])
        loc = locales.load(render.detect_locale(state) or self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="choice",
            domain="receiptcat",
            prompt=build_trace_prompt(state, loc),
            response_model=TraceDraft,
            seeds=(),
            spec={
                "case": spec.case.as_dict(),
                "amount": spec.case.amount,
                "category": spec.case.category,
                "band": spec.case.band,
                "decoy": spec.case.decoy,
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
        names = render.category_names(loc)
        seen = draft.data.get("category")
        tier = draft.data.get("tier")
        if not isinstance(seen, str) or not isinstance(tier, int) or isinstance(tier, bool):
            return Verdict(
                True,
                reason=f"反解草稿的字段形态不对，判不了：{draft.data!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        if seen not in set(names.values()):
            return Verdict(
                True,
                reason=f"反解草稿里的类别名 {seen!r} 不在归类表里，判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        if not 0 <= tier < len(features.TIERS):
            return Verdict(
                True,
                reason=f"反解草稿里的档位下标 {tier} 越界，判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        want_name = names[labels.category]
        want_tier = questions.tier_index(labels.tier)
        if seen != want_name or tier != want_tier:
            return Verdict(
                False,
                reason=(
                    f"计划是「{want_name}」+ 档位 {want_tier}，"
                    f"反解读成「{seen}」+ 档位 {tier}"
                ),
                raw={"traced_category": seen, "traced_tier": tier},
            )
        return Verdict(
            True,
            reason=f"反解与计划一致：「{want_name}」+ 档位 {want_tier}",
            raw={"traced_category": seen, "traced_tier": tier},
        )
