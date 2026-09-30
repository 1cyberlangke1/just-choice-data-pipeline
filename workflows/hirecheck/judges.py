"""招聘初筛门禁的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：候选人材料得**兑现注入特征**——非空、不带围栏、长度落在区间内、六个
  结构键都在、`note:` 那一行不是空的、`note:` 那一行**没有写进初筛结论**（范式 §9.59：写了就把结构
  键上的标签搬进了自由文字）、`note:` 那一行**没有提到岗位的硬性技能**（`H2` 考的就是技能表里有没有
  它，备注里再提一次材料就自相矛盾），按字面算出来的命中组合与计划逐条相同，而且**没有把判据原文
  整段贴进去**；整段 state 不许有格式残留（围栏行）；
- `FrameJudge`（零 token）：state 的骨架必须完整且规范——判据表里 H1…H4 一个不少、顺序与
  `features.TAG_IDS` 一致，定级规则两行与处置档位表五行的**原文**与代码渲染的规范行逐字相同。任何
  一处对不上都说明渲染挂错了 case、或者语言文件缺了等级，标签会跟着一起错；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端**读一遍**，答「这份材料命中了哪几条判据」与
  「定几级」，必须与计划算出来的结论一致。这是「渲染必须可反解」的落地：题面上的答案就是从 state 里
  读出来的（等级还要按权重公式算一遍），所以这段 state 必须真的读得出那两个结论。

为什么「命中了哪几条判据」既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此重试），
模型那一遍是**读者视角的复核**——机械口径只认字符，材料要是写成「本次命中的判据是：……」这种自指
句子，机械口径照样通过，而读者读出来的东西没有意义，这种样本该丢。两遍问的不是同一件事。

**等级那一问为什么必须有**：`grade` 是这一族的中心题型，而它考的是「拿权重算一遍」这一步。反解只查
判据的话，权重印错（少一项、公式与权重表打架）在单测里看不出来——真机产物会显示成「模型答不对定
级」，其实是材料本身读不出答案。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，也不要一条标签和文本对不
上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后还拿不到草稿，只能记成 `skipped=True`
（没判成），否则端点抖动会被写成合成质量问题。
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
from .spec import HireCheckSpec

#: 格式残留：单独成行的围栏。候选人材料里不会出现反引号围栏，所以这一份清单是安全的。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown", "```md", "```yaml", "```log"}
)


class TraceFact(BaseModel):
    """反解草稿里的一条命中：判据编号 + 让它命中的那段原文。

    为什么要连原文一起要：模型报「命中 H2」却抄不出那段话，是**误报**，不是数据错——材料里到底有没有
    那个形状是机械可查的（`features.PATTERNS`），所以让它把证据抄出来，代码再拿正则复核一遍，抄不出
    来的那条直接不算。
    """

    tag: str = Field(description="命中的判据编号（H1 / H2 / H3 / H4）")
    quote: str = Field(description="从候选人材料里**原样抄出来**的那一段——就是它满足了这条判据")


class TraceDraft(BaseModel):
    """反解草稿：这份材料命中了哪几条判据，以及按定级规则算几级。

    取值域由 `features.TAG_IDS` 与 `features.LEVELS` 定，认不出来的编号一律算「没判成」
    （`skipped=True`）而不是「不通过」——那是反解格式没对齐，不是数据本身错。
    """

    tags: list[TraceFact] = Field(description="命中的判据（每条带编号与引文）；一条都没有就给空数组")
    level: int = Field(description="按定级规则算出来的初筛等级（1~5 的整数）")


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。输出：提示词。
    预期行为：把整段 state（含判据表、定级规则、档位表与一屏候选人材料）原样交给后端，只问「命中了
    哪几条判据、定几级」——反解用的口径与渲染期用的机械口径是同一份（判据连同原文都印在 state 里），
    否则「答对了」与「标签对了」不是一回事。
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


def quote_holds(tag: str, quote: str, body: str) -> bool:
    """输入：判据编号、反解抄出来的引文、一屏候选人材料。输出：这条命中站不站得住。

    预期行为：三件事同时成立才算数——引文**原样**在材料里（空白差异不算抄错）、引文**自己**满足那条
    判据、**整屏材料**也满足那条判据。第三条是范式 §9.47 的落地：判据是**整段文本**的性质，引文只是
    「你确实读到过这里」的凭据；只把形状塞进引文而整段不算命中，是误报。
    """
    if tag not in features.TAG_IDS:
        return False
    if _squash(quote) not in _squash(body):
        return False
    pattern = features.PATTERNS.get(tag)
    return bool(pattern and pattern.search(quote) and pattern.search(body))


def _dropped_note(dropped: list[str]) -> str:
    """输入：被丢掉的误报编号。输出：附在理由后面的一句说明（没有误报就是空串）。"""
    if not dropped:
        return ""
    return (
        f"（另有 {len(dropped)} 条报出来的判据抄不出原文、引文不合判据，"
        f"或按整屏材料根本不算命中，已按没命中处理：{dropped}）"
    )


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：候选人材料必须兑现注入特征。

    字段：`locale` 语言（判据文案按它取，报错信息才读得懂）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：没有规格或 state 不是字符串时记「没判成」；有规格就按字面重算命中组合，与计划逐条
        比。规格类型不对（串了别的 workflow）也算没判成。
        """
        locales.load(self.locale)
        if not isinstance(spec, HireCheckSpec):
            return Verdict(True, reason="没有 spec，跳过兑现检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过兑现检查", skipped=True)
        parsed = render.parse_state(state)
        if parsed.text is None or not parsed.text.strip():
            return Verdict(False, reason="候选人材料那一块解析不出来或为空")
        lang = render.detect_locale(state) or self.locale
        loc = locales.load(lang)
        missing = features.missing_keys(parsed.text)
        if missing:
            return Verdict(False, reason=f"候选人材料缺这几行：{list(missing)}")
        note = features.line_value(parsed.text, render.PROSE_KEY)
        if not note:
            return Verdict(False, reason=f"候选人材料的 `{render.PROSE_KEY}:` 那一行是空的")
        verdicts = render.note_verdicts(note, loc)
        if verdicts:
            return Verdict(
                False,
                reason=(
                    f"候选人材料的 `{render.PROSE_KEY}:` 那一行写进了初筛结论（{verdicts}）——"
                    "它只讲这个人在做什么，不许替他下结论"
                ),
            )
        if features.note_leaks_skill(note):
            return Verdict(
                False,
                reason=(
                    f"候选人材料的 `{render.PROSE_KEY}:` 那一行提到了岗位的硬性技能"
                    f"（{features.REQUIRED_SKILL}）——技能只写在 `skills:` 那一行上"
                ),
            )
        found = features.hits_of(parsed.text)
        if found != spec.case.flags:
            return Verdict(
                False,
                reason=(
                    f"候选人材料按字面算出来的命中组合是 {found or '无'}，"
                    f"计划要的是 {spec.case.flags or '无'}"
                ),
            )
        pasted = render.pasted_rules(parsed.text, loc)
        if pasted:
            return Verdict(False, reason=f"候选人材料把判据原文原样贴了进去：{pasted}")
        leftovers = [
            line for line in (raw.strip() for raw in state.splitlines()) if line in LEFTOVER_LINES
        ]
        if leftovers:
            return Verdict(False, reason=f"state 里有格式残留：{sorted(set(leftovers))}")
        return Verdict(True, reason="候选人材料兑现了计划，且没有格式残留")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：state 的骨架必须完整且规范。

    字段：`locale` 语言。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：判据表里的编号必须是 `H1…H4` 且顺序一致；定级规则必须**逐字**等于代码渲染的两行
        （公式或权重少一项 = 考生算不出等级）；档位表必须逐字等于五行的规范形态；候选人材料那一块必须
        解析得出来、六个结构键齐全、`note:` 那一行非空。
        """
        locales.load(self.locale)
        if not isinstance(spec, HireCheckSpec):
            return Verdict(True, reason="没有 spec，跳过骨架检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过骨架检查", skipped=True)
        parsed = render.parse_state(state)
        lang = render.detect_locale(state) or self.locale
        loc = locales.load(lang)
        problems: list[str] = []
        if parsed.facts != features.TAG_IDS:
            problems.append(
                f"判据表列出的编号是 {list(parsed.facts)}，应为 {list(features.TAG_IDS)}"
            )
        expected_rule = tuple(render.rule_lines(loc))
        if parsed.rule != expected_rule:
            problems.append(
                f"定级规则与规范行不一致（应为 {len(expected_rule)} 行，实际 {len(parsed.rule)} 行）"
            )
        expected_actions = tuple(render.action_lines(loc))
        if parsed.actions != expected_actions:
            problems.append(
                f"处置档位表与规范行不一致（应为 {len(expected_actions)} 行，"
                f"实际 {len(parsed.actions)} 行）"
            )
        if parsed.text is None or not parsed.text.strip():
            problems.append("候选人材料那一块解析不出来或为空")
        else:
            missing = features.missing_keys(parsed.text)
            if missing:
                problems.append(f"候选人材料缺这几行：{list(missing)}")
            elif not features.line_value(parsed.text, render.PROSE_KEY):
                problems.append(f"候选人材料的 `{render.PROSE_KEY}:` 那一行是空的")
        if not problems:
            return Verdict(True, reason="判据表、定级规则、档位表与候选人材料骨架都与规范形态一致")
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

        预期行为：判据与等级**两样都要对**——不一致时把「计划的」与「反解出来的」列在理由里。端点故障、
        草稿缺字段、不认识的编号与越界的等级都只算「没判成」（`skipped=True`）——记成「不通过」会把端点
        抖动或格式漂移写成数据质量问题。
        """
        if not isinstance(spec, HireCheckSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="choice",
            domain="hirecheck",
            prompt=build_trace_prompt(str(sample["state"]), loc),
            response_model=TraceDraft,
            seeds=(),
            spec={
                "case": spec.case.as_dict(),
                "hits": list(spec.case.tags()),
                "channel": spec.case.channel,
                "level": spec.plan.level,
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
        level = draft.data.get("level")
        if not isinstance(seen, list):
            return Verdict(
                True,
                reason=f"反解草稿缺 tags 字段，判不了：{seen!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 5:
            return Verdict(
                True,
                reason=f"反解草稿里的等级不是 1~5 的整数，判不了：{level!r}",
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
        if level != labels.level:
            return Verdict(
                False,
                reason=(
                    f"计划里的等级是 {labels.level}，反解读成 {level}"
                    f"（命中组合一致，说明读者没把权重与公式算对）"
                ),
                raw={"traced_hits": reported, "traced_level": level},
            )
        return Verdict(
            True,
            reason=(
                f"反解与计划一致：命中 {list(labels.hits) or '无'}，等级 {level}"
                f"{_dropped_note(dropped)}"
            ),
            raw={"traced_hits": reported, "traced_level": level},
        )
