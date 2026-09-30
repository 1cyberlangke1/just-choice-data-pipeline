"""打分与评分的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：待评材料必须存在，整段 state 不许有格式残留，
  材料形状必须与注入的五维特征逐条对得上（句数 / 分条 / 数字 / 留空占位 / 出处标记）；
- `FrameJudge`（零 token）：材料清单那一行必须逐格等于注入特征，三张表必须逐行等于规范表
  ——渲染串味（挂错 tags、漏行、错格）全在这一关拦下；
- `TraceJudge`（一次后端调用）：把整段材料交回后端**读一遍**，反解「有没有留空占位」与
  「有没有出处标记」两维，两个答案都必须与注入特征一致。这是「渲染必须可反解」的落地
  （PLAN 3.4 第 ③ 档）：评分单上那两行观测不印数字，只能从材料里数出来，所以这两维必须
  真的能从材料里读回来，否则这道题没人答得对。

为什么正文两维既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此重试），
模型那一遍是**读者视角的复核**——材料要是把 `TODO` 写在引用的示例里、读起来像「没写完」，
机械口径照样通过，而读者会读岔，这种样本该丢。两遍问的不是同一件事。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，也不要一条
标签和文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后还拿不到草稿，
只能记成 `skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict, leftover_in
from jc.jsontypes import JsonValue

from . import locales, render
from .spec import RubricSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签（围栏在材料里由 `render.check_draft` 与
#: 卫生判别器各查一次；这里再按「单独成行」查一遍整段 state——模型偶尔会把整段回答
#: 包进 ``` 里，那样的行会落在材料段之外）。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown", "```md", "```yaml"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")

#: 反解时重问的原子维度（维度名与 `features.DIMS` 完全一致，选项用同一份 locale）。
TRACE_FAMILIES: Final[tuple[str, ...]] = ("placeholders", "citations")


class TraceDraft(BaseModel):
    """反解草稿：反解「有没有留空占位」与「有没有出处标记」两维。

    取值域必须与 `features.PLACEHOLDERS` / `features.CITATIONS` 以及 locale 里
    `criteria` 的 key 一致；这条由 `tests/workflows/rubric/test_locales.py` 锁住，不靠人记。
    """

    placeholders: Literal["no", "yes"]
    citations: Literal["no", "yes"]


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。
    输出：提示词。预期行为：把两维的原子题连同选项原样再问一遍——反解用的口径与渲染期
    用的机械口径是同一份（都写在 locale 的 `trace` 里），否则「答对了」与「标签对了」
    不是一回事。
    """
    lines = [
        str(loc["verify"]["intro"]),
        "",
        str(loc["verify"]["state_header"]),
        state,
        "",
        str(loc["verify"]["questions_header"]),
    ]
    for index, dim in enumerate(TRACE_FAMILIES, start=1):
        lines.append(f"{index}. {loc['trace'][dim]['q']}")
        for name, desc in loc["trace"][dim]["criteria"].items():
            lines.append(str(loc["verify"]["option_line"]).format(name=name, desc=desc))
    return "\n".join(lines)


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：材料得有内容、形状得与五维特征对得上、整段 state 不许有残留。

    查三类：**单独成行的围栏 / HTML 换行标签**、**待评材料是空的**、**材料形状与特征对不上**
    （口径直接复用 `render.check_draft`：渲染期查一遍是为了重试，这里再查一遍是为了兜住
    **不是本 workflow 渲染出来的** state——判别器可能被单独拿去跑产物）。

    口径压到最窄——宁可漏检也不误判，因为误判会直接拒掉好数据。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只看形状与残留。"""
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(False, reason="state 不是文本")
        loc = locales.load(self.locale)
        problems: list[str] = []
        leftover = leftover_in(state, LEFTOVER_LINES, LEFTOVER_TAGS)
        if leftover:
            problems.append(f"state 里有格式残留：{leftover!r}")
        parsed = render.parse_state(state)
        if parsed.is_empty:
            problems.append("待评材料那一段是空的")
        if isinstance(spec, RubricSpec):
            try:
                render.check_draft({"body": parsed.body}, spec.tags, loc)
            except ValueError as exc:
                problems.append(str(exc))
        if not problems:
            return Verdict(True, reason="材料形状与注入特征一致，state 里没有残留")
        return Verdict(False, reason="；".join(problems))


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：清单行与三张表必须逐格等于规范形态。

    查四处：**材料清单**（篇幅 / 形式 / 数据）、**评分单**（五行，含观测那一格该印数字
    还是该印「按材料数」）、**档位表**、**处置表**；任何一处对不上都判不通过——那是渲染
    挂错了 tags，标签会跟着一起错。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做结构比对。"""
        if not isinstance(spec, RubricSpec):
            return Verdict(True, reason="没有 spec，跳过框架比对", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过框架比对", skipped=True)
        loc = locales.load(self.locale)
        tags = spec.tags
        problems: list[str] = []
        expected = render.ManifestState(length=tags.length, form=tags.form, data=tags.data)
        parsed_manifest = render.manifest_of(state, loc)
        if parsed_manifest != expected:
            problems.append(f"材料清单行与注入特征对不上：解析出 {parsed_manifest}，应为 {expected}")
        checks = (
            ("评分单", render.score_of(state, loc), render.score_rows(tags, loc)),
            ("档位表", render.band_of(state, loc), render.band_rows()),
            ("处置表", render.gate_of(state, loc), render.gate_rows()),
        )
        for label, parsed, canonical in checks:
            if parsed != canonical:
                problems.append(
                    f"{label}与规范表对不上：解析出 {len(parsed)} 行，应为 {len(canonical)} 行"
                )
        if not problems:
            return Verdict(True, reason="材料清单行与三张表都与注入特征一致")
        return Verdict(False, reason="；".join(problems), raw={"manifest": str(parsed_manifest)})


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整段材料交回后端读一遍反解两维，必须与注入的一致。

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
        if not isinstance(spec, RubricSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        if self.backend is None:
            return Verdict(True, reason="没有后端，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="text",
            domain="grading",
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
            f"{dim}: 注入 {getattr(labels, dim)!r}，反解成 {data[dim]!r}"
            for dim in TRACE_FAMILIES
            if data[dim] != getattr(labels, dim)
        ]
        return Verdict(
            not problems,
            reason="; ".join(problems),
            raw={"traced": {dim: data[dim] for dim in TRACE_FAMILIES}},
        )
