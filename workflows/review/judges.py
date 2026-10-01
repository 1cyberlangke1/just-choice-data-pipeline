"""改动审查的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：改动内容必须存在、每行都带 `+` / `-` / 空格前缀、行数与清单里
  的规模对得上，整段 state 不许有格式残留；
- `FrameJudge`（零 token）：清单那一行必须逐格等于注入特征（文件类型 / 变更类型 / 规模），
  三张表必须逐行等于规范表——渲染串味（挂错 tags、漏行、错格）全在这一关拦下；
- `TraceJudge`（一次后端调用）：把整条改动交回后端反解**有没有凭据字面量**与**有没有调试
  输出**，两个答案都必须与注入特征一致。这是「渲染必须可反解」的落地（PLAN 3.4 第 ③ 档）：
  正文是模型写的，唯一由文本承载的两个标签就是这两维，它们必须能从正文里读回来，
  否则这道题没人答得对。

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
from jc import llmjudge

from . import features, locales, render
from .spec import ReviewSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签（围栏在正文里由 `render.check_draft` 与
#: 卫生判别器各查一次；这里再按「单独成行」查一遍整段 state——模型偶尔会把整段回答
#: 包进 ``` 里，那样的行会落在正文段之外）。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown", "```diff", "```python"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")

#: 反解时重问的原子题族（题族名与数据集里完全一致，选项也用同一份 locale）。
TRACE_FAMILIES: Final[tuple[str, ...]] = ("secret", "debug")


class TraceDraft(BaseModel):
    """反解草稿：反解「有没有凭据字面量」与「有没有调试输出」两维。

    取值域必须与 `features.SECRETS` / `features.DEBUGS` 以及 locale 里 `criteria`
    的 key 一致；这条由 `tests/workflows/review/test_locales.py` 锁住，不靠人记。
    """

    secret: Literal["no", "yes"]
    debug: Literal["no", "yes"]


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。
    输出：提示词。预期行为：把 `TRACE_FAMILIES` 里的原子题连同选项再问一遍，选项名走
    `jc.llmjudge.trace_options`——**必须是草稿收得下的那组取值**（这两族是 `yes` / `no`，
    数据集那道 noul 题的 criteria key 却是 `true` / `false`，靠 locale 的
    `verify.trace_options` 覆盖节对齐）。
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
        for name, desc in llmjudge.trace_options(family, loc):
            lines.append(str(loc["verify"]["option_line"]).format(name=name, desc=desc))
    return "\n".join(lines)


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：改动内容得有内容、每行带前缀、行数与清单里的规模对得上。

    查六类：**单独成行的围栏 / HTML 换行标签**、**改动内容是空的**、**有行缺前缀**、
    **行数与规模对不上**、**开头抄了段落名**、**正文里写了表名**。
    口径压到最窄——宁可漏检也不误判，因为误判会直接拒掉好数据。

    后两类与 `render.check_draft` 是同一套判据：渲染期查一遍是为了重试，这里再查一遍是为了
    兜住**不是本 workflow 渲染出来的** state（判别器可能被单独拿去跑产物）。

    注意「每行都带前缀」查的是**成品 state**：`render.normalize_body_lines` 在拼装时已经把
    「缩进在标记前」与「上下文行没写空格前缀」两种草稿拉直了（范式 §9.19），所以这条在
    正常管线里不会触发，专门拦别处来的、没规范化的 state。

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
        leftover = leftover_in(state, LEFTOVER_LINES, LEFTOVER_TAGS)
        if leftover:
            return Verdict(False, reason=f"state 里有格式残留：{leftover!r}")
        lines = render.body_lines(state, loc)
        if not lines:
            return Verdict(False, reason="改动内容是空的")
        leaked = render.label_leak("\n".join(lines), loc)
        if leaked:
            return Verdict(False, reason=f"改动内容把段落名「{leaked}」也写进去了")
        echoed = render.banned_echo("\n".join(lines), loc)
        if echoed:
            return Verdict(False, reason=f"改动内容里写了表名：{'、'.join(echoed)}")
        bad = [line for line in lines if line[:1] not in features.LINE_PREFIXES]
        if bad:
            return Verdict(False, reason=f"改动内容有一行没有 + / - / 空格前缀：{bad[0]!r}")
        manifest = render.manifest_of(state, loc)
        if manifest is not None:
            low, high = features.size_line_bounds(manifest.size)
            if not low <= len(lines) <= high:
                return Verdict(
                    False,
                    reason=f"改动内容有 {len(lines)} 行，与清单里的规模 {manifest.size!r} 要求的 {low}~{high} 行对不上",
                )
        return Verdict(True, reason=f"改动内容 {len(lines)} 行、每行都有前缀")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：清单那一行与三张表都必须与注入特征一致。

    查四处：**清单行**（文件类型 / 变更类型 / 规模）、**审查规则表**、**严重度表**、
    **门控表**；任何一处对不上都判不通过——那是渲染挂错了 tags，标签会跟着一起错。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做结构比对。"""
        if not isinstance(spec, ReviewSpec):
            return Verdict(True, reason="没有 spec，跳过框架比对", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过框架比对", skipped=True)
        loc = locales.load(self.locale)
        tags = spec.tags
        problems: list[str] = []
        expected = render.ManifestState(
            file_kind=tags.file_kind, change_kind=tags.change_kind, size=tags.size
        )
        parsed_manifest = render.manifest_of(state, loc)
        if parsed_manifest != expected:
            problems.append(f"清单行与注入特征对不上：解析出 {parsed_manifest}，应为 {expected}")
        checks = (
            ("审查规则表", render.rules_of(state, loc), render.rule_rows(loc)),
            ("严重度表", render.severity_of(state, loc), render.severity_rows(loc)),
            ("门控表", render.gate_of(state, loc), render.gate_rows(loc)),
        )
        for label, parsed, canonical in checks:
            if parsed != canonical:
                problems.append(f"{label}与规范表对不上：解析出 {len(parsed)} 行，应为 {len(canonical)} 行")
        if not problems:
            return Verdict(True, reason="清单行与三张表都与注入特征一致")
        return Verdict(False, reason="；".join(problems), raw={"manifest": str(parsed_manifest)})


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整条改动交回后端反解两维，必须与注入的一致。

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
        if not isinstance(spec, ReviewSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        if self.backend is None:
            return Verdict(True, reason="没有后端，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="text",
            domain="code-review",
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
