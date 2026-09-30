"""测试影响选择的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：改动得**兑现注入特征**——标题与改动都非空、不带围栏、长度落在
  区间内、行形状是 diff、按字面读出来的落点与计划逐条相同，而且**没有把两张表整段贴进去**；
  整段 state 不许有格式残留；
- `FrameJudge`（零 token）：state 的骨架必须完整且规范——覆盖表里 `P1`…`P5` 一个不少、顺序与
  `features.PATH_IDS` 一致，档位表里四档一个不少、顺序与 `features.LEVELS` 一致，提交标题与
  改动两块都解析得出来。任何一处对不上都说明渲染挂错了 case、或者语言文件缺了文案，标签会跟着
  一起错；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端**读一遍**，答「这次改动改到了哪些文件」，
  必须与计划算出来的落点一致。这是「渲染必须可反解」的落地：题面上的答案就是从 state 里读出来的。

为什么「改到了哪些文件」既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此重试），
模型那一遍是**读者视角的复核**——机械口径只认 `diff --git` 那一行，改动要是写成「本次改动调整了
handlers.py 里的边界判断」这种自指句，机械口径照样读得出落点（只要头一行在），而读者读出来的
东西没有意义，这种样本该丢。两遍问的不是同一件事。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，也不要一条标签和文本
对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后还拿不到草稿，只能记成
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
from .spec import ImpactSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签。
LEFTOVER_LINES: Final[frozenset[str]] = render.LEFTOVER_LINES
LEFTOVER_TAGS: Final[tuple[str, ...]] = render.LEFTOVER_TAGS


class TracePath(BaseModel):
    """反解草稿里的一条「改到了这个文件」：文件路径 + 让它站得住的那段原文。

    为什么要连原文一起要：模型报「改到了 handlers.py」却抄不出那一行，是**误报**，不是数据错
    ——改动里到底有没有这个文件的头是机械可查的（`features.HEADER_RE`），所以让它把证据抄出来，
    代码再复核一遍，抄不出来的那条直接不算。
    """

    file: str = Field(
        description="改动改到的文件路径（照 `diff --git a/… b/…` 里 b/ 后面那一段原样抄）"
    )
    quote: str = Field(
        description="从改动里**原样抄出来**的那一段——就是它证明这个文件被改到了"
    )


class TraceDraft(BaseModel):
    """反解草稿：只回答「这次改动改到了哪些文件」。

    取值域由 `features.PATH_FILES` 定，认不出来的文件路径一律算「没判成」（`skipped=True`）而不是
    「不通过」——那是反解格式没对齐，不是数据本身错。
    """

    files: list[TracePath] = Field(
        description="改动改到的文件（每个带路径与引文）；一个都没改到就给空数组"
    )


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。输出：提示词。
    预期行为：把整段 state（含覆盖表、档位表、提交标题与改动）原样交给后端，只问「改到了哪些
    文件」——反解用的口径与渲染期用的机械口径是同一份（`diff --git` 那一行），否则「答对了」与
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


def quote_holds(pid: str, quote: str, diff: str) -> bool:
    """输入：路径编号、反解抄出来的引文、改动正文。输出：这条「改到了」站不站得住。

    预期行为：两件事同时成立才算数——引文**原样出现在改动里**（空白折叠后比对，模型重排空白
    不算抄错），且引文里**真的带着这个文件在改动里的那一行**（`b/<文件路径>`）——只抄一段别的
    文件里的代码，不算证明了这条路径。
    """
    if not quote or _squash(quote) not in _squash(diff):
        return False
    return f"b/{features.PATH_FILES[pid]}" in quote


def _dropped_note(dropped: list[str]) -> str:
    """输入：被丢掉的误报路径编号。输出：附在理由后面的一句说明（没有误报就是空串）。"""
    if not dropped:
        return ""
    return f"（另有 {len(dropped)} 条误报：报出来的路径抄不出原文或引文不带该文件的头，已按没改到处理：{dropped}）"


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：改动必须兑现注入特征。

    字段：`locale` 语言（表格文案按它取，报错信息才读得懂）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：没有规格或 state 不是字符串时记「没判成」；有规格就按字面重算落点，与计划逐条
        比，再把 `render.draft_problems` 那一整套形状闸跑一遍。规格类型不对（串了别的 workflow）
        也算没判成。
        """
        locales.load(self.locale)
        if not isinstance(spec, ImpactSpec):
            return Verdict(True, reason="没有 spec，跳过兑现检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过兑现检查", skipped=True)
        parsed = render.parse_state(state)
        if parsed.diff is None or not parsed.diff.strip():
            return Verdict(False, reason="改动那一块解析不出来或为空")
        if parsed.title is None or not parsed.title.strip():
            return Verdict(False, reason="提交标题那一块解析不出来或为空")
        found = features.touch_key(parsed.touched)
        if found != spec.case.touch:
            return Verdict(
                False,
                reason=(
                    f"改动按字面读出来的落点是 {found or '无'}，"
                    f"计划要的是 {spec.case.touch or '无'}"
                ),
            )
        lang = render.detect_locale(state) or self.locale
        loc = locales.load(lang)
        problems = render.draft_problems(
            render.ContentDraft(title=parsed.title, diff=parsed.diff), spec.case, loc
        )
        if problems:
            return Verdict(False, reason="；".join(problems))
        leftovers = [
            line for line in (raw.strip() for raw in state.splitlines()) if line in LEFTOVER_LINES
        ]
        if leftovers:
            return Verdict(False, reason=f"state 里有格式残留：{sorted(set(leftovers))}")
        if any(tag in state for tag in LEFTOVER_TAGS):
            return Verdict(False, reason="state 里混进了 HTML 换行标签")
        return Verdict(True, reason="改动兑现了计划，且没有格式残留")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：state 的骨架必须完整且规范。

    字段：`locale` 语言。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：覆盖表里的编号必须是 `P1`…`P5` 且顺序一致、档位表里的档位必须是四档且顺序
        一致、提交标题与改动两块都解析得出来。
        """
        locales.load(self.locale)
        if not isinstance(spec, ImpactSpec):
            return Verdict(True, reason="没有 spec，跳过骨架检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过骨架检查", skipped=True)
        parsed = render.parse_state(state)
        problems: list[str] = []
        if parsed.paths != features.PATH_IDS:
            problems.append(
                f"覆盖表列出的编号是 {list(parsed.paths)}，应为 {list(features.PATH_IDS)}"
            )
        if parsed.levels != features.LEVELS:
            problems.append(
                f"档位表列出的档位是 {list(parsed.levels)}，应为 {list(features.LEVELS)}"
            )
        if parsed.title is None or not parsed.title.strip():
            problems.append("提交标题那一块解析不出来或为空")
        if parsed.diff is None or not parsed.diff.strip():
            problems.append("改动那一块解析不出来或为空")
        if not problems:
            return Verdict(True, reason="覆盖表、档位表与改动骨架都与规范形态一致")
        return Verdict(False, reason="；".join(problems))


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整段 state 交回后端读一遍，读出来的落点必须与计划一致。

    字段：`backend` 生成后端（由 `bind_backend` 注入）；`locale` 语言。
    """

    backend: GenerationBackend
    locale: str
    name: str = "trace"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：不一致时把「计划的」与「反解出来的」列在理由里，方便人看是哪里写歪了。端点故障、
        草稿缺字段与不认识的文件路径都只算「没判成」（`skipped=True`）——记成「不通过」会把端点
        抖动或格式漂移写成数据质量问题。
        """
        if not isinstance(spec, ImpactSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="choice",
            domain="testimpact",
            prompt=build_trace_prompt(str(sample["state"]), loc),
            response_model=TraceDraft,
            seeds=(),
            spec={
                "case": spec.case.as_dict(),
                "touch": list(spec.case.touched()),
                "files": [features.PATH_FILES[pid] for pid in spec.case.touched()],
                "flavor": spec.case.flavor,
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
        seen = draft.data.get("files")
        if not isinstance(seen, list):
            return Verdict(
                True,
                reason=f"反解草稿缺 files 字段，判不了：{seen!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        pairs: list[tuple[str, str]] = []
        for item in seen:
            if not isinstance(item, dict):
                return Verdict(
                    True,
                    reason=f"反解草稿的 files 项不是「路径 + 引文」对象，判不了：{item!r}",
                    raw={"raw_text": draft.raw_text[:500]},
                    skipped=True,
                )
            pairs.append((str(item.get("file", "")), str(item.get("quote", ""))))
        unknown = [path for path, _ in pairs if path not in features.FILE_TO_ID]
        if unknown:
            return Verdict(
                True,
                reason=f"反解草稿里出现覆盖表之外的文件路径 {unknown}，判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        body = render.parse_state(str(sample["state"])).diff or ""
        reported: list[str] = []
        dropped: list[str] = []
        for path, quote in pairs:
            pid = features.FILE_TO_ID[path]
            (reported if quote_holds(pid, quote, body) else dropped).append(pid)
        if set(reported) != set(labels.touched):
            return Verdict(
                False,
                reason=(
                    f"计划里改到 {list(labels.touched) or '无'}，"
                    f"反解读成 {reported or '无'}{_dropped_note(dropped)}"
                ),
                raw={"traced_paths": reported},
            )
        return Verdict(
            True,
            reason=f"反解与计划一致：改到 {list(labels.touched) or '无'}{_dropped_note(dropped)}",
            raw={"traced_paths": reported},
        )
