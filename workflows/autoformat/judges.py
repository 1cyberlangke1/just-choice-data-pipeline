"""结构恢复的判别器：把产物与「文本本身」对一遍。

输入：样本（含 state / questions / answers）、真值规格。输出：`Verdict`。
预期行为：三个判别器走**三条不同的路**：

1. `hygiene`（零 token）——state 里不许有排版标记残留，各块的字面要求（长度、结尾标点、标记词）
   必须成立。这条路抓的是「渲染把标记漏出来了」；
2. `roundtrip`（零 token）——从 state **文本**反解出块结构，再重算四个题族的答案，逐题比对。
   这条路抓的是「答案与文本不符」，它用的读法与构造侧完全独立；
3. `trace`（一次调用）——把纯文本原样交给读者模型，问它读出了什么结构，再与计划比。这条路抓的是
   「机械口径对得上、但人读起来是另一回事」。

`recompute` 与 `compare` 都对外暴露：`audit` 要拿它们在**产物层面**重算同一套东西，这样老产物也能
事后查（不用重跑生成）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Judge, Verdict
from jc.jsontypes import JsonValue
from jc.workflow import family_of
from pydantic import BaseModel, Field

from . import features, locales, parse, questions
from .features import KIND_CALLOUT, KIND_CODE, KIND_HEADING, KIND_QUOTE
from .spec import DocSpec

# 排版标记：行首出现这些，说明「标记被剥掉」这件事没兑现。
MARKER_PREFIXES: Final[tuple[str, ...]] = ("#", "- ", "* ", "> ", "|")
# 行内出现这些同样算残留（加粗、代码围栏）。
MARKER_INFIX: Final[tuple[str, ...]] = ("**", "`" * 3)


class TraceDraft(BaseModel):
    """反解草稿：只回答「这段纯文本的块结构是什么」。

    取值域由 `features.BLOCK_KINDS` 定；认不出来的类型一律算「没判成」（`skipped=True`）而不是
    「不通过」——那是反解格式没对齐，不是数据本身错。
    """

    kinds: list[str] = Field(description="按从上到下的顺序，每个内容块的类型")
    callout_kind: str = Field(default="", description="有提示框就写 note / tip / warning，没有就留空")
    list_ordered: bool = Field(default=False, description="有列表就写它有没有先后顺序，没有就填 false")


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。输出：提示词。
    预期行为：把整段 state 原样交给后端，只问「读出了什么结构」——反解用的口径与渲染期用的机械
    口径是同一份（读法都印在题面里），否则「答对了」与「标签对了」不是一回事。
    「列表有没有序」那一条的判据在题面里点名**顺序词**（顺序词表由语言表给），
    免得读者按语义去猜「这几项看起来像有先后顺序」——那正是这一条最容易吵起来的地方。
    """
    steps = str(loc["doc"]["list_sep"]).join(parse.seq_words(loc))
    return "\n".join([str(loc["verify"]["ask"]).format(steps=steps), "", state])


def option_names(question: dict[str, JsonValue]) -> list[str]:
    """输入：一道 choice 题。输出：按**产物里写的顺序**排好的选项名。

    预期行为：答案编号就是在这个顺序里的下标（见 `jc.assemble`）——审计要拿它把编号换回选项名与
    选项描述，才能与纯代码重算的结果比对。
    """
    criteria = question.get("criteria")
    return list(criteria) if isinstance(criteria, dict) else []


def answer_value(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> JsonValue:
    """输入：一道题、它的答案。输出：可比的值。

    预期行为：choice 给**选项名**（不是编号——编号只由组装器产生），noul 给布尔；编号越界或类型
    不对给 None（那是对不上，不是崩）。
    """
    if question["type"] == "choice":
        names = option_names(question)
        index = answer.get("choice")
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        if not 0 <= index < len(names):
            return None
        return names[index]
    if question["type"] == "noul":
        value = answer.get("noul")
        return value if isinstance(value, bool) else None
    return answer.get("score")


def recompute(doc: parse.ParsedDoc, loc: dict[str, JsonValue]) -> dict[str, list[JsonValue]]:
    """输入：反解出来的文档、语言字典。输出：{题族: 期望答案列表}（按题序）。

    预期行为：完全从**文本**重算——并段由块切分给，块类型由块本身给，提示框种类由关键词给，
    列表有序性由步骤词给。
    """
    commands = parse.command_tokens(loc)
    pairs = parse.balanced_pairs(parse.asked_pairs(doc.lines, commands), doc.same_block)
    callouts = doc.callouts()
    items = doc.list_items()
    return {
        "join": [doc.same_block(left, right) for left, right in pairs],
        "block_type": list(doc.kinds()),
        "callout_kind": [callouts[0].callout_kind] if callouts else [],
        "list_ordered": [items[0].list_ordered] if items else [],
    }


def compare(sample: dict[str, JsonValue], expected: dict[str, list[JsonValue]]) -> list[str]:
    """输入：一条样本、期望答案表。输出：差异列表（空 = 全对）。

    预期行为：按题号走一遍，同族按出现顺序对上第几条；题多了、题少了、答案不对都各记一笔——
    不静默取第一个。
    """
    seen: dict[str, int] = {family: 0 for family in questions.FAMILIES}
    problems: list[str] = []
    questions_field = sample["questions"]
    answers_field = sample["answers"]
    if not isinstance(questions_field, dict) or not isinstance(answers_field, dict):
        return ["样本的 questions / answers 不是对象"]
    for qid, question in questions_field.items():
        family = family_of(qid)
        if family not in seen:
            problems.append(f"未知题族 {family!r}（题号 {qid}）")
            continue
        index = seen[family]
        seen[family] += 1
        want = expected.get(family, [])
        if index >= len(want):
            problems.append(f"{qid} 多出来了：文本里只算得出 {len(want)} 道 {family} 题")
            continue
        answer = answers_field.get(qid, {})
        got = answer_value(question, answer if isinstance(answer, dict) else {})
        if got != want[index]:
            problems.append(f"{qid}：答案是 {got!r}，按文本重算应为 {want[index]!r}")
    for family, count in seen.items():
        want = expected.get(family, [])
        if count < len(want):
            problems.append(f"{family} 少了 {len(want) - count} 道题")
    return problems


def _marker_leftover(state: str) -> str:
    """输入：state 文本。输出：命中的排版标记残留（没有就空串）。"""
    for line in state.splitlines():
        body = line.partition("|")[2].strip()
        if not body:
            continue
        for prefix in MARKER_PREFIXES:
            if body.startswith(prefix):
                return f"行首标记 {prefix!r}"
    for infix in MARKER_INFIX:
        if infix in state:
            return f"行内标记 {infix!r}"
    return ""


def _block_problems(doc: parse.ParsedDoc, loc: dict[str, JsonValue]) -> list[str]:
    """输入：反解出来的文档、语言字典。输出：块层面不合规的地方。

    预期行为：逐块查标记词有没有跑到不该出现的块里，以及标题有没有以标点结尾——两条都是
    **从文本本身**读出来的。
    """
    words = parse.callout_words(loc)
    marks = parse.attribution_marks(loc)
    problems: list[str] = []
    for index, block in enumerate(doc.blocks, start=1):
        text = block.text
        if block.kind == KIND_CALLOUT:
            if not any(text.startswith(word) for word in words):
                problems.append(f"第 {index} 块被读成提示框，却没以任何关键词开头：{text[:30]!r}")
            continue
        if block.kind == KIND_CODE:
            # 命令行块豁免标记词检查：读法里「全命令行 → 代码块」这一条排在标记词判据之前，
            # 且它的判据是词法的（每行以工具名开头）。英文署名前缀 `--` 与长选项（`--version`）
            # 同形，不豁免的话每个英文代码块都会被假判。
            continue
        if any(word in text for word in words):
            problems.append(f"第 {index} 块（{block.kind}）里出现了提示框关键词：{text[:30]!r}")
        if block.kind != KIND_QUOTE and any(mark in text for mark in marks):
            problems.append(f"第 {index} 块（{block.kind}）里出现了署名前缀：{text[:30]!r}")
        if block.kind == KIND_HEADING and parse.ends_terminal(text):
            problems.append(f"第 {index} 块是标题，却以标点结尾：{text!r}")
    return problems


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：state 得反解得出、且不许有排版标记残留。"""

    name = "hygiene"
    locale: str

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本（与忽略的规格）。输出：Verdict。"""
        del spec
        loc = locales.load(self.locale)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(False, reason=f"state 不是字符串：{type(state).__name__}")
        leftover = _marker_leftover(state)
        if leftover:
            return Verdict(False, reason=f"state 里有排版标记残留：{leftover}")
        try:
            doc = parse.parse_state(state, loc)
        except ValueError as exc:
            return Verdict(False, reason=f"state 反解失败：{exc}")
        problems = _block_problems(doc, loc)
        if problems:
            return Verdict(False, reason="；".join(problems[:3]), raw={"problems": problems})
        return Verdict(True, reason="没有标记残留，各块的标记词都落在该落的地方")


@dataclass(frozen=True)
class RoundtripJudge:
    """从 state 文本反解后重算四个题族，逐题比对。"""

    name = "roundtrip"
    locale: str

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本（与忽略的规格）。输出：Verdict。"""
        del spec
        loc = locales.load(self.locale)
        try:
            doc = parse.parse_state(str(sample["state"]), loc)
        except ValueError as exc:
            return Verdict(False, reason=f"state 反解失败：{exc}")
        problems = compare(sample, recompute(doc, loc))
        if problems:
            return Verdict(False, reason="；".join(problems[:3]), raw={"problems": problems})
        return Verdict(True, reason="文本反解后四个题族全部对上")


@dataclass(frozen=True)
class TraceJudge:
    """把纯文本交给读者模型，问它读出了什么结构，再与计划比。

    预期行为：只比**这份计划里真的有的东西**——没有提示框就不比 `callout_kind`，没有列表就不比
    `list_ordered`。类型序列则一定比（它每份文档都有）。认不出来的类型算 `skipped`。
    """

    name = "trace"
    backend: GenerationBackend
    locale: str

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：端点故障、草稿缺字段、认不出的块类型都只算「没判成」（`skipped=True`）——
        记成「不通过」会把端点抖动或格式漂移写成数据质量问题；只有「读出来的结构和计划不一致」
        才是真的不通过。全部 provider 都挂了则往上抛（整批该停下）。
        """
        if not isinstance(spec, DocSpec):
            return Verdict(
                True,
                reason=f"没有 spec，跳过反解（拿到 {type(spec).__name__}）",
                skipped=True,
            )
        loc = locales.load(self.locale)
        state = str(sample["state"])
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="choice",
            domain="autoformat",
            prompt=build_trace_prompt(state, loc),
            response_model=TraceDraft,
            seeds=(),
            spec={},
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
        reported = draft.data.get("kinds")
        if not isinstance(reported, list):
            return Verdict(
                True,
                reason=f"反解草稿缺 kinds 字段，判不了：{draft.data!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        kinds = [str(item) for item in reported]
        unknown = [item for item in kinds if item not in features.BLOCK_KINDS]
        if unknown:
            return Verdict(
                True,
                reason=f"反解给出了认不出的块类型 {unknown}，判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        want = list(spec.plan.kinds())
        if kinds != want:
            return Verdict(
                False,
                reason=f"读者读出的块类型是 {kinds}，计划要的是 {want}",
                raw={"traced_kinds": kinds},
            )
        callouts = spec.plan.callout_kinds()
        if callouts and str(draft.data.get("callout_kind", "")) != callouts[0]:
            return Verdict(
                False,
                reason=f"读者读出的提示框种类是 {draft.data.get('callout_kind')!r}，计划是 {callouts[0]!r}",
            )
        flags = spec.plan.list_flags()
        if flags and bool(draft.data.get("list_ordered", False)) != flags[0]:
            return Verdict(
                False,
                reason=f"读者读出的列表有序性是 {draft.data.get('list_ordered')!r}，计划是 {flags[0]!r}",
            )
        return Verdict(True, reason="读者模型读出的结构与计划一致", raw={"traced_kinds": kinds})


def all_judges(locale: str, backend: GenerationBackend | None) -> tuple[Judge, ...]:
    """输入：语言、可选后端。输出：三个判别器（trace 需要后端，没有就不给）。"""
    out: list[Judge] = [HygieneJudge(locale), RoundtripJudge(locale)]
    if backend is not None:
        out.append(TraceJudge(backend, locale))
    return tuple(out)
