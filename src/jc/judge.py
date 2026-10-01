"""判别器抽象（通用层）：判断「这条样本对不对」。

设计要点（对应 PLAN 6.3）：
- 判别器不是真理：输出必须带理由与原始结果，不能只给 pass/fail；
- 判别器由 **workflow 提供**（`Workflow.make_judges`），通用层只定义协议与聚合方式；
- 判别器本身可被评估：把人工抽检当金标准即可（`reviews.jsonl`）。

输入/输出：`Verdict` 是结论；`run_judges` 跑一组判别器并返回 (名字, 结论) 列表。
"""

from __future__ import annotations
from . import jsonio
from jc.jsontypes import JsonValue

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

if TYPE_CHECKING:  # 只在类型检查时引入：通用层不装 [llm] extra 也要能跑
    from .backend.types import GenerationBackend

#: LLM 判别器的名字前缀（`llm:solvable`）。通用层负责装配它们（要后端），
#: 各 workflow 只管在自己的判别器名单里写名字，不必重复接线。
LLM_PREFIX: Final[str] = "llm:"


def squash(text: str) -> str:
    """输入：一段文本。输出：空白折叠成单个空格之后的文本。

    预期行为：只为比对用，不改样本本身——判别器拿它把「换行 / 多空格」的差异抹平，
    免得同一句话因为排版不同被判成两回事。
    """
    return " ".join(text.split())


def quote_candidates(raw: str) -> tuple[str, ...]:
    """输入：反解交回来的引文字段（声明是 str）。输出：候选引文，按原顺序去重，至少一条。

    预期行为：字段本该是**一段**原样抄出来的文本，读者却常把「抄了几段」直接塞进来——一整个
    JSON 数组字符串、一串连外层方括号都丢了的带引号元素、几段用换行拼在一起、每段外面还各自
    裹一层 JSON 引号（真机实测四种都出过，见范式 §9.88）。这里把包装一层层拆掉：`_as_array`
    拆数组 → 按行拆 → 剥外层引号。**拆出来的每一段仍须原样出现在正文里才算数**（由 `quote_hit`
    判），所以拆包装只会少丢真命中，不会多认误报。
    """
    pieces: list[str] = []
    for chunk in _as_array(raw) or (raw,):
        for line in chunk.splitlines() or [chunk]:
            if not line.strip():
                continue
            pieces.append(line)
            bare = _unquote(line)
            if bare != line and bare.strip():
                pieces.append(bare)
    unique = tuple(dict.fromkeys(pieces))
    return unique or (raw,)


def _as_array(raw: str) -> tuple[str, ...] | None:
    """输入：一段文本。输出：它作为「JSON 字符串数组」读出来的元素（读不出来就是 None）。

    预期行为：认两种写法——**本身就是数组**（`["甲", "乙"]`），以及读者连外层方括号一起丢了、
    只剩一串带引号的元素（`"甲", "乙"`）。空数组、没有字符串元素、语法坏，一律给 None，
    由调用方退回原文。
    """
    text = raw.strip()
    form = text if text.startswith("[") and text.endswith("]") else f"[{text}]"
    try:
        loaded = jsonio.loads(form)
    except jsonio.StrictJsonError:
        return None
    if not isinstance(loaded, list):
        return None
    items = tuple(item for item in loaded if isinstance(item, str) and item.strip())
    return items or None


def _unquote(text: str) -> str:
    """输入：一段文本。输出：剥掉最外层 JSON 引号之后的文本（不是 JSON 字符串就原样返回）。"""
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        try:
            loaded = jsonio.loads(text)
        except jsonio.StrictJsonError:
            return text
        if isinstance(loaded, str):
            return loaded
    return text


def quote_hit(quote: str, body: str) -> str:
    """输入：引文字段、正文。输出：第一条**原样出现在正文里**的候选（都没有就是空串）。

    预期行为：逐个候选按空白折叠后比对，返回命中的那一条**原文**——调用方接着拿它比机械判据，
    比的是候选本身而不是整个数组字符串（拿整串去比必然为假，那条命中会被当成误报丢掉）。
    """
    squashed_body = squash(body)
    for candidate in quote_candidates(quote):
        squashed = squash(candidate)
        if squashed and squashed in squashed_body:
            return candidate
    return ""


def leftover_in(state: str, lines: Iterable[str], tags: Iterable[str]) -> str:
    """输入：state 文本、残留行清单、残留标记清单。输出：命中的残留（没有就空串）。

    预期行为：先按**整行**比对（去掉首尾空白后全等，所以围栏单独成行才算残留），再按**子串**
    比对标签；两份清单的顺序就是优先级，先命中的先返回。清单由各 workflow 给（判什么算残留是
    领域知识），比对本身是通用的。
    """
    for line in state.splitlines():
        if line.strip() in lines:
            return line.strip()
    for tag in tags:
        if tag in state:
            return tag
    return ""


def partition_names(names: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """输入：判别器名单。输出：(规则判别器名, LLM 视角名)。

    预期行为：只按前缀切分，顺序与调用方声明的一致，不改写也不去重；
    `llm:` 后面留空（或写了不存在的视角）由装配方报错，不在这里静默丢掉。
    """
    rules = tuple(name for name in names if not name.startswith(LLM_PREFIX))
    llm = tuple(name[len(LLM_PREFIX) :] for name in names if name.startswith(LLM_PREFIX))
    return rules, llm


@dataclass(frozen=True)
class Verdict:
    """一条判别结论。

    字段：ok 是否通过；score 可选分数（0~1）；reason 人类可读理由；raw 原始结果（留档审计）；
    skipped 表示**这条根本没判成**（例如端点故障、视角不适用）。

    `skipped=True` 时 `ok` 不代表数据好坏，消费方必须先看 `skipped`：把「没判成」当成
    「不通过」会把端点抖动记成数据质量问题（实测踩过），当成「通过」又不该悄悄发生——
    所以两者都不是，单独记一笔账。
    """

    ok: bool
    score: float | None = None
    reason: str = ""
    raw: dict[str, JsonValue] = field(default_factory=dict)
    skipped: bool = False


@runtime_checkable
class Judge(Protocol):
    """判别器协议：给样本（与可选真值 spec）出一个结论。"""

    name: str

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：只判定、不修改样本。"""


def run_judges(
    sample: dict[str, JsonValue], spec: object, judges: tuple[Judge, ...]
) -> list[tuple[str, Verdict]]:
    """依次跑所有判别器。

    输入：样本、真值 spec、判别器元组。
    输出：[(判别器名, Verdict)]，顺序与入参一致。
    预期行为：任何判别器抛异常都转成「不通过」的 Verdict（不让管线崩），
    **除了标了 `fatal_for_run` 的**——那是「整批该停下」（全部 provider 都挂了），
    不是「这条样本不对」，吞掉它会产出一整个「全被拒」的假数据集。
    """
    out: list[tuple[str, Verdict]] = []
    for judge in judges:
        try:
            out.append((judge.name, judge.judge(sample, spec)))
        except Exception as exc:  # noqa: BLE001 —— 判别器崩了要记下来，不能影响整批
            if getattr(exc, "fatal_for_run", False):
                raise
            out.append((judge.name, Verdict(False, reason=f"判别器异常：{exc}")))
    return out


class JudgeModule(Protocol):
    """一个 workflow 的 `judges` 模块：三件套判别器由各包自己定义。"""

    HygieneJudge: Callable[[str], Judge]
    FrameJudge: Callable[[str], Judge]
    TraceJudge: Callable[["GenerationBackend", str], Judge]


def standard_judges(
    names: tuple[str, ...],
    locale: str,
    module: JudgeModule,
    backend: "GenerationBackend | None",
) -> tuple[Judge, ...]:
    """输入：判别器名单、语言、workflow 的 judges 模块、后端。输出：判别器实例。

    预期行为：`hygiene` / `frame` 只吃语言，`trace` 还要后端（没被注入直接报错——
    静默降级会让「文本由模型写」这个前提失效），未知名字直接报错并列出可用名字。
    19 个包的 `make_judges` 逐字相同，机制只留这一份，各包只管把模块与后端交进来。
    """
    out: list[Judge] = []
    for name in names:
        if name == "hygiene":
            out.append(module.HygieneJudge(locale))
        elif name == "frame":
            out.append(module.FrameJudge(locale))
        elif name == "trace":
            if backend is None:
                raise RuntimeError("trace 判别器需要后端，但还没被 bind_backend 注入")
            out.append(module.TraceJudge(backend, locale))
        else:
            raise KeyError(f"未知判别器：{name!r}（可用：hygiene / frame / trace）")
    return tuple(out)


def verify_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。输出：提示词。
    预期行为：把整段 state（判据表 / 档位表 / 素材原文都在里面）原样交给后端，只问一句
    「踩了哪几条」——反解用的口径与渲染期用的机械口径是同一份，否则「答对了」与
    「标签对了」不是一回事。18 个包各抄一份，机制只留这一份。
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
