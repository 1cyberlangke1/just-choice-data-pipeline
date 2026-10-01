"""多视角 LLM 判别器（通用层，可选）：让模型以某个视角看数据，只出意见、不碰标签。

对应 PLAN 6.2 的「多视角判别器」。五只眼睛，按需开启（判别器名单里写 `llm:<视角>`）：

| 视角 | 看什么 |
|---|---|
| `solvable` | 只看素材与题面，答案能不能**唯一**确定 |
| `answer` | 金答案是不是真的对（独立重算一遍） |
| `distractors` | choice 的干扰项质量（同类、不重复、没有素材里不存在的取值） |
| `levels` | score 的等级是否互斥且覆盖（样本里没有 score 题就跳过） |
| `wording` | 措辞（通顺、无歧义、与素材语言一致、没有模板味） |

四条纪律：

1. **只出意见，不碰标签**：读的是组装好的样本（素材 + 题面 + 金答案），输出
   `{verdict, score, reason}`，标签一个字都不改；
2. **端点故障不算数据质量问题**：重试与降级都试过之后还失败，返回 `skipped=True`
   的结论——样本照收，但单独记账（`judge-skipped.jsonl`，见 `jc.pipeline`）；
3. **不适用就跳过**：样本里没有 score 题时 `levels` 没有对象，直接跳过，不白烧一次调用；
4. **要评测的模型别拿来过滤**：用哪个模型判别，就会删掉它自己不会做的样本，数据会系统性
   偏向它（PLAN 6.2）。判别器只做质检，不做「按模型口味删数据」。

输入：契约样本（`jc.contract` 形态）。输出：`Verdict`。
提示词在 `locales/judge/<lang>.toml`——质检也要说数据的语言，否则模型会拿错语言挑毛病。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field

from .backend.types import BackendError, Draft, GenerationBackend, GenerationTask
from .contract import gold_text
from .judge import LLM_PREFIX, Verdict

LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales") / "judge"

#: 五个视角。顺序只用于报错时列出来，没有别的含义。
PERSPECTIVES: Final[tuple[str, ...]] = ("solvable", "answer", "distractors", "levels", "wording")

#: 视角 → 它要求样本里有哪种题型；`None` = 什么样本都能看。
REQUIRES_KIND: Final[dict[str, str | None]] = {
    "solvable": None,
    "answer": None,
    "distractors": "choice",
    "levels": "score",
    "wording": None,
}


class Judgement(BaseModel):
    """模型交回来的结论（结构化输出的形状：字段少才不容易跑偏）。"""

    verdict: bool = Field(description="这个视角下有没有问题：true = 没问题")
    score: float = Field(description="把握程度，0~1")
    reason: str = Field(description="一句话依据：指出具体位置或具体毛病")


def available_locales() -> list[str]:
    """输入：无。输出：有质检提示词的语言列表（按文件名）。"""
    return sorted(path.stem for path in LOCALE_DIR.glob("*.toml"))


#: 反解提示词里「选项名用取值本身」的覆盖节写在哪：`verify.trace_options.<题族>`。
TRACE_OPTIONS_SECTION: Final[str] = "trace_options"


def trace_options(
    family: str, loc: dict[str, JsonValue], *, section: str = "fam"
) -> list[tuple[str, str]]:
    """输入：题族名、语言字典、题族所在的那一节（默认 `fam`）。输出：反解提示词的选项。

    预期行为：优先用 `verify.trace_options.<题族>`——那一节按**取值本身**列选项名；没有这一节
    就退回 `<section>.<题族>.criteria`。

    为什么要有这一节：题族在数据集里可能是 noul，`criteria` 的 key 是 `true` / `false`，
    而反解草稿要的是取值本身（`yes` / `no`、`clean` / `spam`）。两套词摆在一起，模型得自己
    翻译，实测就是翻译错——`search` 的 en 侧 6 条里 5 条把 `clean` 读成 `spam`，`review` 的
    `secret` / `debug` 也栽在同一处（真机拒稿理由全是「注入 yes，反解成 no」）。其余题族的
    key 本来就是取值，退回去与覆盖节等价，所以只有 noul 那几族需要写覆盖节。

    选项名与草稿取值的一致性由 `tests/core/test_trace_options.py` 逐族盯着，不靠人记。
    """
    verify = loc.get("verify")
    override = verify.get(TRACE_OPTIONS_SECTION) if isinstance(verify, dict) else None
    if isinstance(override, dict):
        node = override.get(family)
        if isinstance(node, dict):
            return [(str(name), str(desc)) for name, desc in node.items()]
    node = loc[section][family]["criteria"]
    assert isinstance(node, dict), f"{section}.{family}.criteria 不是表"
    return [(str(name), str(desc)) for name, desc in node.items()]


@lru_cache(maxsize=8)
def load(locale: str) -> dict[str, JsonValue]:
    """输入：语言代码。输出：提示词字典（带缓存）。

    预期行为：文件不存在时报人话并列出可用语言——提示词缺了就没法判，
    静默退回另一种语言会让「英文数据被中文挑毛病」这种假报警混进来。
    """
    path = LOCALE_DIR / f"{locale}.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"没有质检提示词 {path.name}；可用：{', '.join(available_locales())}"
        )
    with path.open("rb") as handle:
        return tomllib.load(handle)


def render_sample(sample: dict[str, JsonValue], loc: dict[str, JsonValue]) -> str:
    """输入：契约样本、语言字典。输出：题目与金答案的可读文本（提示词用）。

    预期行为：一律给**文字**而不是编号——编号是组装器的私事，模型看到编号只会困惑；
    选项按 criteria 的书写顺序列出，金答案取对应文字。
    """
    lines: list[str] = []
    for qid, question in sample.get("questions", {}).items():
        answer = sample.get("answers", {}).get(qid, {})
        lines.append(f"- {qid}（{question.get('type', '?')}）：{question.get('instructions', '')}")
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            joined = " / ".join(str(value) for value in criteria.values())
            lines.append(f"  {loc['options_label']}：{joined}")
        elif isinstance(criteria, list):
            joined = " / ".join(str(value) for value in criteria)
            lines.append(f"  {loc['levels_label']}：{joined}")
        lines.append(f"  {loc['answer_label']}：{gold_text(question, answer)}")
    return "\n".join(lines)


def build_prompt(sample: dict[str, JsonValue], loc: dict[str, JsonValue], perspective: str) -> str:
    """输入：契约样本、语言字典、视角。输出：提示词。

    预期行为：素材 + 题目与金答案 + 这个视角的判断标准。**一次只问一个视角**——
    一次问五个视角，模型只会挑最好答的那个答，另外四个的结论就是编的。
    """
    view = loc["perspectives"][perspective]
    lines = [
        loc["intro"],
        "",
        loc["material_head"],
        str(sample.get("state", "")),
        "",
        loc["question_head"],
        render_sample(sample, loc),
        "",
        f"{loc['perspective_head']}：{view['title']}",
        loc["rules_head"],
    ]
    lines += [f"- {rule}" for rule in view["rules"]]
    lines += ["", loc["closing"]]
    return "\n".join(lines)


def has_kind(sample: dict[str, JsonValue], kind: str) -> bool:
    """输入：契约样本、题型。输出：样本里有没有这种题型。"""
    return any(question.get("type") == kind for question in sample.get("questions", {}).values())


def to_verdict(perspective: str, draft: Draft) -> Verdict:
    """输入：视角名、后端草稿。输出：Verdict。

    预期行为：缺字段或类型不对按「没判成」处理（跳过），**绝不把解析失败当成数据有问题**；
    拿到结论就原样转（`ok` = 模型的 verdict，`score` = 它的把握）。
    """
    data = draft.data if isinstance(draft.data, dict) else {}
    raw = {
        "perspective": perspective,
        "provider": draft.provider,
        "model": draft.model,
        "mode": draft.mode,
        "score": data.get("score"),
        "raw_text": draft.raw_text[:1000],
    }
    verdict = data.get("verdict")
    if not isinstance(verdict, bool):
        return Verdict(
            True, reason=f"质检结论里没有 verdict 字段：{data!r}"[:200], raw=raw, skipped=True
        )
    score = data.get("score")
    return Verdict(
        ok=verdict,
        score=float(score) if isinstance(score, (int, float)) else None,
        reason=str(data.get("reason", "")),
        raw=raw,
    )


def make_judge(perspective: str, locale: str, backend: GenerationBackend) -> "LlmJudge":
    """输入：视角名、语言、后端。输出：判别器。

    预期行为：视角名不认识直接报错并列出可用的——静默忽略等于「写了名字却没跑」。
    """
    if perspective not in PERSPECTIVES:
        raise KeyError(f"未知质检视角 {perspective!r}（可用：{', '.join(PERSPECTIVES)}）")
    return LlmJudge(perspective=perspective, locale=locale, backend=backend)


@dataclass(frozen=True)
class LlmJudge:
    """一个视角的 LLM 判别器（实现 `jc.judge.Judge` 协议）。

    字段：视角名、语言、后端。**无状态**：同一条样本判两次结果一样（除了模型本身的抖动），
    所以并发跑多条样本时不用锁。
    """

    perspective: str
    locale: str
    backend: GenerationBackend

    @property
    def name(self) -> str:
        """输入：无。输出：判别器名（`llm:<视角>`）；统计与拒绝码都用它。"""
        return f"{LLM_PREFIX}{self.perspective}"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：契约样本、真值 spec（用不上——骨架真值就在样本里）。输出：Verdict。

        预期行为：不适用 → 跳过；端点故障（重试与降级都试过）→ 跳过并留原因；
        拿到结论 → 转成 Verdict。**任何情况下都不改样本**。
        """
        needed = REQUIRES_KIND[self.perspective]
        if needed is not None and not has_kind(sample, needed):
            return Verdict(
                True,
                reason=f"样本里没有 {needed} 题，{self.perspective} 视角不适用",
                skipped=True,
            )
        loc = load(self.locale)
        task = GenerationTask(
            task_id=f"{sample.get('id', 'sample')}-judge-{self.perspective}",
            kind="judge",  # 不是题型：这一步只出结论，题型与编号由组装器决定
            domain="quality-check",
            prompt=build_prompt(sample, loc, self.perspective),
            response_model=Judgement,
            spec={"perspective": self.perspective, "locale": self.locale},
        )
        try:
            draft = self.backend.generate(task)
        except BackendError as exc:
            if getattr(exc, "fatal_for_run", False):
                raise  # 全部 provider 都挂了：整批该停下（PLAN 3.1.4）
            return Verdict(
                True,
                reason=f"质检没跑成（端点故障，不是数据问题）：{exc}",
                raw={"error": type(exc).__name__},
                skipped=True,
            )
        return to_verdict(self.perspective, draft)
