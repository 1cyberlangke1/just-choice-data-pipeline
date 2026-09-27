"""模型路由与降级 workflow 的判别器：结构往返 + 答案重算。

输入：样本（`state` / `questions` / `answers`）、真值规格、语言字典。输出：`Verdict`。
预期行为：两个判别器都**只看文本**——
`roundtrip` 把画像反解后与真值逐格比对；`label` 反解后**独立重算**每个题族的答案。
重算走的是 `spec` 里那套规则函数（与构造侧同一套），但输入全部来自文本：
**任务类型只出现在正文里**，判别器得把正文读回去，这正是要训的那件事。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jc.judge import Judge, Verdict
from jc.workflow import family_of

from . import locales, parse, render
from . import spec as spec_mod
from .spec import RouteSpec


def reverse(table: dict[str, str]) -> dict[str, str]:
    """输入：{键: 原文}。输出：{原文: 键}。

    预期行为：原文重复时报错——重复会让反解有歧义，属于语言文件的硬伤。
    """
    out: dict[str, str] = {}
    for key, text in table.items():
        if text in out:
            raise ValueError(f"文案重复：{text!r}（{out[text]} 与 {key}）")
        out[text] = key
    return out


def task_of(body: str, loc: dict[str, Any]) -> str:
    """输入：正文原文、语言字典。输出：任务类型键。

    预期行为：看哪个**任务短语**原样出现在正文里；恰好一个才算数——认不出、或认出多个
    都报错（那是渲染与语言文件对不上的硬伤，不是「数据难判」，不能糊过去）。
    """
    hits = [
        task for task in spec_mod.TASKS if loc["task_phrase"][task] in body
    ]
    if len(hits) != 1:
        raise ValueError(f"正文里认不出唯一的任务类型：{hits}")
    return hits[0]


def recompute(request: parse.ParsedRequest, loc: dict[str, Any]) -> dict[str, Any]:
    """输入：反解出来的画像、语言字典。输出：{题族: 金答案}。

    预期行为：把每一格的**原文**翻回键，再走 `spec` 的规则函数重算。没有升级链时
    `next` **不出现在结果里**——题族缺一格与「多出一格」是两回事，`LabelJudge` 两边都查。
    """
    task = task_of(request.body, loc)
    out: dict[str, Any] = {
        "tier": spec_mod.tier_of(task),
        "effort": spec_mod.effort_of(
            reverse(loc["complexity"])[request.complexity],
            reverse(loc["latency"])[request.latency],
        ),
        "is_first": spec_mod.is_first_of(reverse(loc["retry"])[request.retry]),
        "degrade": spec_mod.degrade_of(reverse(loc["cost"])[request.cost]),
        "cache": spec_mod.cache_of(reverse(loc["data"])[request.data]),
    }
    if spec_mod.has_next(task):
        out["next"] = spec_mod.next_tier_of(task)
    return out


def roundtrip_diff(
    spec: RouteSpec, request: parse.ParsedRequest, loc: dict[str, Any]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐格一致）。"""
    checks = (
        ("请求号", request.request_id, spec.request_id),
        ("复杂度", request.complexity, loc["complexity"][spec.case.complexity]),
        ("延迟预算", request.latency, loc["latency"][spec.case.latency]),
        ("成本敏感", request.cost, loc["cost"][spec.case.cost]),
        ("尝试记录", request.retry, loc["retry"][spec.case.retry]),
        ("数据分级", request.data, loc["data"][spec.case.data]),
        ("正文", request.body, render.body_text(spec, loc)),
    )
    return [
        f"{name}：渲染 {got!r} vs 期望 {want!r}" for name, got, want in checks if got != want
    ]


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：画像反解后必须与真值规格逐格一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, RouteSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        try:
            request = parse.parse_request(sample["state"], parse.label_index(loc))
        except ValueError as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        diff = roundtrip_diff(spec, request, loc)
        return Verdict(not diff, reason="; ".join(diff[:5]), raw={"diff": diff})


@dataclass(frozen=True)
class LabelJudge:
    """答案判别器：反解文本后独立重算金答案，逐题比对。"""

    name: str = "label"
    locale: str = "zh"
    max_problems: int = 5

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec（不用）。输出：Verdict。预期行为：答案对不上就判不通过。"""
        del spec
        loc = locales.load(self.locale)
        try:
            request = parse.parse_request(sample["state"], parse.label_index(loc))
            expected = recompute(request, loc)
        except (ValueError, KeyError) as exc:
            return Verdict(False, reason=f"重算失败：{exc}")
        problems = self._check(sample, expected)
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )

    def _check(self, sample: dict[str, Any], expected: dict[str, Any]) -> list[str]:
        """输入：样本、重算出来的答案。输出：问题列表（空 = 全部对得上）。

        预期行为：逐题比对；**少题**那一趟遍历期望表（不是产物）——整族缺席时遍历产物
        会一声不响，漏报比误报危险。
        """
        problems: list[str] = []
        seen: set[str] = set()
        for qid, question in sample["questions"].items():
            family = family_of(qid)
            if family not in expected:
                problems.append(f"{qid}: 未知题族或不该出这道题")
                continue
            seen.add(family)
            want = expected[family]
            answer = sample["answers"][qid]
            qtype = question["type"]
            if qtype == "choice":
                options = list(question["criteria"])
                index = answer["choice"]
                if not 0 <= index < len(options):
                    problems.append(f"{qid}: 答案编号越界（{index}）")
                    continue
                got: Any = options[index]
            elif qtype == "noul":
                got = answer["noul"]
            else:
                got = answer["score"]
            if got != want:
                problems.append(f"{qid}: 答案是 {got!r}，按画像重算是 {want!r}")
        for family in sorted(set(expected) - seen):
            problems.append(f"{family} 少了 1 道题")
        return problems
