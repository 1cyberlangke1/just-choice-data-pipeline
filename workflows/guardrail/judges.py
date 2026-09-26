"""内容审核（护栏）workflow 的判别器：结构往返 + 答案重算。

输入：样本（`state` / `questions` / `answers`）、真值规格、语言字典。输出：`Verdict`。
预期行为：两个判别器都**只看文本**——`roundtrip` 把内容反解后与真值逐格比对；
`label` 反解后**独立重算**每个题族的答案。
重算的第一步是**认句**：正文每一行拿去和「六条条款 × 三种措辞的违规句」以及擦边句模板
做**逐行精确匹配**；只有匹配上违规句的才算违规，并按它们在正文里的先后决定 `top_clause`
的并列。擦边句与填充句不参与任何判断——这正是要训的那件事（见关键词就报警是错的）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jc.judge import Judge, Verdict
from jc.workflow import family_of

from . import locales, parse, render
from . import spec as spec_mod
from .spec import GuardSpec


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


def sentence_lookup(loc: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """输入：语言字典。输出：{整句文本: (类型, 条款)}。

    预期行为：违规句要把三种措辞的包裹都摊开（同一句在 plain / quoted / softened 下是
    三段不同的文本）；擦边句与填充句原样收。用**整行精确匹配**而不是子串匹配——
    包裹后的句子包含裸句，子串匹配会让 plain 与 quoted 同时命中，并列规则就废了。
    """
    out: dict[str, tuple[str, str]] = {}
    for clause in spec_mod.CLAUSES:
        raw = loc["sentence"]["violation"][clause]
        for wording, wrapper in loc["sentence"]["wrapper"].items():
            text = wrapper.format(s=raw)
            if text in out:
                raise ValueError(f"句子文案重复：{text!r}（{wording} / {out[text]}）")
            out[text] = ("violation", clause)
        mention = loc["sentence"]["mention"][clause]
        if mention in out:
            raise ValueError(f"句子文案重复：{mention!r}（擦边句 / {out[mention]}）")
        out[mention] = ("mention", clause)
    for filler in loc["sentence"]["filler"]:
        if filler in out:
            raise ValueError(f"句子文案重复：{filler!r}（填充句 / {out[filler]}）")
        out[filler] = ("filler", "")
    return out


def read_body(
    body: tuple[str, ...], loc: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """输入：正文每一句、语言字典。输出：(违规条款（按正文先后）, 擦边条款)。

    预期行为：逐行查表；查不到的句子直接报错——那说明渲染漏了模板或语言文件缺句，
    是硬伤，不能当成填充句蒙过去。返回的违规条款顺序**就是正文顺序**，
    `top_clause_of` 的并列规则直接吃它。
    """
    table = sentence_lookup(loc)
    violations: list[str] = []
    mentions: list[str] = []
    for line in body:
        found = table.get(line)
        if found is None:
            raise ValueError(f"正文里有一句认不出来：{line!r}")
        kind, clause = found
        if kind == "violation":
            violations.append(clause)
        elif kind == "mention":
            mentions.append(clause)
    return tuple(violations), tuple(mentions)


def recompute(message: parse.ParsedMessage, loc: dict[str, Any]) -> dict[str, Any]:
    """输入：反解出来的内容、语言字典。输出：{题族: 重算出来的答案}。

    预期行为：先把原文翻回键（传播面 / 方向 / 受众 / 历史按标签表反查，违规条款按正文反查），
    再调用 `spec` 那套规则函数——**与构造侧同一套规则**，所以「表格写错」与「渲染写歪」
    会暴露成同一类错误，不会两边各错一半还互相印证。
    """
    reach = reverse(loc["reach"])[message.reach]
    direction = reverse(loc["direction"])[message.direction]
    audience = reverse(loc["audience"])[message.audience]
    history = reverse(loc["history"])[message.history]
    violations, _mentions = read_body(message.body, loc)
    severity = spec_mod.severity_of(violations, reach)
    result: dict[str, Any] = {
        "severity": severity,
        "action": spec_mod.action_of(
            severity=severity,
            direction=direction,
            reach=reach,
            audience=audience,
            history=history,
            violations=violations,
        ),
        "top_clause": spec_mod.top_clause_of(violations, violations),
        "minor_exposure": spec_mod.minor_exposure_of(audience, violations),
        "repeat_offender": spec_mod.repeat_offender_of(history, severity),
    }
    for clause in spec_mod.CLAUSES:
        result[f"violates_{clause}"] = spec_mod.violated(clause, violations)
    return result


def roundtrip_diff(
    spec: GuardSpec, message: parse.ParsedMessage, loc: dict[str, Any]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐格一致）。"""
    checks = (
        ("内容号", message.content_id, spec.content_id),
        ("传播面", message.reach, loc["reach"][spec.case.reach]),
        ("方向", message.direction, loc["direction"][spec.case.direction]),
        ("受众", message.audience, loc["audience"][spec.case.audience]),
        ("历史", message.history, loc["history"][spec.case.history]),
        ("正文", "\n".join(message.body), "\n".join(render.body_lines(spec, loc))),
    )
    return [
        f"{name}：渲染 {got!r} vs 期望 {want!r}" for name, got, want in checks if got != want
    ]


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：内容反解后必须与真值规格逐格一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, GuardSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        try:
            message = parse.parse_message(sample["state"], parse.label_index(loc))
        except ValueError as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        diff = roundtrip_diff(spec, message, loc)
        return Verdict(not diff, reason="; ".join(diff[:5]), raw={"diff": diff})


@dataclass(frozen=True)
class LabelJudge:
    """答案判别器：反解文本后独立重算金答案，逐题比对。"""

    name: str = "label"
    locale: str = "zh"
    max_problems: int = 5

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec（不用）。输出：Verdict。预期行为：答案对不上就判不通过。"""
        loc = locales.load(self.locale)
        try:
            message = parse.parse_message(sample["state"], parse.label_index(loc))
            expected = recompute(message, loc)
        except (ValueError, KeyError) as exc:
            return Verdict(False, reason=f"重算失败：{exc}")
        problems = self._check(sample, expected)
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )

    def _check(self, sample: dict[str, Any], expected: dict[str, Any]) -> list[str]:
        """输入：样本、重算出来的答案。输出：问题列表（空 = 全部对得上）。"""
        problems: list[str] = []
        for qid, question in sample["questions"].items():
            want = expected.get(family_of(qid))
            if want is None:
                problems.append(f"{qid}: 未知题族")
                continue
            answer = sample["answers"][qid]
            qtype = question["type"]
            if qtype == "choice":
                options = list(question["criteria"])
                index = answer["choice"]
                if not 0 <= index < len(options):
                    problems.append(f"{qid}: 答案编号越界（{index}）")
                    continue
                got = options[index]
                if got != want:
                    problems.append(f"{qid}: 答案指向 {got}，按内容重算是 {want}")
            elif qtype == "noul":
                if answer["noul"] is not want:
                    problems.append(f"{qid}: 答案是 {answer['noul']}，按内容重算是 {want}")
            elif answer["score"] != want:
                problems.append(f"{qid}: 答案是 {answer['score']}，按内容重算是 {want}")
        return problems
