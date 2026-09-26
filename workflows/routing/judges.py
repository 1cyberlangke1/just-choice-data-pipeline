"""工单分诊 workflow 的判别器：结构往返 + 答案重算。

输入：样本（`state` / `questions` / `answers`）、真值规格、语言字典。输出：`Verdict`。
预期行为：两个判别器都**只看文本**——
`roundtrip` 把工单反解后与真值逐格比对；`label` 反解后**独立重算**每个题族的答案。
重算走的是 `spec` 里那套规则函数（与构造侧同一套），但输入全部来自文本：
**问题类型只出现在正文里**，判别器得把正文读回去，这正是要训的那件事。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from jc.judge import Judge, Verdict
from jc.workflow import family_of

from . import locales, parse, render
from . import spec as spec_mod
from .spec import TicketSpec


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


def issue_of(body: str, area_text: str, loc: dict[str, Any]) -> str:
    """输入：正文原文、产品域原文、语言字典。输出：问题类型键。

    预期行为：把每个问题句模板用同一个产品域名渲染一遍，看哪一句**原样**出现在正文里；
    恰好一句才算数——认不出、或认出多句都报错（那是渲染与语言文件对不上的硬伤，
    不是「数据难判」，不能糊过去）。
    """
    hits = [
        key
        for key in spec_mod.ISSUES
        if loc["body"]["issue"][key].format(area=area_text) in body
    ]
    if len(hits) != 1:
        raise ValueError(f"正文里认不出唯一的问题类型：{hits}")
    return hits[0]


def elapsed_of(created: str, current: str) -> int:
    """输入：创建时间、当前时间（`HH:MM`）。输出：已等待分钟数。

    预期行为：按同一天的钟点相减；当前时间早于创建时间就报错（渲染写反了）。
    """
    start = dt.datetime.strptime(created, "%H:%M")
    end = dt.datetime.strptime(current, "%H:%M")
    minutes = int((end - start).total_seconds() // 60)
    if minutes < 0:
        raise ValueError(f"当前时间早于创建时间：{created!r} → {current!r}")
    return minutes


def recompute(ticket: parse.ParsedTicket, loc: dict[str, Any]) -> dict[str, Any]:
    """输入：反解出来的工单、语言字典。输出：{题族: 重算出来的答案}。

    预期行为：先把原文翻回键（产品域 / 等级 / 影响面 / 紧急度 / 合规标记按标签表反查，
    问题类型按正文反查），再调用 `spec` 那套规则函数——**与构造侧同一套规则**，
    所以「表格写错」与「渲染写歪」会暴露成同一类错误，不会两边各错一半还互相印证。
    """
    area = reverse(loc["area"])[ticket.area]
    tier = reverse(loc["tier"])[ticket.tier]
    impact = reverse(loc["impact"])[ticket.impact]
    urgency = reverse(loc["urgency"])[ticket.urgency]
    compliance = reverse(loc["compliance"])[ticket.compliance]
    issue = issue_of(ticket.body, ticket.area, loc)
    elapsed = elapsed_of(ticket.created, ticket.current)
    severity = spec_mod.severity_of(impact, urgency)
    limit = spec_mod.sla_limit_of(severity, tier)
    return {
        "severity": severity,
        "queue": spec_mod.queue_of(issue, compliance),
        "escalation": spec_mod.escalation_of(compliance),
        "critical_area": spec_mod.critical_area_of(area),
        "needs_human": spec_mod.needs_human_of(severity, compliance, tier),
        "sla_breach": spec_mod.sla_breach_of(elapsed, limit),
        "data_risk": spec_mod.data_risk_of(issue, compliance),
        "priority": spec_mod.priority_of(severity, tier),
    }


def roundtrip_diff(
    spec: TicketSpec, ticket: parse.ParsedTicket, loc: dict[str, Any]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐格一致）。"""
    checks = (
        ("工单号", ticket.ticket_id, spec.ticket_id),
        ("产品域", ticket.area, loc["area"][spec.case.area]),
        ("客户等级", ticket.tier, loc["tier"][spec.case.tier]),
        ("影响面", ticket.impact, loc["impact"][spec.case.impact]),
        ("紧急度", ticket.urgency, loc["urgency"][spec.case.urgency]),
        ("合规标记", ticket.compliance, loc["compliance"][spec.case.compliance]),
        ("创建时间", ticket.created, spec.created),
        ("当前时间", ticket.current, spec.current),
        ("正文", ticket.body, render.body_text(spec, loc)),
    )
    return [
        f"{name}：渲染 {got!r} vs 期望 {want!r}"
        for name, got, want in checks
        if got != want
    ]


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：工单反解后必须与真值规格逐格一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, TicketSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        try:
            ticket = parse.parse_ticket(sample["state"], parse.label_index(loc))
        except ValueError as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        diff = roundtrip_diff(spec, ticket, loc)
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
            ticket = parse.parse_ticket(sample["state"], parse.label_index(loc))
            expected = recompute(ticket, loc)
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
                    problems.append(f"{qid}: 答案指向 {got}，按工单重算是 {want}")
            elif qtype == "noul":
                if answer["noul"] is not want:
                    problems.append(f"{qid}: 答案是 {answer['noul']}，按工单重算是 {want}")
            elif answer["score"] != want:
                problems.append(f"{qid}: 答案是 {answer['score']}，按工单重算是 {want}")
        return problems
