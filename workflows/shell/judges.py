"""命令审批 workflow 的判别器：构造式数据用纯规则判别即可（**零 token**）。

- `RoundtripJudge`：把请求文本反解回字段，与真值 spec **逐字**比对
  （抓渲染错位、字段漏写、命令被截断）；
- `LabelJudge`：反解文本后**独立重算**每道题的金答案（命令 → `classify` → 事实 → 答案），
  与数据里的答案比对。它只看渲染出来的文本与题面，不看 spec 里的答案——
  否则就是「拿答案验证答案」。

两个判别器都**先看执行环境**再认命令：方言认错会立刻表现为「命令认不出来」，
而不是静默用错模板、给出一个看似合理的答案。

输入/输出：只出 `Verdict`，不改样本。对不上就判不通过——宁可少要几条，
也不要一条「题面和命令对不上」的数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jc.judge import Verdict
from jc.workflow import family_of

from . import locales, parse, render
from . import spec as spec_mod
from .spec import TARGETS, CommandSpec, Target


def named_target(goal: str, dialect: str) -> Target | None:
    """输入：用户目标文本、方言。输出：文本里点名的对象（最长匹配），或 None。

    预期行为：只按**该方言**的路径表匹配，并取**最长**匹配——对象表里 `orders`（库）
    是 `…/v1/orders`（URL）的子串，只按「第一个命中」取会把 URL 目标认成库目标。
    """
    found = [target for target in TARGETS if target.path_in(dialect) in goal]
    if not found:
        return None
    return max(found, key=lambda target: len(target.path_in(dialect)))


def roundtrip_diff(spec: CommandSpec, parsed: parse.ParsedRequest, loc: dict[str, Any]) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：差异列表（空 = 一致）。

    预期行为：五个字段逐字比（连缩进被吃掉、命令被折行都算差异）。
    """
    diff: list[str] = []
    if parsed.dialect != spec.case.dialect:
        diff.append(f"执行环境：渲染 {parsed.dialect!r} vs 期望 {spec.case.dialect!r}")
    if parsed.cwd != spec.cwd:
        diff.append(f"工作目录：渲染 {parsed.cwd!r} vs 期望 {spec.cwd!r}")
    want_goal = render.goal_text(spec, loc)
    if parsed.goal != want_goal:
        diff.append(f"用户目标：渲染 {parsed.goal!r} vs 期望 {want_goal!r}")
    if parsed.command != spec.command:
        diff.append(f"命令：渲染 {parsed.command!r} vs 期望 {spec.command!r}")
    if parsed.reads != spec.reads:
        diff.append(f"已读文件：渲染 {parsed.reads!r} vs 期望 {spec.reads!r}")
    return diff


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：请求文本反解后必须与真值 spec 完全一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, CommandSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        try:
            parsed = parse.parse_request(sample["state"], parse.label_index(loc))
        except ValueError as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        diff = roundtrip_diff(spec, parsed, loc)
        return Verdict(not diff, reason="; ".join(diff[:5]), raw={"diff": diff})


def recompute(parsed: parse.ParsedRequest) -> dict[str, Any]:
    """输入：反解结果。输出：{题族名: 按文本重算的答案}。

    预期行为：**只吃文本**——命令走 `spec.classify`（按执行环境认出的方言），
    对象一致性走「用户目标里点名的对象是否就是命令操作的那个」。
    两个来源都不是 spec，所以它能独立验证 spec。
    """
    flags = spec_mod.classify(parsed.command, parsed.dialect)
    named = named_target(parsed.goal, parsed.dialect)
    aligned = named is not None and named.name == flags.target_name
    return {
        "verdict": spec_mod.verdict_of(flags, aligned),
        "target_owner": flags.owner,
        "matched_goal": aligned,
        "elevated": flags.elevated,
        "network": flags.network,
        "destructive": flags.destructive,
        "touches_secrets": flags.secret,
        "blast_radius": spec_mod.blast_of(flags),
    }


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
            parsed = parse.parse_request(sample["state"], parse.label_index(loc))
        except ValueError as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        problems = self._check(sample, parsed)
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )

    def _check(self, sample: dict[str, Any], parsed: parse.ParsedRequest) -> list[str]:
        """输入：样本、反解结果。输出：问题列表（空 = 全部对得上）。"""
        problems: list[str] = []
        if named_target(parsed.goal, parsed.dialect) is None:
            problems.append(f"用户目标里认不出任何对象：{parsed.goal!r}")
        try:
            expected = recompute(parsed)
        except ValueError as exc:
            return [*problems, f"命令认不出来：{exc}"]
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
                    problems.append(f"{qid}: 答案指向 {got}，按命令重算是 {want}")
            elif qtype == "noul":
                if answer["noul"] is not want:
                    problems.append(f"{qid}: 答案是 {answer['noul']}，按命令重算是 {want}")
            elif answer["score"] != want:
                problems.append(f"{qid}: 答案是 {answer['score']}，按命令重算是 {want}")
        return problems
