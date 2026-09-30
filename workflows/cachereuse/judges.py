"""缓存复用判定（cachereuse）workflow 的判别器：反解 state 后独立重算四个答案。

输入：样本（state + 题目 + 答案）+ 真值规格。输出：`Verdict`。
预期行为：
- `roundtrip`：反解出来的两条轨迹，必须与真值规格**逐步逐字一致**（含来源那一行）；
- `label`：只看 state，把每一步的原文归一化成键之后独立重算四个答案再与产物比对——
  所以「生成器把答案算错了」与「渲染把某一步写歪了」会分别被抓到。

归一化不复用渲染器的任何计算：写法表在语言文件里，反查实现在本文件的 `to_keys` 里，
两边靠 `tests/workflows/cachereuse/test_locales.py` 与题面一致性测试锁住。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from collections.abc import Callable
from dataclasses import dataclass
from jc.judge import Verdict
from jc.workflow import family_of

from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import ReuseSpec, Step

def to_keys(side: tuple[tuple[str, str], ...], words: dict[str, JsonValue]) -> tuple[Step, ...]:
    """输入：一侧的 (动作原文, 元素原文) 序列、合并反查表。输出：步骤键序列。

    预期行为：动作与元素各自查表；**查不到就报人话**——那说明渲染写出了表外的写法，
    不能静默当成「不同」（那会把渲染 bug 伪装成一道难题）。
    """
    actions = words["action"]
    targets = words["target"]
    out: list[Step] = []
    for action, target in side:
        if not isinstance(actions, dict) or not isinstance(targets, dict):
            raise ValueError("反查表形状不对（期望 {写法: 键}）")
        if action not in actions:
            raise ValueError(f"动作不在写法表里：{action!r}")
        if target not in targets:
            raise ValueError(f"元素不在写法表里：{target!r}")
        out.append((str(actions[action]), str(targets[target])))
    return tuple(out)


def flavor_key(text: str, words: dict[str, JsonValue]) -> str:
    """输入：来源那一行的原文、合并反查表。输出：来源键。查不到直接报错（同上）。"""
    table = words["flavor"]
    if not isinstance(table, dict) or text not in table:
        raise ValueError(f"来源不在写法表里：{text!r}")
    return str(table[text])


def facts_of(parsed: parse.ParsedReuse, words: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """输入：反解结果、合并反查表。输出：从文本重算出来的四个答案（外加三个派生量）。

    预期行为：先把原文归一化成键，再交给 `spec.observed_labels`——**标签规则只有一份实现**
    （在 `spec` 里），构造侧与审计侧共用，免得两处各写一遍迟早漂移。
    """
    labels = spec_mod.observed_labels(
        to_keys(parsed.cached, words), to_keys(parsed.current, words)
    )
    labels["flavor"] = flavor_key(parsed.flavor, words)
    return labels


def option_text(family: str, facts: dict[str, JsonValue], loc: dict[str, JsonValue]) -> str:
    """输入：题族、事实、语言字典。输出：该题族金答案对应的**选项描述原文**。

    预期行为：用与 `questions.options_of` 同一个模板拼出来再比——比「从描述里切出编号」稳，
    模板改了也不会悄悄比错。
    """
    if family == "diverge_at":
        return str(loc["fam"]["diverge_at"]["criteria"][str(facts["diverge"])])
    if family == "diff_kind":
        return str(loc["fam"]["diff_kind"]["criteria"][str(facts["diff_kind"])])
    raise ValueError(f"未知题族：{family!r}（选项只有 diverge_at 与 diff_kind 两族）")


def expected_of(family: str, facts: dict[str, JsonValue], loc: dict[str, JsonValue]) -> str | bool | int:
    """输入：题族、事实、语言字典。输出：按 state 重算出来的答案。

    预期行为：choice 比选项描述原文、noul 比布尔、score 比档位下标；未知题族直接报错
    （新加了题族却忘了在这儿实现，不能静默放过）。
    """
    if family in questions.CHOICE_FAMILIES:
        return option_text(family, facts, loc)
    if family == "reuse_half":
        return bool(facts["reuse_half"])
    if family == "reuse_band":
        return spec_mod.BANDS.index(str(facts["band"]))
    raise ValueError(f"未知题族：{family!r}（可用：{questions.FAMILIES}）")


def check_sample(
    sample: dict[str, JsonValue], facts: dict[str, JsonValue], loc: dict[str, JsonValue]
) -> list[str]:
    """输入：样本、从 state 重算出来的事实、语言字典。输出：问题列表（空 = 全部对得上）。

    预期行为：与 `check_sample_any_locale` 同一把尺，只是选项描述**只认这一种语言**的写法
    ——判别器判的是某一条确定的样本，它的语言是确定的，另一种语言的文案就是渲染写歪了。
    做成模块级函数是为了让 `audit` 也能直接复用同一套判据（判别器与审计必须用同一把尺）。
    """
    return _compare(sample, facts, lambda family: (expected_of(family, facts, loc),))


def expected_variants(family: str, facts: dict[str, JsonValue]) -> tuple[str | bool | int, ...]:
    """输入：题族、事实。输出：该题族在**各支持语言**下可接受的答案（去重保序）。

    预期行为：choice 两族的答案是**选项描述原文**（按语言各写一份），所以各语言都收一份；
    score / noul 与语言无关，取一份即可。未知题族直接报错（与 `expected_of` 同一口径）。
    """
    if family in questions.CHOICE_FAMILIES:
        out: list[str | bool | int] = []
        for lang in locales.available():
            text = option_text(family, facts, locales.load(lang))
            if text not in out:
                out.append(text)
        return tuple(out)
    return (expected_of(family, facts, locales.load(locales.available()[0])),)


def check_sample_any_locale(
    sample: dict[str, JsonValue], facts: dict[str, JsonValue]
) -> list[str]:
    """输入：样本、从 state 重算出来的事实。输出：问题列表（空 = 全部对得上）。

    预期行为：与 `check_sample` 同一把尺，只是选项描述按**任一支持语言**的写法认——
    产物按 `locales[seq % len]` 轮转渲染，审计拿到的语言未必是产物那一种；只认一种语言的
    文案会把另一种语言的产物整片误判成「答案指向 …，按 state 重算是 …」。
    """
    return _compare(sample, facts, lambda family: expected_variants(family, facts))


def _show(want: tuple[str | bool | int, ...]) -> str:
    """输入：可接受的答案。输出：给人看的写法（多个候选用 ` / ` 连）。"""
    return " / ".join(str(value) for value in want)


def _compare(
    sample: dict[str, JsonValue],
    facts: dict[str, JsonValue],
    acceptable: Callable[[str], tuple[str | bool | int, ...]],
) -> list[str]:
    """输入：样本、重算出来的事实、按题族给可接受答案的函数。输出：问题列表（空 = 全对）。

    预期行为：逐题按**题族**重算再比对——choice 比选项描述原文、noul 比布尔、score 比档位。
    可接受的答案做成元组，是为了让审计一次给出「各语言的选项描述」（判别器只给一个）。
    """
    problems: list[str] = []
    for qid, question in sample["questions"].items():
        family = family_of(qid)
        try:
            want = acceptable(family)
        except (ValueError, KeyError) as exc:
            problems.append(f"{qid}: 重算不出答案（{exc}）")
            continue
        answer = sample["answers"][qid]
        if question["type"] == "choice":
            # 组装后的 `criteria` 是「选项名 → 选项描述」，`choice` 是**描述在其中的书写顺序**；
            # 判别器要比的是描述原文，所以取 `values()` 而不是 key。
            options = list(question["criteria"].values())
            index = answer["choice"]
            if not isinstance(index, int) or not 0 <= index < len(options):
                problems.append(f"{qid}: 答案编号越界（{index}）")
                continue
            if options[index] not in want:
                problems.append(f"{qid}: 答案指向 {options[index]}，按 state 重算是 {_show(want)}")
        elif question["type"] == "noul":
            if answer["noul"] not in want:
                problems.append(f"{qid}: 答案是 {answer['noul']}，按 state 重算是 {_show(want)}")
        elif question["type"] == "score":
            if answer["score"] not in want:
                problems.append(f"{qid}: 答案是 {answer['score']}，按 state 重算是 {_show(want)}")
        else:
            problems.append(f"{qid}: 未知题型 {question['type']!r}")
    return problems


def roundtrip_diff(
    spec: ReuseSpec, parsed: parse.ParsedReuse, loc: dict[str, JsonValue]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐步逐字一致）。"""
    problems: list[str] = []
    if parsed.request_id != spec.query_id:
        problems.append(f"请求号：渲染 {parsed.request_id!r} vs 期望 {spec.query_id!r}")
    want_flavor = loc["flavor"][spec.case.flavor]
    if parsed.flavor != want_flavor:
        problems.append(f"来源：渲染 {parsed.flavor!r} vs 期望 {want_flavor!r}")
    cached, current = spec_mod.traces_of(spec.case)
    spellings = (0, render.spelling_index(spec.case.spelling))
    for side, name in enumerate(("缓存", "当前")):
        steps = (cached, current)[side]
        got = parsed.side(side)
        if len(got) != len(steps):
            problems.append(f"{name}轨迹：渲染 {len(got)} 步 vs 期望 {len(steps)} 步")
            continue
        for index, (action, target) in enumerate(steps):
            want = render.step_text(loc, action, target, spellings[side])
            rendered = f"{got[index][0]}{loc['meta']['quote_open']}{got[index][1]}{loc['meta']['quote_close']}"
            if rendered != want:
                problems.append(f"{name}第 {index + 1} 步：渲染 {rendered!r} vs 期望 {want!r}")
    return problems


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：反解之后必须与真值规格逐步逐字一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, ReuseSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        try:
            parsed = parse.parse_state(sample["state"], parse.tables_of(loc))
        except ValueError as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        diff = roundtrip_diff(spec, parsed, loc)
        return Verdict(not diff, reason="; ".join(diff[:5]), raw={"diff": diff})


@dataclass(frozen=True)
class LabelJudge:
    """答案判别器：反解 state 后独立重算四个答案，逐题比对。"""

    name: str = "label"
    locale: str = "zh"
    max_problems: int = 5

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec（不用）。输出：Verdict。预期行为：答案对不上就判不通过。"""
        loc = locales.load(self.locale)
        try:
            parsed = parse.parse_state(sample["state"], parse.tables_of(loc))
            facts = facts_of(parsed, locales.words())
            problems = check_sample(sample, facts, loc)
        except (ValueError, KeyError) as exc:
            return Verdict(False, reason=f"重算失败：{exc}")
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )
