"""情感与情绪（sentiment）的判别器：反解 state 后独立重算四个答案。

输入：样本（state + 题目 + 答案）+ 真值规格。输出：`Verdict`。预期行为：
- `roundtrip`：反解出来的场景、词表、每个回合与规则段，必须与真值规格**逐项逐格一致**；
- `label`：只看 state，把每个回合的情绪词翻回档位之后独立重算四个答案再与产物比对——
  所以「生成器把答案算错了」与「渲染把某个词写歪了」会分别被抓到。

标签规则只有一份实现（在 `features` 里），构造侧与审计侧共用；判别器这边只负责把文本
翻成「分值序列」这种结构输入。
"""

from __future__ import annotations

from dataclasses import dataclass

from jc.judge import Verdict
from jc.jsontypes import JsonValue
from jc.workflow import compare_answers

from . import features, locales, parse, questions, render
from .spec import SentimentSpec


def facts_of(parsed: parse.ParsedArc) -> dict[str, JsonValue]:
    """输入：反解结果。输出：从对话重算出来的四个答案。

    预期行为：把分值序列交给 `features.observed_labels`——**标签规则只有一份实现**，
    构造侧与审计侧共用，免得两处各写一遍迟早漂移。
    """
    labels = features.observed_labels(parsed.levels)
    return {
        "tone": labels.tone,
        "pivot": labels.pivot_key,
        "drop": labels.drop,
        "rebound": labels.rebound,
    }


def option_text(family: str, facts: dict[str, JsonValue], loc: dict[str, JsonValue]) -> str:
    """输入：题族、事实、语言字典。输出：该题族金答案对应的**选项描述原文**。

    预期行为：用与 `questions` 同一张表拼出来再比——比「从描述里切出编号」稳，表改了也不会
    悄悄比错。两个 choice 族各有一张表。
    """
    if family == "tone":
        options = questions.tone_options(loc)
    elif family == "pivot":
        options = questions.pivot_options(loc)
    else:
        raise ValueError(f"未知题族：{family!r}（带选项的只有 {questions.CHOICE_FAMILIES}）")
    key = str(facts[family])
    if key not in options:
        raise ValueError(f"重算出来的选项名 {key!r} 不在选项表里：{sorted(options)}")
    return options[key]


def expected_of(
    family: str, facts: dict[str, JsonValue], loc: dict[str, JsonValue]
) -> str | bool | int:
    """输入：题族、事实、语言字典。输出：按 state 重算出来的答案。

    预期行为：choice 比选项描述原文、noul 比布尔、score 比档位下标；未知题族直接报错
    （新加了题族却忘了在这儿实现，不能静默放过）。
    """
    if family in questions.CHOICE_FAMILIES:
        return option_text(family, facts, loc)
    if family in questions.SCORE_FAMILIES:
        return int(facts[family])
    if family in questions.NOUL_FAMILIES:
        return bool(facts[family])
    raise ValueError(f"未知题族：{family!r}（可用：{questions.FAMILIES}）")


def expected_variants(family: str, facts: dict[str, JsonValue]) -> tuple[str | bool | int, ...]:
    """输入：题族、事实。输出：该题族在**各支持语言**下可接受的答案（去重保序）。

    预期行为：两个 choice 族的答案是**选项描述原文**（按语言各写一份），所以各语言都收一份；
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


def check_sample(
    sample: dict[str, JsonValue], facts: dict[str, JsonValue], loc: dict[str, JsonValue]
) -> list[str]:
    """输入：样本、从 state 重算出来的事实、语言字典。输出：问题列表（空 = 全部对得上）。

    预期行为：与 `check_sample_any_locale` 同一把尺，只是选项描述**只认这一种语言**的写法
    ——判别器判的是某一条确定的样本，它的语言是确定的，另一种语言的文案就是渲染写歪了。
    """
    return compare_answers(sample, lambda family: (expected_of(family, facts, loc),))


def check_sample_any_locale(
    sample: dict[str, JsonValue], facts: dict[str, JsonValue]
) -> list[str]:
    """输入：样本、从 state 重算出来的事实。输出：问题列表（空 = 全部对得上）。

    预期行为：与 `check_sample` 同一把尺，只是选项描述按**任一支持语言**的写法认——
    产物按 `locales[seq % len]` 轮转渲染，审计拿到的语言未必是产物那一种。
    """
    return compare_answers(sample, lambda family: expected_variants(family, facts))


def roundtrip_diff(
    spec: SentimentSpec, parsed: parse.ParsedArc, loc: dict[str, JsonValue]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐项逐格一致）。"""
    problems: list[str] = []
    if parsed.request_id != spec.query_id:
        problems.append(f"请求号：渲染 {parsed.request_id!r} vs 期望 {spec.query_id!r}")
    case = spec.case
    want_title = render.topic_text(loc, case.topic)[0]
    if parsed.topic != want_title:
        problems.append(f"场景：渲染 {parsed.topic!r} vs 期望 {want_title!r}")
    for level in features.LEVELS:
        want_words = tuple(str(word) for word in loc["word"][str(level)])
        if parsed.words.get(level) != want_words:
            problems.append(f"词表第 {level} 档：渲染 {parsed.words.get(level)} vs 期望 {want_words}")
    levels = case.levels
    if parsed.count != len(levels):
        problems.append(f"回合数：渲染 {parsed.count} vs 期望 {len(levels)}")
        return problems
    for index, (turn, want_level) in enumerate(zip(parsed.turns, levels, strict=True)):
        want_speaker = render.speaker_text(loc, case.topic, index)
        want_word = render.word_text(loc, want_level, features.word_slot(case.arc, index))
        if turn.speaker != want_speaker:
            problems.append(f"第 {index + 1} 回合的说话人：渲染 {turn.speaker!r} vs 期望 {want_speaker!r}")
        if turn.level != want_level:
            problems.append(f"第 {index + 1} 回合的档位：渲染 {turn.level} vs 期望 {want_level}（{want_word}）")
    rules = render.rules_lines(loc)
    want_rules = tuple(line[len(str(loc["meta"]["rules_prefix"])) :].strip() for line in rules)
    if parsed.rules != want_rules:
        problems.append(f"判读规则：渲染 {len(parsed.rules)} 行 vs 期望 {len(want_rules)} 行")
    return problems


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：反解之后必须与真值规格逐项逐格一致。"""

    locale: str = "zh"
    name: str = "roundtrip"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, SentimentSpec):
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

    locale: str = "zh"
    name: str = "label"
    max_problems: int = 5

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec（不用）。输出：Verdict。预期行为：答案对不上就判不通过。"""
        del spec
        loc = locales.load(self.locale)
        try:
            parsed = parse.parse_state(sample["state"], parse.tables_of(loc))
            facts = facts_of(parsed)
            problems = check_sample(sample, facts, loc)
        except (ValueError, KeyError) as exc:
            return Verdict(False, reason=f"重算失败：{exc}")
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )
