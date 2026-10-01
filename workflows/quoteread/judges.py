"""行情读数判读（quoteread）的判别器：反解 state 后独立重算四个答案。

输入：样本（state + 题目 + 答案）+ 真值规格。输出：`Verdict`。预期行为：
- `roundtrip`：反解出来的读数表，必须与真值规格**逐行逐字一致**（含抬头那一行）；
- `label`：只看 state，把每行的原文归一化成名键与基准点之后独立重算四个答案再与产物比对
  ——所以「生成器把答案算错了」与「渲染把某一行写歪了」会分别被抓到。

归一化不复用渲染器的任何计算：标的名表与单位写法在语言文件里，读数→基准点的换算实现在
本文件的 `parse_bp` 里，两边靠 `tests/workflows/quoteread/test_locales.py` 与题面一致性
测试锁住。
"""

from __future__ import annotations

from dataclasses import dataclass

from jc.judge import Verdict
from jc.jsontypes import JsonValue
from jc.workflow import compare_answers

from . import features, locales, parse, questions, render
from .features import BP_PER_PCT, UNITS
from .spec import QuoteSpec


def split_value(text: str, units: dict[str, JsonValue]) -> tuple[str, str]:
    """输入：读数原文（如 `-1.50%`）、单位反查表。输出：(数字原文, 单位键)。

    预期行为：**最长后缀优先**——单位写法之间有前缀关系时才不会把 `bp` 认成 `b`；
    数字部分不许是空串（`%` 单独一个词不是读数），认不出就报人话。
    """
    for word in sorted(units, key=len, reverse=True):
        if text.endswith(str(word)) and len(text) > len(str(word)):
            return text[: -len(str(word))].strip(), str(units[word])
    raise ValueError(f"读数认不出单位：{text!r}（表里的写法：{sorted(units)}）")


def parse_bp(text: str, unit_key: str) -> int:
    """输入：数字原文（如 `-0.45`）、单位键。输出：基准点数。

    预期行为：百分比按 `1% = 100 基准点` 展开成**整数**（`0.45` → 45），全程整数算术，
    不经过浮点——`float("-0.45") * 100` 会得到 `-45.000000000000006`。写法不合规就报人话，
    不许静默取近似值（那会把渲染 bug 伪装成一道难题）。
    """
    negative = text.startswith("-")
    body = text.lstrip("+-")
    if unit_key == "pct":
        whole, _, frac = body.partition(".")
        if not whole.isdigit() or len(frac) != 2 or not frac.isdigit():
            raise ValueError(f"百分比读数的写法不对：{text!r}（期望两位小数）")
        bp = int(whole) * BP_PER_PCT + int(frac)
    elif unit_key == "bp":
        if not body.isdigit():
            raise ValueError(f"基准点读数的写法不对：{text!r}（期望整数）")
        bp = int(body)
    else:
        raise ValueError(f"未知单位键：{unit_key!r}（可用：{UNITS}）")
    return -bp if negative else bp


def to_readings(
    rows: tuple[tuple[str, str], ...], words: dict[str, JsonValue]
) -> tuple[tuple[str, int], ...]:
    """输入：反解出来的 (标的名原文, 读数原文) 序列、合并反查表。输出：(名键, 基准点) 序列。

    预期行为：标的名与单位各自查表；**查不到就报人话**——那说明渲染写出了表外的写法，
    不能静默当成「读数为零」（那会把渲染 bug 伪装成一道难题）。
    """
    names = words["name"]
    units = words["unit"]
    if not isinstance(names, dict) or not isinstance(units, dict):
        raise ValueError("反查表形状不对（期望 {原文: 键}）")
    out: list[tuple[str, int]] = []
    for name_text, value_text in rows:
        if name_text not in names:
            raise ValueError(f"标的名不在名字表里：{name_text!r}")
        number, unit_key = split_value(value_text, units)
        out.append((str(names[name_text]), parse_bp(number, unit_key)))
    return tuple(out)


def facts_of(parsed: parse.ParsedQuote, words: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """输入：反解结果、合并反查表。输出：从文本重算出来的四个答案（外加两条素材）。

    预期行为：先把原文归一化成名键与基准点，再交给 `features.observed_labels`——**标签规则
    只有一份实现**（在 `features` 里），构造侧与审计侧共用，免得两处各写一遍迟早漂移。
    """
    readings = to_readings(parsed.rows, words)
    names = [name for name, _ in readings]
    values = tuple(bp for _, bp in readings)
    labels = features.observed_labels(values)
    return {
        "names": list(names),
        "values": list(values),
        "tier": labels.tier,
        "leader": names[labels.leader],
        "alert": labels.alert,
        "divergent": labels.divergent,
    }


def option_text(family: str, facts: dict[str, JsonValue], loc: dict[str, JsonValue]) -> str:
    """输入：题族、事实、语言字典。输出：该题族金答案对应的**选项描述原文**。

    预期行为：用与 `questions.options_of` 同一张表拼出来再比——比「从描述里切出名键」稳，
    表改了也不会悄悄比错。
    """
    if family == "leader":
        table = loc["name"]
        if not isinstance(table, dict):
            raise ValueError("语言表的 [name] 必须是对象")
        return str(table[str(facts["leader"])])
    raise ValueError(f"未知题族：{family!r}（选项只有 leader 一族）")


def expected_of(
    family: str, facts: dict[str, JsonValue], loc: dict[str, JsonValue]
) -> str | bool | int:
    """输入：题族、事实、语言字典。输出：按 state 重算出来的答案。

    预期行为：choice 比选项描述原文、noul 比布尔、score 比档位下标；未知题族直接报错
    （新加了题族却忘了在这儿实现，不能静默放过）。
    """
    if family in questions.CHOICE_FAMILIES:
        return option_text(family, facts, loc)
    if family == "tier":
        return int(facts["tier"])
    if family in questions.NOUL_FAMILIES:
        return bool(facts[family])
    raise ValueError(f"未知题族：{family!r}（可用：{questions.FAMILIES}）")


def expected_variants(family: str, facts: dict[str, JsonValue]) -> tuple[str | bool | int, ...]:
    """输入：题族、事实。输出：该题族在**各支持语言**下可接受的答案（去重保序）。

    预期行为：choice 那一族的答案是**选项描述原文**（按语言各写一份），所以各语言都收一份；
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
    做成模块级函数是为了让 `audit` 也能直接复用同一套判据（判别器与审计必须用同一把尺）。
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
    spec: QuoteSpec, parsed: parse.ParsedQuote, loc: dict[str, JsonValue]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐行逐字一致）。"""
    problems: list[str] = []
    if parsed.request_id != spec.query_id:
        problems.append(f"请求号：渲染 {parsed.request_id!r} vs 期望 {spec.query_id!r}")
    case = spec.case
    triples = tuple(zip(case.name_ids, case.bp_values, case.unit_seq, strict=True))
    if parsed.count != len(triples):
        problems.append(f"读数行数：渲染 {parsed.count} vs 期望 {len(triples)}")
        return problems
    for index, (name_id, bp, unit_key) in enumerate(triples, start=1):
        want_name = render.name_text(loc, name_id)
        want_value = render.format_value(loc, bp, unit_key)
        got_name, got_value = parsed.rows[index - 1]
        if got_name != want_name or got_value != want_value:
            problems.append(
                f"第 {index} 行：渲染 {got_name!r} / {got_value!r} vs 期望 {want_name!r} / {want_value!r}"
            )
    return problems


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：反解之后必须与真值规格逐行逐字一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, QuoteSpec):
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
        del spec
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
