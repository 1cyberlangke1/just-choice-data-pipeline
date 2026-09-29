"""记录匹配（match）workflow 的判别器：反解 state 后独立重算八个答案。

输入：样本（state + 题目 + 答案）+ 真值规格。输出：`Verdict`。
预期行为：
- `roundtrip`：反解出来的两条记录，必须与真值规格**逐格一致**（含额外栏写没写）；
- `label`：只看 state，独立重算八个答案再与产物比对——所以「生成器把答案算错了」与
  「渲染把某一栏写歪了」会分别被抓到。

归一化不复用渲染器的任何计算：规则文本写在语言文件里，实现写在本文件的 `normalize` 里，
两边靠 `tests/workflows/match/test_locales.py` 与题面一致性测试锁住。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from jc.judge import Verdict
from jc.workflow import family_of

from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import MatchSpec

#: 规格里的「数字 + 单位」（`750 ml` / `0.75 L` / `750ml`）。
SIZE_TOKEN: Final[re.Pattern[str]] = re.compile(r"(\d+(?:\.\d+)?)\s*(ml|l)\b", re.IGNORECASE)
#: 度数里的数字（`5.0% vol` / `5.0 度` / `5%vol`）。
NUMBER: Final[re.Pattern[str]] = re.compile(r"\d+(?:\.\d+)?")


def normalize(field: str, text: str, loc: dict[str, JsonValue]) -> str:
    """输入：栏名、原文、语言字典。输出：归一化后的值（字符串，可直接比）。

    预期行为：判据里那四条规则的**唯一实现**——名称 / 品牌忽略大小写与连续空格；
    规格换算成 ml 的数值；度数取数值；包装查同义词表（表就是语言文件里的 `[pack]`）。
    读不出数值 / 查不到同义词都报人话：那说明渲染写出了表外的写法。
    """
    if field in ("name", "brand"):
        return " ".join(text.split()).casefold()
    if field == "size":
        found = SIZE_TOKEN.search(text)
        if found is None:
            raise ValueError(f"规格读不出「数值 + 单位」：{text!r}")
        value = float(found.group(1))
        if found.group(2).lower() == "l":
            value *= 1000
        return f"{value:g}"
    if field == "abv":
        found = NUMBER.search(text)
        if found is None:
            raise ValueError(f"度数读不出数值：{text!r}")
        return f"{float(found.group(0)):g}"
    table = locales.reverse(loc["pack"])
    if text not in table:
        raise ValueError(f"包装不在同义词表里：{text!r}")
    return table[text]


def kinds_of(left: tuple[str, ...], right: tuple[str, ...], loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：两侧每栏的原文、语言字典。输出：每一栏的关系。

    预期行为：**先看原文是否逐字相同**（相同就是 `same`），再看归一化后是否相同
    （相同就是 `variant`），都不相同才是 `diff`——顺序反了会把 `same` 读成 `variant`。
    """
    out: list[str] = []
    for field, first, second in zip(spec_mod.FIELDS, left, right):
        if first == second:
            out.append(spec_mod.SAME)
        elif normalize(field, first, loc) == normalize(field, second, loc):
            out.append(spec_mod.VARIANT)
        else:
            out.append(spec_mod.DIFF)
    return tuple(out)


@dataclass(frozen=True)
class Facts:
    """从 state 反解并重算出来的事实：两侧原文、每一栏的关系、四个派生量。"""

    left: tuple[str, ...]
    right: tuple[str, ...]
    kinds: tuple[str, ...]
    agree: int
    band: int
    verdict: str
    variant_count: int

    def same(self, field: str) -> bool:
        """输入：栏名。输出：这一栏是不是一致（`diff` 之外都算一致）。"""
        return self.kinds[spec_mod.FIELDS.index(field)] != spec_mod.DIFF


def recompute(parsed: parse.ParsedMatch, loc: dict[str, JsonValue]) -> Facts:
    """输入：反解结果、语言字典。输出：`Facts`。

    预期行为：只从文本本身算——逐栏归一化、比、数个数；四个派生量走 `spec` 里与构造侧
    **同一套规则**（判别器重算的是「读出来的两条记录」，不是复用生成时算好的答案）。
    """
    left = tuple(parsed.side(0)[field] for field in spec_mod.FIELDS)
    right = tuple(parsed.side(1)[field] for field in spec_mod.FIELDS)
    kinds = kinds_of(left, right, loc)
    agree = spec_mod.agree_of(kinds)
    return Facts(
        left=left,
        right=right,
        kinds=kinds,
        agree=agree,
        band=spec_mod.band_of(agree),
        verdict=spec_mod.verdict_of(kinds),
        variant_count=spec_mod.variant_count_of(kinds),
    )


def option_text(family: str, facts: Facts, loc: dict[str, JsonValue]) -> str:
    """输入：题族、事实、语言字典。输出：该题族金答案对应的**选项描述原文**。

    预期行为：用与 `questions.options_of` 同一个模板拼出来再比——比「从描述里切出编号」
    稳，模板改了也不会悄悄比错。
    """
    if family == "verdict":
        return loc["fam"]["verdict"]["criteria"][facts.verdict]
    return loc["fam"]["variant_count"]["criteria"][f"v{facts.variant_count}"]


def expected_of(family: str, facts: Facts, loc: dict[str, JsonValue]) -> str | bool | int:
    """输入：题族、事实、语言字典。输出：按 state 重算出来的答案。

    预期行为：choice 比选项描述原文、noul 比布尔、score 比档位下标；未知题族直接报错
    （新加了题族却忘了在这儿实现，不能静默放过）。
    """
    if family in questions.CHOICE_FAMILIES:
        return option_text(family, facts, loc)
    if family == "band":
        return facts.band
    if family in questions.NOUL_FIELDS:
        return facts.same(questions.NOUL_FIELDS[family])
    raise ValueError(f"未知题族：{family!r}")


def check_sample(sample: dict[str, JsonValue], facts: Facts, loc: dict[str, JsonValue]) -> list[str]:
    """输入：样本、从 state 重算出来的事实、语言字典。输出：问题列表（空 = 全部对得上）。

    预期行为：与 `check_sample_any_locale` 同一把尺，只是选项描述**只认这一种语言**的写法
    ——判别器判的是某一条确定的样本，它的语言是确定的，另一种语言的文案就是渲染写歪了。
    做成模块级函数是为了让 `audit` 也能直接复用同一套判据（判别器与审计必须用同一把尺）。
    """
    return _compare(sample, facts, lambda family: (expected_of(family, facts, loc),))


def expected_variants(family: str, facts: Facts) -> tuple[str | bool | int, ...]:
    """输入：题族、事实。输出：该题族在**各支持语言**下可接受的答案（去重保序）。

    预期行为：choice 两族的答案是**选项描述原文**（按语言各写一份），所以各语言都收一份；
    band / noul 与语言无关，取一份即可。未知题族直接报错（与 `expected_of` 同一口径）。
    """
    if family in questions.CHOICE_FAMILIES:
        out: list[str | bool | int] = []
        for lang in locales.available():
            text = option_text(family, facts, locales.load(lang))
            if text not in out:
                out.append(text)
        return tuple(out)
    return (expected_of(family, facts, locales.load(locales.available()[0])),)


def check_sample_any_locale(sample: dict[str, JsonValue], facts: Facts) -> list[str]:
    """输入：样本、从 state 重算出来的事实。输出：问题列表（空 = 全部对得上）。

    预期行为：与 `check_sample` 同一把尺，只是选项描述按**任一支持语言**的写法认——
    产物按 `locales[seq % len]` 轮转渲染，审计拿到的语言未必是产物那一种；只认一种语言
    的文案会把另一种语言的产物整片误判成「答案指向 …，按 state 重算是 …」。
    """
    return _compare(sample, facts, lambda family: expected_variants(family, facts))


def _show(want: tuple[str | bool | int, ...]) -> str:
    """输入：可接受的答案。输出：给人看的写法（多个候选用 ` / ` 连）。"""
    return " / ".join(str(value) for value in want)


def _compare(
    sample: dict[str, JsonValue],
    facts: Facts,
    acceptable: Callable[[str], tuple[str | bool | int, ...]],
) -> list[str]:
    """输入：样本、重算出来的事实、按题族给可接受答案的函数。输出：问题列表（空 = 全部对得上）。

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


def roundtrip_diff(spec: MatchSpec, parsed: parse.ParsedMatch, loc: dict[str, JsonValue]) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐格一致）。"""
    problems: list[str] = []
    if parsed.request_id != spec.query_id:
        problems.append(f"请求号：渲染 {parsed.request_id!r} vs 期望 {spec.query_id!r}")
    left, right = render.cells_of(spec.case, spec_mod.diff_pool(spec.split))
    for side, name in enumerate(("A", "B")):
        values = (left, right)[side]
        got = parsed.side(side)
        for index, field in enumerate(spec_mod.FIELDS):
            spelling = render.spelling_of(spec.case, side, spec.case.pattern[index])
            want = render.cell(field, values[index], spelling, loc)
            if got.get(field) != want:
                problems.append(f"{name}/{field}：渲染 {got.get(field)!r} vs 期望 {want!r}")
    if spec.case.extra == "origin":
        for side, name in enumerate(("A", "B")):
            want = loc["extra"][("origin_left", "origin_right")[side]]
            got = parsed.extra_of(side).get("origin")
            if got != want:
                problems.append(f"{name}/产地：渲染 {got!r} vs 期望 {want!r}")
    else:
        written = [parsed.extra_of(side).get("origin") for side in (0, 1)]
        if any(value is not None for value in written):
            problems.append(f"产地：渲染写了 {written}，但这条样本不该有额外栏")
    return problems


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：反解之后必须与真值规格逐格一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, MatchSpec):
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
    """答案判别器：反解 state 后独立重算八个答案，逐题比对。"""

    name: str = "label"
    locale: str = "zh"
    max_problems: int = 5

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec（不用）。输出：Verdict。预期行为：答案对不上就判不通过。"""
        loc = locales.load(self.locale)
        try:
            parsed = parse.parse_state(sample["state"], parse.tables_of(loc))
            facts = recompute(parsed, loc)
            problems = check_sample(sample, facts, loc)
        except (ValueError, KeyError) as exc:
            return Verdict(False, reason=f"重算失败：{exc}")
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )
