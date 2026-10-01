"""多批次排名聚合（siftrank）workflow 的判别器：反解 state 后独立重算四个答案。

输入：样本（state + 题目 + 答案）+ 真值规格。输出：`Verdict`。
预期行为：
- `roundtrip`：反解出来的四个批次，必须与真值规格**逐批逐条一致**（含提示与清单那两行）；
- `label`：只看 state，把条目名归一化成键之后独立重算四个答案再与产物比对——所以
  「生成器把答案算错了」与「渲染把某个批次写歪了」会分别被抓到。

归一化不复用渲染器的任何计算：条目名的写法表在语言文件里，反查实现在本文件的 `to_keys` 里，
两边靠 `tests/workflows/siftrank/test_locales.py` 与题面一致性测试锁住。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from dataclasses import dataclass

from jc.judge import Verdict
from jc.workflow import compare_answers, family_of

from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import RankSpec


def to_keys(names: tuple[str, ...], words: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：一串条目名、合并反查表。输出：条目键序列。

    预期行为：查不到就报人话——那说明渲染写出了表外的名字，不能静默当成「另一个条目」
    （那会把渲染 bug 伪装成一道难题）。
    """
    table = words["item"]
    if not isinstance(table, dict):
        raise ValueError("反查表形状不对（期望 {名字: 条目键}）")
    out: list[str] = []
    for name in names:
        if name not in table:
            raise ValueError(f"条目名不在写法表里：{name!r}")
        out.append(str(table[name]))
    return tuple(out)


def pair_of(sample: dict[str, JsonValue], words: dict[str, JsonValue]) -> tuple[str, str]:
    """输入：样本、合并反查表。输出：`pair_order` 那一族问的两个条目键。

    预期行为：从该题 `a_first` / `b_first` 两个选项描述里各找出**唯一**一个条目名——按语言
    文件写的描述是「{条目甲} 更相关」，所以它一定只含一个条目名。找不出唯一一个就报人话
    （那是构造或渲染 bug，不能猜）。
    """
    names = words["item"]
    if not isinstance(names, dict):
        raise ValueError("反查表形状不对（期望 {名字: 条目键}）")
    for qid, question in sample["questions"].items():
        if family_of(qid) != "pair_order":
            continue
        criteria = question["criteria"]
        if not isinstance(criteria, dict):
            raise ValueError(f"{qid}: 选项形状不对")
        found: list[str] = []
        for key in ("a_first", "b_first"):
            text = str(criteria.get(key, ""))
            hits = {str(value) for word, value in names.items() if word in text}
            if len(hits) != 1:
                raise ValueError(f"{qid}: 选项 {key} 里认出 {len(hits)} 个条目名（期望 1 个）")
            found.append(next(iter(hits)))
        if found[0] == found[1]:
            raise ValueError(f"{qid}: 两个选项指向同一个条目（{found[0]}）")
        return found[0], found[1]
    raise ValueError("样本里没有 pair_order 那一族，认不出被比较的两个条目")


def facts_of(
    parsed: parse.ParsedRank, words: dict[str, JsonValue], pair: tuple[str, str]
) -> dict[str, JsonValue]:
    """输入：反解结果、合并反查表、被比较的两个条目。输出：从文本重算出来的全部答案。

    预期行为：先把条目名归一化成键，再交给 `spec.observed_labels`——**聚合规则只有一份实现**
    （在 `spec` 里），构造侧与审计侧共用，免得两处各写一遍迟早漂移。
    """
    batches = tuple(to_keys(batch, words) for batch in parsed.batches)
    labels = spec_mod.observed_labels(batches)
    totals = labels["totals"]
    if not isinstance(totals, dict):
        raise ValueError("总分表形状不对（期望 {条目键: 总分}）")
    labels["relation"] = spec_mod.relation_of(
        {str(key): _as_int(value) for key, value in totals.items()}, pair[0], pair[1]
    )
    return labels


def option_text(
    family: str,
    facts: dict[str, JsonValue],
    loc: dict[str, JsonValue],
    pair: tuple[str, str],
) -> str:
    """输入：题族、事实、语言字典、被比较的两个条目。输出：金答案对应的**选项描述原文**。

    预期行为：用与 `questions` 同一个模板拼出来再比——比「从描述里切出条目名」稳，
    模板改了也不会悄悄比错。
    """
    if family == "agg_top":
        return str(loc["item"][str(facts["top"])])
    if family == "pair_order":
        names = {"a": str(loc["item"][pair[0]]), "b": str(loc["item"][pair[1]])}
        table = loc["fam"]["pair_order"]["criteria"]
        return str(table[str(facts["relation"])]).format(**names)
    raise ValueError(f"未知题族：{family!r}（选项只有 agg_top 与 pair_order 两族）")


def expected_of(
    family: str,
    facts: dict[str, JsonValue],
    loc: dict[str, JsonValue],
    pair: tuple[str, str],
) -> str | bool | int:
    """输入：题族、事实、语言字典、被比较的两个条目。输出：按 state 重算出来的答案。

    预期行为：choice 比选项描述原文、noul 比布尔、score 比等级下标；未知题族直接报错
    （新加了题族却忘了在这儿实现，不能静默放过）。
    """
    if family in questions.CHOICE_FAMILIES:
        return option_text(family, facts, loc, pair)
    if family == "cut_size":
        return spec_mod.CUTS.index(_as_int(facts["cut"]))
    if family == "stable":
        return bool(facts["stable"])
    raise ValueError(f"未知题族：{family!r}（可用：{questions.FAMILIES}）")


def expected_variants(
    family: str, facts: dict[str, JsonValue], pair: tuple[str, str]
) -> tuple[str | bool | int, ...]:
    """输入：题族、事实、被比较的两个条目。输出：该题族在**各支持语言**下可接受的答案。

    预期行为：choice 两族的答案是**选项描述原文**（按语言各写一份），所以各语言都收一份；
    score / noul 与语言无关，取一份即可。未知题族直接报错（与 `expected_of` 同一口径）。
    """
    if family in questions.CHOICE_FAMILIES:
        out: list[str | bool | int] = []
        for lang in locales.available():
            text = option_text(family, facts, locales.load(lang), pair)
            if text not in out:
                out.append(text)
        return tuple(out)
    return (expected_of(family, facts, locales.load(locales.available()[0]), pair),)


def check_sample(
    sample: dict[str, JsonValue],
    facts: dict[str, JsonValue],
    loc: dict[str, JsonValue],
    pair: tuple[str, str],
) -> list[str]:
    """输入：样本、从 state 重算出来的事实、语言字典、被比较的两个条目。输出：问题列表。

    预期行为：与 `check_sample_any_locale` 同一把尺，只是选项描述**只认这一种语言**的写法
    ——判别器判的就是「这条产物自己那个语言」。
    """
    return compare_answers(sample, lambda family: (expected_of(family, facts, loc, pair),))


def check_sample_any_locale(
    sample: dict[str, JsonValue], facts: dict[str, JsonValue], pair: tuple[str, str]
) -> list[str]:
    """输入：样本、从 state 重算出来的事实、被比较的两个条目。输出：问题列表。

    预期行为：与 `check_sample` 同一把尺，只是选项描述按**任一支持语言**的写法认——产物按
    `locales[seq % len]` 轮转渲染，审计拿到的语言未必是产物那一种；只认一种语言的文案会把
    另一种语言的产物整片误判成「答案指向 …，按 state 重算是 …」。
    """
    return compare_answers(sample, lambda family: expected_variants(family, facts, pair))


def _as_int(value: JsonValue) -> int:
    """输入：一个 JSON 值。输出：整数。不是整数（含布尔）就报人话。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"期望整数，实际 {value!r}")
    return value


def roundtrip_diff(
    spec: RankSpec, parsed: parse.ParsedRank, loc: dict[str, JsonValue]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐批逐条一致）。"""
    problems: list[str] = []
    if parsed.request_id != spec.request_id:
        problems.append(f"请求号：渲染 {parsed.request_id!r} vs 期望 {spec.request_id!r}")
    want_prompt = str(loc["flavor"][spec.case.flavor])
    if parsed.prompt != want_prompt:
        problems.append(f"排序提示：渲染 {parsed.prompt!r} vs 期望 {want_prompt!r}")
    want_listing = tuple(render.item_text(loc, key) for key in spec_mod.ITEMS)
    if parsed.listing != want_listing:
        problems.append(f"条目清单：渲染 {list(parsed.listing)} vs 期望 {list(want_listing)}")
    if len(parsed.batches) != len(spec.batches):
        problems.append(f"批次数：渲染 {len(parsed.batches)} vs 期望 {len(spec.batches)}")
    else:
        for index, batch in enumerate(spec.batches, start=1):
            want = tuple(render.item_text(loc, key) for key in batch)
            if parsed.batches[index - 1] != want:
                problems.append(
                    f"批次 {index}：渲染 {list(parsed.batches[index - 1])} vs 期望 {list(want)}"
                )
    return problems


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：反解之后必须与真值规格逐批逐条一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, RankSpec):
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
            pair = pair_of(sample, locales.words())
            facts = facts_of(parsed, locales.words(), pair)
            problems = check_sample(sample, facts, loc, pair)
        except (ValueError, KeyError) as exc:
            return Verdict(False, reason=f"重算失败：{exc}")
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )
