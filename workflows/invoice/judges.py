"""单据 workflow 的判别器：构造式数据用纯规则判别即可（**零 token**）。

- `RoundtripJudge`：把渲染文本反解回字段与明细，与真值 spec **逐字**比对
  （抓渲染错位、字段漏写、数值写错）；
- `LabelJudge`：反解文本后**独立重算**每道题的金答案，与数据里的答案比对
  （抓答案算错、选项与单据对不上）。它只看渲染出来的文本 + 题面，不看 spec 里的答案，
  否则就是「拿答案验证答案」。

输入/输出：只出 `Verdict`，不改样本。对不上就判不通过——宁可少要几条，
也不要一条「题面和单据对不上」的数据。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from dataclasses import dataclass


from jc.judge import Verdict

from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import InvoiceSpec


def roundtrip_diff(spec: InvoiceSpec, parsed: parse.ParsedInvoice, loc: dict[str, JsonValue]) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：差异列表（空 = 一致）。

    预期行为：字段**逐字**比（连货币符号、千分位、日期写法都算），明细逐格比；
    单据里多出别的字段也算差异（说明渲染多写了东西）。
    """
    diff: list[str] = []
    expected = render.expected_fields(spec, loc)
    for slot, want in expected.items():
        got = parsed.fields.get(slot)
        if got != want:
            diff.append(f"字段 {slot}：渲染 {got!r} vs 期望 {want!r}")
    for slot in parsed.fields:
        if slot not in expected:
            diff.append(f"多出字段 {slot}（这份单据不该有）")
    want_items = render.expected_items(spec, loc)
    got_items = [(item.name, item.qty, item.unit, item.amount) for item in parsed.items]
    if got_items != want_items:
        diff.append(f"明细不一致：渲染 {got_items} vs 期望 {want_items}")
    return diff


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：渲染文本反解后必须与真值 spec 完全一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, InvoiceSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        parsed = parse.parse_document(sample["state"], loc)
        diff = roundtrip_diff(spec, parsed, loc)
        return Verdict(not diff, reason="; ".join(diff[:5]), raw={"diff": diff})


def _recompute_bool(family: str, parsed: parse.ParsedInvoice) -> bool | None:
    """输入：noul 题族名、反解结果。输出：按文本重算的答案，或 None（字段解不出来）。"""
    if family == "subtotal_consistent":
        subtotal = parsed.money("subtotal")
        if subtotal is None or not parsed.items:
            return None
        return subtotal == sum(item.amount for item in parsed.items)
    if family == "tax_consistent":
        subtotal = parsed.money("subtotal")
        tax = parsed.money("tax")
        rate = parsed.rate()
        if subtotal is None or tax is None or rate is None:
            return None
        return tax == spec_mod.percent_of(subtotal, rate)
    return None


def _recompute_bad_lines(parsed: parse.ParsedInvoice) -> int:
    """输入：反解结果。输出：几行的「数量 × 单价」与金额对不上，封顶 3。"""
    return min(3, sum(1 for item in parsed.items if item.amount != item.qty * item.unit))


def _value_problems(qid: str, slot: str, description: object, parsed: parse.ParsedInvoice) -> list[str]:
    """输入：题号、目标槽、选项描述、反解结果。输出：问题列表（空 = 这个选项与单据对得上）。

    预期行为：**按类归一比**——日期比日期三元组、金额与单价比整数、其余比字符串。
    这样只抓「值写错了」，不会因为货币符号或日期写法不同而误判。
    """
    if slot.startswith("line_"):
        index = int(slot.split("_")[1]) - 1
        if not 0 <= index < len(parsed.items):
            return [f"{qid}: 选项 {slot} 超出明细行数（{len(parsed.items)} 行）"]
        want = parse.as_int(str(description))
        got = parsed.items[index].unit
        if want != got:
            return [f"{qid}: 选项 {slot} 写的是 {description!r}，单据里是 {got}"]
        return []
    raw = parsed.text(slot)
    if slot in spec_mod.DATE_SLOTS:
        left, right = parse.as_date(raw), parse.as_date(str(description))
    elif slot in spec_mod.MONEY_SLOTS:
        left, right = parse.as_int(raw), parse.as_int(str(description))
    else:
        left, right = raw, str(description)
    if left is None or right is None:
        return [f"{qid}: 选项 {slot} 或单据里的值解不出来（单据 {raw!r} / 选项 {description!r}）"]
    if left != right:
        return [f"{qid}: 选项 {slot} 写的是 {description!r}，单据里是 {raw!r}"]
    return []


@dataclass(frozen=True)
class LabelJudge:
    """答案判别器：反解文本后独立重算金答案，逐题比对。

    除了「答案指向的选项」，choice 的**每个**选项描述都要与单据里对应字段的值对得上——
    只验答案那个选项会漏掉「干扰项描述写错」，那会让题目本身有歧义（两个选项都说得通）。
    """

    name: str = "label"
    locale: str = "zh"
    max_problems: int = 5

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec（不用）。输出：Verdict。预期行为：答案对不上就判不通过。"""
        loc = locales.load(self.locale)
        parsed = parse.parse_document(sample["state"], loc)
        problems = self._check(sample, parsed, loc)
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )

    def _check(
        self, sample: dict[str, JsonValue], parsed: parse.ParsedInvoice, loc: dict[str, JsonValue]
    ) -> list[str]:
        """输入：样本、反解结果、语言字典。输出：问题列表（空 = 全部对得上）。"""
        problems: list[str] = []
        for qid, question in sample["questions"].items():
            found = questions.locate(question["instructions"], loc, len(parsed.items))
            if found is None:
                problems.append(f"{qid}: 认不出题面（{question['instructions']!r}）")
                continue
            family, target = found
            answer = sample["answers"][qid]
            if question["type"] == "choice":
                options = list(question["criteria"])
                index = answer["choice"]
                if not 0 <= index < len(options):
                    problems.append(f"{qid}: 答案编号越界（{index}）")
                    continue
                chosen = options[index]
                if chosen != target:
                    problems.append(f"{qid}: 答案指向 {chosen}，但题面问的是 {target}")
                for name in options:
                    problems.extend(
                        _value_problems(qid, name, question["criteria"][name], parsed)
                    )
            elif question["type"] == "noul":
                want = _recompute_bool(family, parsed)
                if want is None:
                    problems.append(f"{qid}: 单据里解不出判据需要的字段")
                elif answer["noul"] is not want:
                    problems.append(f"{qid}: 答案是 {answer['noul']}，按文本重算是 {want}")
            else:
                want = _recompute_bad_lines(parsed)
                if answer["score"] != want:
                    problems.append(f"{qid}: 答案是 {answer['score']}，按文本重算是 {want}")
        return problems
