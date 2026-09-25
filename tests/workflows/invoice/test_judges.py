"""判别器的测试：构造式数据的两个规则判别器（零 token）必须真的会拦错。"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import pytest

from jc import assemble
from workflows.invoice import judges as judges_mod, locales, parse, questions, render, spec as sp

LOCALES = locales.available()


def _sample(
    make_spec: Any, seed: int = 5, lang: str = "zh", **overrides: Any
) -> tuple[dict[str, Any], sp.InvoiceSpec]:
    """输入：规格工厂、种子、语言、case 覆盖项。输出：(组装好的样本, 真值规格)。"""
    loc = locales.load(lang)
    spec = make_spec(seed, **overrides)
    rng = random.Random(seed)
    labeled = questions.build(spec, loc, spec.phrasing, rng)
    sample = assemble.assemble_sample(
        "invoice-train-0000000", render.build(spec, loc), labeled, rng
    )
    return sample, spec


def _first_choice(sample: dict[str, Any]) -> str:
    """输入：样本。输出：第一道 choice 的题号。"""
    return next(qid for qid, item in sample["questions"].items() if item["type"] == "choice")


@pytest.mark.parametrize("lang", LOCALES)
def test_roundtrip_judge_passes_a_correct_sample(lang: str, make_spec: Any) -> None:
    """输入：正确渲染的样本 + 真值。输出：Verdict。

    预期行为：通过、差异列表为空（字段逐字比、明细逐格比）。
    """
    sample, spec = _sample(make_spec, 141, lang, lines=3, defects=1)
    verdict = judges_mod.RoundtripJudge(locale=lang).judge(sample, spec)
    assert verdict.ok is True
    assert verdict.raw["diff"] == []
    parsed = parse.parse_document(sample["state"], locales.load(lang))
    assert judges_mod.roundtrip_diff(spec, parsed, locales.load(lang)) == []


def test_roundtrip_judge_skips_without_a_spec(make_spec: Any) -> None:
    """输入：样本但没有真值规格。输出：Verdict。

    预期行为：跳过（不误判），并在 raw 里标注——宁可少拒一条，也不能凭空判错。
    """
    sample, _ = _sample(make_spec, 151)
    verdict = judges_mod.RoundtripJudge().judge(sample, None)
    assert verdict.ok is True and verdict.raw["skipped"] is True


@pytest.mark.parametrize("lang", LOCALES)
def test_roundtrip_judge_catches_a_different_document(lang: str, make_spec: Any) -> None:
    """输入：另一份单据的文本 + 本份的真值。输出：Verdict。

    预期行为：判不通过——渲染出来的东西必须与真值逐字对得上。
    """
    sample, spec = _sample(make_spec, 161, lang)
    other, _ = _sample(make_spec, 999, lang)
    verdict = judges_mod.RoundtripJudge(locale=lang).judge(
        {"state": other["state"]}, spec
    )
    assert verdict.ok is False
    assert verdict.raw["diff"]


def test_roundtrip_judge_names_the_field_that_disagrees(make_spec: Any) -> None:
    """输入：真值里的「总计」被改掉（文本没变）。输出：Verdict。

    预期行为：判不通过，理由里点名 `total`——判别器拿文本与真值对，方向无所谓。
    """
    sample, spec = _sample(make_spec, 171)
    bad = dataclasses.replace(spec, total=spec.total + 1)
    verdict = judges_mod.RoundtripJudge().judge(sample, bad)
    assert verdict.ok is False
    assert any("total" in item for item in verdict.raw["diff"])


def test_roundtrip_diff_flags_an_extra_field(make_spec: Any) -> None:
    """输入：反解结果里多出一个字段。输出：差异列表。

    预期行为：多写字段也算差异（说明渲染写了这份单据不该有的东西）。
    """
    loc = locales.load("zh")
    spec = make_spec(181, tax_state="none")
    parsed = parse.parse_document(render.build(spec, loc), loc)
    extra = parse.ParsedInvoice(fields={**parsed.fields, "tax": "¥1"}, items=parsed.items)
    diff = judges_mod.roundtrip_diff(spec, extra, loc)
    assert any("多出字段 tax" in item for item in diff)


@pytest.mark.parametrize("lang", LOCALES)
def test_label_judge_passes_a_correct_sample(lang: str, make_spec: Any) -> None:
    """输入：正确组装的样本。输出：Verdict。

    预期行为：通过、零问题——它只看渲染文本与题面，不看真值里的答案。
    """
    sample, _ = _sample(make_spec, 191, lang, lines=4, defects=2)
    verdict = judges_mod.LabelJudge(locale=lang).judge(sample)
    assert verdict.ok is True and verdict.raw["problems"] == []


def test_label_judge_catches_a_wrong_choice_index(make_spec: Any) -> None:
    """输入：把 choice 的答案编号改成另一个选项。输出：Verdict。

    预期行为：判不通过，并点出「答案指向的选项不是题面问的那个」。
    """
    sample, _ = _sample(make_spec, 201)
    qid = _first_choice(sample)
    size = len(sample["questions"][qid]["criteria"])
    sample["answers"][qid]["choice"] = (sample["answers"][qid]["choice"] + 1) % size
    verdict = judges_mod.LabelJudge().judge(sample)
    assert verdict.ok is False
    assert any("题面问的是" in item for item in verdict.raw["problems"])


def test_label_judge_catches_swapped_option_descriptions(make_spec: Any) -> None:
    """输入：把两个选项的描述对调。输出：Verdict。

    预期行为：判不通过——每个选项的描述都要与单据里对应字段的值一致；
    只验「答案那个选项」会漏掉写错的干扰项，那会让题目本身有歧义。
    """
    sample, _ = _sample(make_spec, 211)
    options = sample["questions"]["date_field"]["criteria"]
    names = list(options)
    options[names[0]], options[names[1]] = options[names[1]], options[names[0]]
    verdict = judges_mod.LabelJudge().judge(sample)
    assert verdict.ok is False
    assert any(names[0] in item or names[1] in item for item in verdict.raw["problems"])


def test_label_judge_catches_an_out_of_range_index(make_spec: Any) -> None:
    """输入：答案编号 99。输出：Verdict。预期行为：判不通过并明说越界。"""
    sample, _ = _sample(make_spec, 221)
    qid = _first_choice(sample)
    sample["answers"][qid]["choice"] = 99
    problems = judges_mod.LabelJudge().judge(sample).raw["problems"]
    assert any("答案编号越界" in item for item in problems)


def test_label_judge_catches_a_flipped_noul_answer(make_spec: Any) -> None:
    """输入：把 noul 答案取反。输出：Verdict。

    预期行为：判不通过——答案必须与按文本重算的结果一致。
    """
    sample, _ = _sample(make_spec, 231)
    qid = next(
        qid for qid, item in sample["questions"].items() if item["type"] == "noul"
    )
    sample["answers"][qid]["noul"] = not sample["answers"][qid]["noul"]
    problems = judges_mod.LabelJudge().judge(sample).raw["problems"]
    assert any("按文本重算是" in item for item in problems)


def test_label_judge_catches_a_wrong_score(make_spec: Any) -> None:
    """输入：把 score 答案改成别的档位。输出：Verdict。预期行为：判不通过。"""
    sample, _ = _sample(make_spec, 241, lines=4, defects=2)
    sample["answers"]["bad_lines"]["score"] = 3
    problems = judges_mod.LabelJudge().judge(sample).raw["problems"]
    assert any("按文本重算是" in item for item in problems)


def test_label_judge_catches_an_unknown_question(make_spec: Any) -> None:
    """输入：一句本 workflow 认不出的题面。输出：Verdict。

    预期行为：判不通过并说「认不出题面」——认不出来时不能默认通过。
    """
    sample, _ = _sample(make_spec, 251)
    sample["questions"][_first_choice(sample)]["instructions"] = "今天天气怎么样？"
    problems = judges_mod.LabelJudge().judge(sample).raw["problems"]
    assert any("认不出题面" in item for item in problems)


def test_label_judge_complains_when_a_needed_field_is_missing(make_spec: Any) -> None:
    """输入：把单据里的「税率」那一行删掉。输出：Verdict。

    预期行为：判不通过并说解不出判据字段——缺字段不能当成「算出来是 False」。
    """
    sample, spec = _sample(make_spec, 261, tax_state="ok")
    label = locales.label_of(locales.load("zh"), "tax_rate", spec.case.labels)
    sample["state"] = "\n".join(
        line for line in sample["state"].split("\n") if not line.startswith(label)
    )
    problems = judges_mod.LabelJudge().judge(sample).raw["problems"]
    assert any("解不出判据需要的字段" in item for item in problems)


def test_label_judge_catches_a_wrong_value_in_an_option(make_spec: Any) -> None:
    """输入：把某个选项的描述换成一个对不上的金额。输出：Verdict。

    预期行为：判不通过——选项描述必须与单据里那个字段的值对得上。
    """
    sample, _ = _sample(make_spec, 271)
    options = sample["questions"]["money_field"]["criteria"]
    name = next(iter(options))
    options[name] = "¥1"
    problems = judges_mod.LabelJudge().judge(sample).raw["problems"]
    assert any(name in item for item in problems)


def test_recompute_helpers(make_spec: Any) -> None:
    """输入：反解结果。输出：重算出来的答案。

    预期行为：`_recompute_bool` 只认得出两个 noul 题族（别的返回 None，不能瞎猜）；
    `_recompute_bad_lines` 封顶 3（等级表只有四档）。
    """
    loc = locales.load("zh")
    spec = make_spec(281, lines=4, defects=3, tax_state="ok")
    parsed = parse.parse_document(render.build(spec, loc), loc)
    assert judges_mod._recompute_bool("subtotal_consistent", parsed) is spec.subtotal_consistent
    assert judges_mod._recompute_bool("tax_consistent", parsed) is spec.tax_consistent
    assert judges_mod._recompute_bool("bad_lines", parsed) is None
    assert judges_mod._recompute_bad_lines(parsed) == 3


def test_recompute_bool_gives_up_on_an_empty_document() -> None:
    """输入：空文档的反解结果。输出：None。

    预期行为：小计或明细解不出来时返回 None（上层记成「解不出判据需要的字段」），
    绝不能当成「算出来是 False」——那会凭空给出一条错数据。
    """
    empty = parse.ParsedInvoice(fields={}, items=())
    assert judges_mod._recompute_bool("subtotal_consistent", empty) is None
    assert judges_mod._recompute_bool("tax_consistent", empty) is None


def test_value_problems_checks_line_options(make_spec: Any) -> None:
    """输入：明细行选项的三种情形。输出：问题列表。

    预期行为：对得上返回空；行号超出明细行数、单价与单据不一致都要报出来。
    """
    loc = locales.load("zh")
    spec = make_spec(291, lines=2, defects=0)
    parsed = parse.parse_document(render.build(spec, loc), loc)
    good = render.format_money(spec.items[0].unit, spec.case.currency, loc)
    assert judges_mod._value_problems("q", "line_1", good, parsed) == []
    assert any(
        "超出明细行数" in item
        for item in judges_mod._value_problems("q", "line_9", "¥1", parsed)
    )
    assert any(
        "单据里是" in item for item in judges_mod._value_problems("q", "line_1", "¥1", parsed)
    )


def test_value_problems_rejects_unparseable_values(make_spec: Any) -> None:
    """输入：日期槽配一句不是日期的描述。输出：问题列表。

    预期行为：解不出来就报错，而不是当成「相等」放过去。
    """
    loc = locales.load("zh")
    spec = make_spec(301)
    parsed = parse.parse_document(render.build(spec, loc), loc)
    problems = judges_mod._value_problems("q", "due", "不是日期", parsed)
    assert any("解不出来" in item for item in problems)
