"""结构恢复的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有两条：

- state 必须反解得出块结构（行号格式、块切分都对）；
- 由文本重推出来的**块类型 / 提示框种类 / 列表有序性 / 并段**必须与四族题的答案一致——比的是
  **选项文案**（不是选项名），所以「选项文案抄错」也在这里暴露；文案是按语言各写一份的，而
  state 里没有抬头可以认语言，所以**任一**支持语言的文案都算对（见 `_option_texts`）。

唯一纯代码读不出来的是「这份纯文本读起来像不像一份真的备忘」——机械口径能算出块结构，但正文要是
写成自指句（「本段是一个段落」），读起来就完全是另一回事。那正是 `judges.TraceJudge` 在合成期把关
的事，审计这里老实写明「这一条靠合成期的反解把关」（范式 §9.13：写不出强校验时不要硬编一个看起来
像校验的东西）。

输入：run 目录、切分名、语言。输出：各切分计数 + 缺失项列表。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from jc.contract import recorded_text
from jc import jsonio
from jc.jsontypes import JsonValue
from jc.workflow import family_of

from . import judges as judges_mod, locales, parse, questions

#: 本 workflow 认得的题族（产物里出现别的就是串版本了）。
KNOWN_FAMILIES: frozenset[str] = frozenset(questions.FAMILIES)
#: 用「选项文案」比的两族（choice）：重算给的是**选项名**（键），产物记的是**文案**。
OPTION_FAMILIES: tuple[str, ...] = ("block_type", "callout_kind")


def _option_texts() -> dict[str, dict[str, set[str]]]:
    """输入：无。输出：{题族: {选项名: 该选项在**各支持语言**下的全部文案}}。

    预期行为：选项文案是按语言各写一份的，而 state 里没有抬头可以认语言（合并读法只用得上标记词），
    所以每个键收一整套文案：产物对上**任一**支持语言的文案即算对，自己另编一套文案会被抓出来。
    """
    table: dict[str, dict[str, set[str]]] = {family: {} for family in OPTION_FAMILIES}
    for lang in locales.available():
        loc = locales.load(lang)
        for family in OPTION_FAMILIES:
            options = (
                questions.block_options(loc)
                if family == "block_type"
                else questions.callout_options(loc)
            )
            for name, text in options.items():
                table[family].setdefault(str(name), set()).add(str(text))
    return table


def _matches(
    family: str, got: JsonValue, want: JsonValue, texts: dict[str, dict[str, set[str]]]
) -> bool:
    """输入：题族、产物里的答案、按文本重算的键、选项文案表。输出：对不对得上。

    预期行为：choice 那两族比**选项文案**（产物记的是文案，重算给的是键）；noul 那两族直接比布尔。
    认不出的键给空集，于是「重算出个不存在的块类型」也会记成对不上，不静默放过。
    """
    if family in texts:
        return str(got) in texts[family].get(str(want), set())
    return got == want


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

    预期行为：`locale` 只用来**当场验一下这个语言存不存在**（别拖到解析阶段才报）；反解一律用
    `render.parse_state` 的合并读法——state 的行号与块结构是语言中立的，没有抬头可以认语言。
    """
    locales.load(locale)
    report: dict[str, JsonValue] = {}
    missing: list[str] = []
    merged = locales.merged()
    texts = _option_texts()
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        counts: dict[str, JsonValue] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "state_bad": 0,
            "join_bad": 0,
            "block_type_bad": 0,
            "callout_bad": 0,
            "ordered_bad": 0,
            "unknown_families": Counter(),
        }
        if not path.exists():
            missing.append(f"{split}: 产物不存在（{path.name}）")
            report[split] = counts
            continue
        for _, row in jsonio.iter_jsonl(path):
            counts["samples"] += 1
            answers = row["answers"]
            recorded: dict[str, JsonValue] = {}
            for qid, question in row["questions"].items():
                family = family_of(qid)
                counts["families"][family] += 1
                counts["types"][question["type"]] += 1
                if family not in KNOWN_FAMILIES:
                    counts["unknown_families"][family] += 1
                recorded[qid] = recorded_text(question, answers[qid])
            try:
                doc = parse.parse_state(str(row["state"]), merged)
            except ValueError:
                counts["state_bad"] += 1
                continue
            expected = judges_mod.recompute(doc, merged)
            for family, key in (
                ("join", "join_bad"),
                ("block_type", "block_type_bad"),
                ("callout_kind", "callout_bad"),
                ("list_ordered", "ordered_bad"),
            ):
                qids = [qid for qid in recorded if family_of(qid) == family]
                want = expected.get(family, [])
                if len(qids) != len(want):
                    counts[key] += 1
                    continue
                for index, qid in enumerate(qids):
                    if not _matches(family, recorded[qid], want[index], texts):
                        counts[key] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("state 反解不出来", "state_bad", "行号格式或块切分不对"),
            ("并段答案与文本不自洽", "join_bad", "join 那一族的答案与按文本重推的对不上"),
            ("块类型答案与文本不自洽", "block_type_bad", "block_type 的答案与按文本重推的对不上"),
            ("提示框种类与文本不自洽", "callout_bad", "callout_kind 的答案与按文本重推的对不上"),
            ("列表有序性与文本不自洽", "ordered_bad", "list_ordered 的答案与按文本重推的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
