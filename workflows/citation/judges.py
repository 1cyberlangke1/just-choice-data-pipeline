"""引文核查 workflow 的判别器：构造式数据用纯规则判别即可（**零 token**）。

- `RoundtripJudge`：把文本反解回字段，与真值 spec **逐字**比对
  （抓字段漏写、小节错位、句子被折行）；
- `LabelJudge`：反解文本后**独立重算**每道题的金答案（文本 → 句子表 → 三条事实 → 答案），
  与数据里的答案比对。它只看渲染出来的文本与题面，不看 spec 里的答案——
  否则就是「拿答案验证答案」。

重算的三条事实（引文在不在文档里、在不在声称的小节里、声称小节的态度）与构造侧走的是
**同一套规则**（`spec.relation_of` / `spec.answers_of`），只是入口不同：构造侧从 spec 的结构，
这里从文本反查回 (主题, 句, 极性) 再算。两边对不上由测试当场抓。

输入/输出：只出 `Verdict`，不改样本。对不上就判不通过——宁可少要几条，
也不要一条「题面和文档对不上」的数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from jc.judge import Verdict
from jc.workflow import family_of

from . import locales, parse, render
from . import spec as spec_mod
from .spec import CitationSpec, Fact


def _put(table: dict[str, Fact], text: str, fact: Fact, lang: str) -> None:
    """输入：表、文本、事实、语言。输出：无。

    预期行为：同一段文本映射到两个**不同**的事实就直接报错——那说明语言文件里有歧义，
    反解会随机挑一个，必须当场拦下来。
    """
    if text in table and table[text].key() != fact.key():
        raise ValueError(
            f"文本 {text!r}（{lang}）对应两个不同的事实：{table[text].key()} 与 {fact.key()}"
        )
    table[text] = fact


@lru_cache(maxsize=4)
def tables() -> tuple[dict[str, Fact], dict[str, Fact]]:
    """输入：无。输出：(句子文本 → 事实, 主张转述文本 → 事实)，合并全部语言。

    预期行为：语言无关——判别器手上没有 case，只能靠「这段文本是表里的哪一句」反查。
    各语言、各主题的文本互不重叠（由 `test_locales.py` 锁住），重叠在这里当场炸。
    """
    sentences: dict[str, Fact] = {}
    claims: dict[str, Fact] = {}
    for lang in locales.available():
        loc = locales.load(lang)
        for topic in spec_mod.TOPIC_NAMES:
            entry = loc["topic"][topic]
            for sentence in spec_mod.SENTENCE_IDS[topic]:
                item = entry["sentence"][sentence]
                for text, negated in ((item["doc"], False), (item["neg"], True)):
                    _put(sentences, text, Fact(topic, sentence, negated), lang)
                for text, negated in ((item["claim"], False), (item["claim_neg"], True)):
                    _put(claims, text, Fact(topic, sentence, negated), lang)
    return sentences, claims


def claim_fact(claim_line: str) -> Fact:
    """输入：「主张」那一行的值。输出：它转述的是哪一句。

    预期行为：认出的转述必须**恰好一条**——认不出或认出多条直接报错，
    静默挑一个会让判别器用错主张，把好数据判成坏的。
    """
    found = [fact for text, fact in tables()[1].items() if text in claim_line]
    if len(found) != 1:
        raise ValueError(f"主张里认出的转述不是恰好一条：{found}（{claim_line!r}）")
    return found[0]


def recompute(parsed: parse.ParsedCitation) -> dict[str, Any]:
    """输入：反解结果。输出：{题族名: 按文本重算的答案}。

    预期行为：**只吃文本**——句子靠句子表反查回事实、引文靠整句匹配定在不在文档里、
    声称出处靠那一行里的编号。三条事实算出来之后，剩下的全走 `spec.answers_of`。
    """
    sentences, _ = tables()
    by_section: dict[int, tuple[Fact, ...]] = {}
    for section in parsed.sections:
        facts = []
        for text in section.sentences:
            fact = sentences.get(text)
            if fact is None:
                raise ValueError(f"这一句不在句子表里：{text!r}")
            facts.append(fact)
        by_section[section.index] = tuple(facts)
    claimed = parse.claimed_index(parsed.source)
    if claimed not in by_section:
        raise ValueError(f"声称的小节 {claimed} 在文本里不存在")
    quote_fact = sentences.get(parsed.quote)
    home: int | None = None
    if quote_fact is not None:
        for index, facts in by_section.items():
            if any(item.key() == quote_fact.key() for item in facts):
                home = index
                break
    found = home is not None
    located = found and home == claimed
    relation = spec_mod.relation_of(claim_fact(parsed.claim), by_section[claimed])
    return spec_mod.answers_of(found, located, relation)


def roundtrip_diff(
    spec: CitationSpec, parsed: parse.ParsedCitation, loc: dict[str, Any]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：差异列表（空 = 一致）。

    预期行为：来源文档行、小节数、每节的编号 / 标题 / 两句、主张、引文、声称出处逐字比。
    """
    diff: list[str] = []
    want_doc = render.doc_line(spec, loc)
    if parsed.doc != want_doc:
        diff.append(f"来源文档：渲染 {parsed.doc!r} vs 期望 {want_doc!r}")
    if len(parsed.sections) != len(spec.sections):
        diff.append(f"小节数：渲染 {len(parsed.sections)} vs 期望 {len(spec.sections)}")
    for want in spec.sections:
        try:
            got = parsed.section(want.index)
        except ValueError:
            diff.append(f"缺少小节 {want.index}")
            continue
        want_heading = render.heading_text(want, loc)
        if got.heading != want_heading:
            diff.append(f"小节 {want.index} 标题：渲染 {got.heading!r} vs 期望 {want_heading!r}")
        want_facts = tuple(render.fact_text(fact, loc) for fact in want.facts)
        if got.sentences != want_facts:
            diff.append(f"小节 {want.index} 句子：渲染 {got.sentences!r} vs 期望 {want_facts!r}")
    for name, got, want in (
        ("主张", parsed.claim, render.claim_line(spec, loc)),
        ("引文", parsed.quote, render.quote_text(spec, loc)),
        ("声称出处", parsed.source, render.source_line(spec, loc)),
    ):
        if got != want:
            diff.append(f"{name}：渲染 {got!r} vs 期望 {want!r}")
    return diff


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：文本反解后必须与真值 spec 完全一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, CitationSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        try:
            parsed = parse.parse_citation(sample["state"], parse.label_index(loc))
        except ValueError as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        diff = roundtrip_diff(spec, parsed, loc)
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
            parsed = parse.parse_citation(sample["state"], parse.label_index(loc))
            expected = recompute(parsed)
        except ValueError as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        problems = self._check(sample, expected)
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )

    def _check(self, sample: dict[str, Any], expected: dict[str, Any]) -> list[str]:
        """输入：样本、按文本重算的答案。输出：问题列表（空 = 全部对得上）。"""
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
                    problems.append(f"{qid}: 答案指向 {got}，按文本重算是 {want}")
            elif qtype == "noul":
                if answer["noul"] is not want:
                    problems.append(f"{qid}: 答案是 {answer['noul']}，按文本重算是 {want}")
            elif answer["score"] != want:
                problems.append(f"{qid}: 答案是 {answer['score']}，按文本重算是 {want}")
        return problems
