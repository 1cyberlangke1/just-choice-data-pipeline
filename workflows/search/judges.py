"""检索 workflow 的判别器：构造式数据用纯规则判别即可（**零 token**）。

- `RoundtripJudge`：把结果页反解回字段，与真值 spec **逐字**比对
  （抓字段漏写、分隔符写错、摘要被截断）；
- `LabelJudge`：反解文本后**独立重算**每道题的金答案（页文本 → 事实 → 答案），
  与数据里的答案比对。它只看渲染出来的文本与题面，不看 spec 里的答案——
  否则就是「拿答案验证答案」。

两族题（问「结果 N」与问「结果 i 与结果 j」）点名的编号**从题面本身读**：
「问的是第几条」是题面的一部分，判别器不该回头问生成器。

输入/输出：只出 `Verdict`，不改样本。对不上就判不通过——宁可少要几条，
也不要一条「题面和页面对不上」的数据。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from jc.judge import Verdict
from jc.workflow import family_of

from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import AUTHORITIES, TOPIC_NAMES, SearchSpec, TruthRow

_NUM: Final[re.Pattern[str]] = re.compile(r"\d+")


def question_indices(sample: dict[str, Any]) -> tuple[int, tuple[int, int]]:
    """输入：样本。输出：(被点名的单条编号, 被点名的那一对编号)。

    预期行为：从**题面**里读编号——同组题面给出的编号必须一致，不一致直接报错，
    不猜也不取第一个：编号对不上说明渲染与出题走偏了，那是要拦下来的错。
    """
    probes: set[int] = set()
    pairs: set[tuple[int, int]] = set()
    for qid, question in sample["questions"].items():
        family = family_of(qid)
        if family in questions.PROBE_FAMILIES:
            numbers = [int(token) for token in _NUM.findall(question["instructions"])]
            if len(numbers) != 1:
                raise ValueError(f"{qid} 的题面里编号不是恰好一个：{question['instructions']!r}")
            probes.add(numbers[0])
        elif family in questions.PAIR_FAMILIES:
            numbers = [int(token) for token in _NUM.findall(question["instructions"])]
            if len(numbers) != 2:
                raise ValueError(f"{qid} 的题面里编号不是恰好两个：{question['instructions']!r}")
            pairs.add((numbers[0], numbers[1]))
    if len(probes) != 1 or len(pairs) != 1:
        raise ValueError(f"题面里的编号不一致：单条 {sorted(probes)}，成对 {sorted(pairs)}")
    return probes.pop(), pairs.pop()


def topic_of_query(page: parse.ParsedPage, loc: dict[str, Any]) -> str:
    """输入：反解出来的页、语言字典。输出：问题问的主题键。

    预期行为：按「问题里出现了哪个主题名」认，必须**恰好一个**——各主题的名字互不包含
    （由 `test_locales.py` 锁住），所以这条判据是干净的。认不出或认出多个直接报错：
    静默挑一个会让判别器用错主题，把好数据判成坏的。
    """
    found = [name for name in TOPIC_NAMES if loc["topic"][name]["label"] in page.query]
    if len(found) != 1:
        raise ValueError(f"用户问题里认出的主题不是恰好一个：{found}（问题：{page.query!r}）")
    return found[0]


def truth_rows_of_page(page: parse.ParsedPage, loc: dict[str, Any]) -> tuple[TruthRow, ...]:
    """输入：反解出来的页、语言字典。输出：每条的**可判定事实**（判别侧）。

    预期行为：**只吃文本**——主题是否对上按「标题里有没有问题的主题名」判、类型是否对上
    按两边渲染出来的类型文本判、时效按日期比大小判、权威按语言文件反查回键。
    不看 spec 里的任何答案。
    """
    topic_label = loc["topic"][topic_of_query(page, loc)]["label"]
    authority_key = {loc["authority"][key]: key for key in AUTHORITIES}
    rows: list[TruthRow] = []
    for row in page.results:
        if row.authority not in authority_key:
            raise ValueError(f"认不出的来源权威：{row.authority!r}")
        rows.append(
            TruthRow(
                index=row.index,
                doc_id=row.doc_id,
                topic_match=topic_label in row.title,
                kind_match=row.kind == page.want_kind,
                fresh=row.updated >= page.since,
                authority=authority_key[row.authority],
                updated=row.updated,
                snippet=row.snippet,
            )
        )
    return tuple(rows)


def recompute(
    page: parse.ParsedPage, loc: dict[str, Any], probe: int, pair: tuple[int, int]
) -> dict[str, Any]:
    """输入：反解出来的页、语言字典、题面点名的单条与成对编号。输出：{题族名: 重算的答案}。

    预期行为：**只吃文本**——所以它能独立验证构造器。
    """
    rows = truth_rows_of_page(page, loc)
    by_index = {row.index: row for row in rows}
    missing = [index for index in (*pair, probe) if index not in by_index]
    if missing:
        raise ValueError(f"题面点名的编号在这一页里不存在：{missing}")
    topic = loc["topic"][topic_of_query(page, loc)]
    points = tuple(topic["points"])
    top = spec_mod.best_of(rows)
    left, right = (by_index[index] for index in pair)
    probe_row = by_index[probe]
    return {
        "best_result": f"result_{top.index}",
        "next_action": spec_mod.action_of(rows, points),
        "relevance": spec_mod.relevance_of(probe_row),
        "covered_points": spec_mod.covers_of(probe_row, points),
        "answerable": spec_mod.covers_of(top, points) == len(points),
        "conflict": spec_mod.conflict_of(left, right, topic["metric"], tuple(topic["values"])),
        "same_source": left.doc_id == right.doc_id,
        "probe_fresh": probe_row.fresh,
    }


def roundtrip_diff(spec: SearchSpec, page: parse.ParsedPage, loc: dict[str, Any]) -> list[str]:
    """输入：真值规格、反解出来的页、语言字典。输出：差异列表（空 = 一致）。

    预期行为：请求头四个字段与每一条的七格逐字比——连缩进被吃掉、摘要被截断都算差异。
    条数不同就直接返回：再比下去只会刷一屏「第 N 条」。
    """
    diff: list[str] = []
    fields = render.topic_fields(spec.topic.name, loc)
    want_query = render.query_text(spec, loc)
    if page.query != want_query:
        diff.append(f"用户问题：渲染 {page.query!r} vs 期望 {want_query!r}")
    if page.want_kind != loc["doc_kind"][spec.want_kind]:
        diff.append(f"要求的文档类型：渲染 {page.want_kind!r} vs 期望 {loc['doc_kind'][spec.want_kind]!r}")
    if page.since != spec.since:
        diff.append(f"时效要求：渲染 {page.since!r} vs 期望 {spec.since!r}")
    want_points = (fields["p1"], fields["p2"])
    if page.points != want_points:
        diff.append(f"问题要点：渲染 {page.points!r} vs 期望 {want_points!r}")
    if len(page.results) != len(spec.results):
        diff.append(f"结果条数：渲染 {len(page.results)} vs 期望 {len(spec.results)}")
        return diff
    for want, got in zip(spec.results, page.results, strict=True):
        row_fields = render.topic_fields(want.topic, loc)
        checks = (
            ("编号", got.index, want.index),
            ("文档号", got.doc_id, want.doc_id),
            ("标题", got.title, loc["title"][want.title]["text"].format(**row_fields)),
            ("类型", got.kind, loc["doc_kind"][want.kind]),
            ("来源", got.authority, loc["authority"][want.authority]),
            ("更新日期", got.updated, want.updated),
            ("摘要", got.snippet, loc["snippet"][want.snippet]["text"].format(**row_fields)),
        )
        for name, rendered, expected in checks:
            if rendered != expected:
                diff.append(f"第 {want.index} 条的{name}：渲染 {rendered!r} vs 期望 {expected!r}")
    return diff


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：结果页反解后必须与真值 spec 完全一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, SearchSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        try:
            page = parse.parse_page(
                sample["state"], parse.label_index(loc), (loc["meta"]["field_sep"],)
            )
        except ValueError as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        diff = roundtrip_diff(spec, page, loc)
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
            page = parse.parse_page(
                sample["state"], parse.label_index(loc), (loc["meta"]["field_sep"],)
            )
            probe, pair = question_indices(sample)
            expected = recompute(page, loc, probe, pair)
        except ValueError as exc:
            return Verdict(False, reason=f"重算失败：{exc}")
        problems = self._check(sample, expected)
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )

    def _check(self, sample: dict[str, Any], expected: dict[str, Any]) -> list[str]:
        """输入：样本、重算出来的答案。输出：问题列表（空 = 全部对得上）。"""
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
                    problems.append(f"{qid}: 答案指向 {got}，按页面重算是 {want}")
            elif qtype == "noul":
                if answer["noul"] is not want:
                    problems.append(f"{qid}: 答案是 {answer['noul']}，按页面重算是 {want}")
            elif answer["score"] != want:
                problems.append(f"{qid}: 答案是 {answer['score']}，按页面重算是 {want}")
        return problems
