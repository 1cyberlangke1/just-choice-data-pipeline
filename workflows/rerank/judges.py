"""检索重排（rerank）workflow 的判别器：反解 state 后独立重算全部七个答案。

输入：样本（state + 题目 + 答案）+ 真值规格。输出：`Verdict`。
预期行为：
- `roundtrip`：反解出来的查询与候选，必须与真值规格**逐格一致**；
- `label`：只看 state 与**题面**，独立重算七个答案再与产物比对——
  所以「生成器把答案算错了」与「渲染把正文写歪了」会分别被抓到。

反解不复用渲染器的任何计算：主题 / 方面词是从**语言文件里的模板生成的正则**抓出来的，
语言文件改字面量时正则自动跟着变（见 `body_regex`）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from jc.judge import Verdict
from jc.workflow import family_of

from . import locales, parse, render
from . import spec as spec_mod
from .spec import Cand, RerankSpec

#: 题面里认候选编号用的正则：词汇表来自 `render.CAND_IDS`（不手写 `C\d`，
#: 免得改了编号规则之后判别器还在认老编号）。前后加断言，避免 `C1` 命中 `C10` 的一部分。
CAND_TOKEN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(cid) for cid in render.CAND_IDS) + r")(?![0-9])"
)


def body_regex(template: str) -> re.Pattern[str]:
    """输入：正文模板（恰好含 `{topic}` 与 `{aspect}` 各一次）。输出：全匹配用的正则。

    预期行为：前一个占位符**懒惰**、后一个**贪婪**，配合 `fullmatch` 使用。
    两个都懒惰时后一个只吃一个字符（`terse` 那种「主题 / 方面」写法会被截成「备」）；
    两个都贪婪时前一个会把分隔符一起吃进去。
    """
    head, rest = template.split("{topic}", 1)
    middle, tail = rest.split("{aspect}", 1)
    return re.compile(
        re.escape(head) + "(?P<topic>.+?)" + re.escape(middle) + "(?P<aspect>.+)" + re.escape(tail)
    )


def nth_regex(template: str) -> re.Pattern[str]:
    """输入：含 `{nth}` 的题面模板。输出：抓位次的正则。

    预期行为：用 `search` 而不是 `fullmatch`——题面末尾还接着判据原文那一段。
    """
    head, tail = template.split("{nth}", 1)
    return re.compile(re.escape(head) + r"(?P<nth>\d+)" + re.escape(tail))


def read_query(text: str, loc: dict[str, Any]) -> tuple[str, str]:
    """输入：查询原文、语言字典。输出：(主题词, 方面词) 的原文。

    预期行为：从 `[query].text` 生成正则；认不出直接报人话——那说明查询被写歪了。
    """
    head, rest = loc["query"]["text"].split("{topic}", 1)
    middle, tail = rest.split("{aspect}", 1)
    pattern = re.compile(
        re.escape(head) + "(?P<topic>.+?)" + re.escape(middle) + "(?P<aspect>.+)" + re.escape(tail)
    )
    found = pattern.fullmatch(text.strip())
    if found is None:
        raise ValueError(f"查询读不出主题与方面：{text!r}")
    return found.group("topic"), found.group("aspect")


@dataclass(frozen=True)
class ReadCand:
    """一条候选反解出来的三件事：主题词、方面词、有没有示例。"""

    topic: str
    aspect: str
    example: bool


def read_cand(body: tuple[str, ...], loc: dict[str, Any]) -> ReadCand:
    """输入：一条候选的正文各行、语言字典。输出：`ReadCand`。

    预期行为：逐行试每种措辞的模板，第一个匹配上的那行给出主题 / 方面 / 措辞；
    再按**同一措辞**的两句示例文案认「有 / 没有示例」；其余行必须能对上噪声句，
    对不上就报错——多出一行认不出的正文，说明渲染漏了模板或模板被改坏了。
    """
    topic = aspect = wording = None
    example: bool | None = None
    used: set[int] = set()
    for index, line in enumerate(body):
        for candidate_wording in spec_mod.WORDINGS:
            found = body_regex(loc["sentence"]["body"][candidate_wording]).fullmatch(line)
            if found is not None:
                topic = found.group("topic")
                aspect = found.group("aspect")
                wording = candidate_wording
                used.add(index)
                break
        if wording is not None:
            break
    if topic is None or aspect is None or wording is None:
        raise ValueError(f"候选正文里读不出主题与方面：{body!r}")
    texts = loc["sentence"]["example"][wording]
    for index, line in enumerate(body):
        if index in used:
            continue
        if line == texts["yes"]:
            example = True
            used.add(index)
        elif line == texts["no"]:
            example = False
            used.add(index)
    if example is None:
        raise ValueError(f"候选正文里读不出「有没有示例」：{body!r}")
    traps = set(loc["sentence"]["trap"].values())
    unknown = [line for index, line in enumerate(body) if index not in used and line not in traps]
    if unknown:
        raise ValueError(f"候选正文里有认不出的行：{unknown}")
    return ReadCand(topic, aspect, example)


@dataclass(frozen=True)
class Facts:
    """从 state 反解并重算出来的事实：候选编号、四条候选、名次、相关度。"""

    cids: tuple[str, ...]
    cands: tuple[Cand, ...]
    order: tuple[int, ...]
    relevance: tuple[int, ...]


def recompute(parsed: parse.ParsedList, loc: dict[str, Any]) -> Facts:
    """输入：反解结果、语言字典。输出：`Facts`。

    预期行为：只从文本本身算——查询的主题 / 方面词决定前两条必要条件，
    候选表头的日期决定第三条，正文里的示例句决定第四条。
    """
    query_topic, query_aspect = read_query(parsed.query, loc)
    reverse_source = locales.reverse(loc["source"])
    by_cid = {cand.cid: cand for cand in parsed.cands}
    missing = [cid for cid in render.CAND_IDS if cid not in by_cid]
    if missing:
        raise ValueError(f"候选表里缺这几条：{missing}")
    cands: list[Cand] = []
    for cid in render.CAND_IDS:
        entry = by_cid[cid]
        read = read_cand(entry.body, loc)
        cands.append(
            Cand(
                topic_ok=read.topic == query_topic,
                aspect_ok=read.aspect == query_aspect,
                example=read.example,
                source=reverse_source[entry.source],
                day=entry.day,
            )
        )
    return Facts(
        cids=render.CAND_IDS,
        cands=tuple(cands),
        order=spec_mod.order_of(tuple(cands)),
        relevance=tuple(spec_mod.relevance(cand) for cand in cands),
    )


def named_cands(text: str) -> list[str]:
    """输入：题面。输出：题面里出现的候选编号（按出现顺序，可能重复）。"""
    return [match.group(1) for match in CAND_TOKEN.finditer(text)]


def asked_nth(text: str, loc: dict[str, Any]) -> int:
    """输入：题面、语言字典。输出：题面里问的位次（1-based）。

    预期行为：逐个问法试；都读不出就报错——位次读错会让整族答案全错，
    宁可炸掉也不能猜一个。
    """
    for template in loc["fam"]["nth"]["q"]:
        found = nth_regex(template).search(text)
        if found is not None:
            return int(found.group("nth"))
    raise ValueError(f"题面里读不出位次：{text.splitlines()[0]!r}")


def option_text(facts: Facts, index: int, loc: dict[str, Any]) -> str:
    """输入：事实、候选下标、语言字典。输出：该候选在选项里的描述原文。

    预期行为：用与 `questions.options_of` 同一个模板拼出来再比——比「从描述里切出编号」
    稳，模板改了也不会悄悄比错。
    """
    cand = facts.cands[index]
    return loc["option"]["cand"].format(
        cid=facts.cids[index], source=loc["source"][cand.source], day=cand.day
    )


def check_sample(sample: dict[str, Any], facts: Facts, loc: dict[str, Any]) -> list[str]:
    """输入：样本、从 state 重算出来的事实、语言字典。输出：问题列表（空 = 全部对得上）。

    预期行为：逐题按**题族**重算再比对——choice 比选项描述原文、noul 比布尔、
    score 比档位下标。做成模块级函数是为了让 `audit` 也能直接复用同一套判据
    （判别器与审计必须用同一把尺，否则「产物审计过了但判别器不过」这种矛盾会长期潜伏）。
    """
    problems: list[str] = []
    for qid, question in sample["questions"].items():
        family = family_of(qid)
        answer = sample["answers"][qid]
        text = question["instructions"]
        try:
            want = expected_of(family, text, facts, loc)
        except (ValueError, KeyError) as exc:
            problems.append(f"{qid}: 从题面读不出判据（{exc}）")
            continue
        qtype = question["type"]
        if qtype == "choice":
            # 组装后的 `criteria` 是「选项名 → 选项描述」的对象，`choice` 是**描述在其中的
            # 书写顺序**；判别器要比的是描述原文，所以取 `values()` 而不是 key。
            options = list(question["criteria"].values())
            index = answer["choice"]
            if not isinstance(index, int) or not 0 <= index < len(options):
                problems.append(f"{qid}: 答案编号越界（{index}）")
                continue
            if options[index] != want:
                problems.append(f"{qid}: 答案指向 {options[index]}，按 state 重算是 {want}")
        elif qtype == "noul":
            if answer["noul"] is not want:
                problems.append(f"{qid}: 答案是 {answer['noul']}，按 state 重算是 {want}")
        elif answer["score"] != want:
            problems.append(f"{qid}: 答案是 {answer['score']}，按 state 重算是 {want}")
    return problems


def expected_of(
    family: str, text: str, facts: Facts, loc: dict[str, Any]
) -> str | bool | int:
    """输入：题族、题面、事实、语言字典。输出：按 state + 题面重算出来的答案。

    预期行为：点名了候选编号 / 位次的题族**从题面读**编号与位次——题面是样本的一部分，
    读它不算作弊，反而是「题面里必须写清判据」这条纪律的落地。
    """
    if family == "first":
        return option_text(facts, facts.order[0], loc)
    if family == "last":
        return option_text(facts, facts.order[-1], loc)
    if family == "nth":
        return option_text(facts, facts.order[asked_nth(text, loc) - 1], loc)
    names = named_cands(text)
    if family == "ahead":
        if len(names) != 2:
            raise ValueError(f"题面里的候选编号不是两个：{names}")
        left, right = (facts.cids.index(name) for name in names)
        return facts.order.index(left) < facts.order.index(right)
    if family == "same_relevance":
        if len(names) != 2:
            raise ValueError(f"题面里的候选编号不是两个：{names}")
        left, right = (facts.cids.index(name) for name in names)
        return facts.relevance[left] == facts.relevance[right]
    if family == "on_topic":
        if len(names) != 1:
            raise ValueError(f"题面里的候选编号不是一个：{names}")
        return spec_mod.on_topic_of(facts.cands[facts.cids.index(names[0])])
    if family == "relevance":
        if len(names) != 1:
            raise ValueError(f"题面里的候选编号不是一个：{names}")
        return facts.relevance[facts.cids.index(names[0])]
    raise ValueError(f"未知题族：{family!r}")


def roundtrip_diff(
    spec: RerankSpec, parsed: parse.ParsedList, loc: dict[str, Any]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐格一致）。"""
    case = spec.case
    want_query = loc["query"]["text"].format(
        topic=loc["topic"][case.topic], aspect=loc["aspect"][case.aspect]
    )
    checks: list[tuple[str, str, str]] = [("查询", parsed.query, want_query)]
    by_cid = {cand.cid: cand for cand in parsed.cands}
    for index, cid in enumerate(render.CAND_IDS):
        cand = case.shape.cands[index]
        got = by_cid.get(cid)
        if got is None:
            checks.append((f"{cid} 表头", "（缺失）", "存在"))
            continue
        checks.append((f"{cid} 来源", got.source, loc["source"][cand.source]))
        checks.append((f"{cid} 更新", got.day, cand.day))
        want_body = "\n".join(render.cand_lines(case, index, loc))
        checks.append((f"{cid} 正文", "\n".join(got.body), want_body))
    return [f"{name}：渲染 {got!r} vs 期望 {want!r}" for name, got, want in checks if got != want]


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：反解之后必须与真值规格逐格一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, RerankSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        try:
            parsed = parse.parse_list(sample["state"], parse.label_index(loc))
        except ValueError as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        diff = roundtrip_diff(spec, parsed, loc)
        return Verdict(not diff, reason="; ".join(diff[:5]), raw={"diff": diff})


@dataclass(frozen=True)
class LabelJudge:
    """答案判别器：反解 state + 题面后独立重算金答案，逐题比对。"""

    name: str = "label"
    locale: str = "zh"
    max_problems: int = 5

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec（不用）。输出：Verdict。预期行为：答案对不上就判不通过。"""
        loc = locales.load(self.locale)
        try:
            parsed = parse.parse_list(sample["state"], parse.label_index(loc))
            facts = recompute(parsed, loc)
            problems = check_sample(sample, facts, loc)
        except (ValueError, KeyError) as exc:
            return Verdict(False, reason=f"重算失败：{exc}")
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )
