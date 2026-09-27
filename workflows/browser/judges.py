"""界面操作 workflow 的判别器：结构往返 + 答案重算。

输入：样本（`state` / `questions` / `answers`）、真值规格、语言字典。输出：`Verdict`。
预期行为：两个判别器走**两条不同的路**——
- `roundtrip`：从 state **文本**把窗口号、任务与五行元素读回来，与真值逐格比对。这条路
  能抓到「渲染把某一格写串味了」（比如状态写错、对象张冠李戴）；
- `label`：拿真值规格重新出一次题，比对题号与答案。这条路能抓到组装器 / 选项打乱出错。

`recompute` 与 `compare` 都对外暴露：`audit` 要拿它们在**产物层面**重算同一套东西，
这样老产物也能事后查（不用重跑生成）。两条路都只吃文本或规格，**都不碰模型**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jc import assemble
from jc.judge import Verdict
from jc.workflow import family_of

from . import locales, parse, questions
from .spec import Element, ScreenSpec


def answer_value(question: dict[str, Any], answer: dict[str, Any]) -> Any:
    """输入：一道题、它的答案。输出：可比的值。

    预期行为：choice 给**选项名**（不是编号——编号只由组装器产生），noul 给布尔；
    编号越界或类型不对给 None（那是对不上，不是崩）。
    """
    if question["type"] == "choice":
        options = list(question["criteria"])
        index = answer.get("choice")
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        if not 0 <= index < len(options):
            return None
        return options[index]
    if question["type"] == "noul":
        value = answer.get("noul")
        return value if isinstance(value, bool) else None
    return answer.get("score")


def rows_from(screen: parse.ParsedScreen) -> tuple[Element, ...]:
    """输入：反解出来的界面。输出：键化后的元素行（查规则用）。

    预期行为：只把三格原文翻回键，不重算任何判断——判断交给 `spec` 的规则函数。
    """
    return tuple(Element(row.role, row.object, row.state) for row in screen.rows)


def recompute(screen: parse.ParsedScreen, loc: dict[str, Any]) -> dict[str, list[Any]]:
    """输入：反解出来的界面、语言字典。输出：{题族: 期望答案列表}（按题序）。

    预期行为：走**同一套出题逻辑**（`questions.build`），只是输入换成从文本读回来的
    元素行与任务——所以「渲染写歪」会直接变成「答案对不上」。问法编号给 0：它只影响
    措辞，不影响答案。
    """
    out: dict[str, list[Any]] = {}
    for item in questions.build(
        rows_from(screen), screen.task, screen.object, loc, 0
    ):
        out.setdefault(item.family, []).append(item.answer)
    return out


def compare(sample: dict[str, Any], expected: dict[str, list[Any]]) -> list[str]:
    """输入：一条样本、期望答案表。输出：差异列表（空 = 全对）。

    预期行为：按题号走一遍，同族按出现顺序对上第几条；题多了、题少了、答案不对、
    题族不认识都各记一笔——不静默取第一个。「少题」那一趟遍历**期望表**（不是产物）：
    遍历产物时整族缺席会一声不响，漏报比误报危险。
    """
    legal = set(questions.FAMILIES)
    seen: dict[str, int] = {}
    problems: list[str] = []
    for qid, question in sample["questions"].items():
        family = family_of(qid)
        if family not in legal:
            problems.append(f"未知题族 {family!r}（题号 {qid}）")
            continue
        index = seen.get(family, 0)
        seen[family] = index + 1
        want = expected.get(family, [])
        if index >= len(want):
            problems.append(f"{qid} 多出来了：文本里只算得出 {len(want)} 道 {family} 题")
            continue
        got = answer_value(question, sample["answers"].get(qid, {}))
        if got != want[index]:
            problems.append(f"{qid}：答案是 {got!r}，按文本重算应为 {want[index]!r}")
    for family, want in expected.items():
        count = seen.get(family, 0)
        if count < len(want):
            problems.append(f"{family} 少了 {len(want) - count} 道题")
    return problems


def roundtrip_diff(
    spec: ScreenSpec, screen: parse.ParsedScreen, loc: dict[str, Any]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐格一致）。"""
    del loc
    checks: list[tuple[str, Any, Any]] = [
        ("窗口号", screen.screen_id, spec.screen_id),
        ("任务意图", screen.task, spec.task),
        ("对象", screen.object, spec.object),
    ]
    if len(screen.rows) != len(spec.rows):
        checks.append(("元素行数", len(screen.rows), len(spec.rows)))
    else:
        for index, (got, want) in enumerate(zip(screen.rows, spec.rows), start=1):
            checks.append(
                (
                    f"第 {index} 行三格",
                    (got.role, got.object, got.state),
                    (want.role, want.object, want.state),
                )
            )
    return [
        f"{name}：渲染 {got!r} vs 期望 {want!r}" for name, got, want in checks if got != want
    ]


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：界面反解后必须与真值规格逐格一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, ScreenSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        try:
            screen = parse.parse_screen(sample["state"], parse.labels_of(loc))
        except (ValueError, KeyError, TypeError) as exc:
            return Verdict(False, reason=f"反解失败：{exc}")
        diff = roundtrip_diff(spec, screen, loc)
        return Verdict(not diff, reason="; ".join(diff[:5]), raw={"diff": diff})


@dataclass(frozen=True)
class LabelJudge:
    """答案判别器：拿真值规格重新出一次题，比对题号与答案。"""

    name: str = "label"
    locale: str = "zh"

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：规格类型不对直接报错。"""
        if not isinstance(spec, ScreenSpec):
            return Verdict(False, reason=f"判别器拿到的规格类型不对：{type(spec).__name__}")
        loc = locales.load(self.locale)
        expected = questions.build(spec.rows, spec.task, spec.object, loc, spec.phrasing)
        qids = assemble.assign_family_ids(expected)
        if list(sample["questions"]) != qids:
            return Verdict(
                False, reason=f"题号与规格不符：{list(sample['questions'])} vs {qids}"
            )
        for qid, item in zip(qids, expected, strict=True):
            got = answer_value(sample["questions"][qid], sample["answers"].get(qid, {}))
            if got != item.answer:
                return Verdict(
                    False, reason=f"{qid}：答案是 {got!r}，按规格应为 {item.answer!r}"
                )
        return Verdict(True, reason="题号与答案与规格逐题一致")
