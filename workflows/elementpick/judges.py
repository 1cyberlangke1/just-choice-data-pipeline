"""界面元素定位 workflow 的判别器：结构往返 + 答案重算。

输入：样本（`state` / `questions` / `answers`）、真值规格、语言字典。输出：`Verdict`。
预期行为：两个判别器走**两条不同的路**——
- `roundtrip`：从 state **文本**把屏号、动作标签与六行元素的四格读回来，与真值逐格比对。
  这条路能抓到「渲染把某一格写串味了」；
- `label`：拿真值规格重新出一次题，比对题号与答案。这条路能抓到组装器 / 选项打乱出错。

`recompute` 与 `compare` 都对外暴露：`audit` 要拿它们在**产物层面**重算同一套东西，
这样老产物也能事后查（不用重跑生成）。两条路都只吃文本或规格，**都不碰模型**。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from dataclasses import dataclass

from jc import assemble
from jc.judge import Verdict
from jc.workflow import answer_value, compare_expected

from . import locales, parse, questions
from .spec import ElementSpec


def recompute(
    screen: parse.ParsedScreen, loc: dict[str, JsonValue]
) -> dict[str, list[JsonValue]]:
    """输入：反解出来的清单、语言字典。输出：{题族: 期望答案列表}（按题序）。

    预期行为：走**同一套出题逻辑**（`questions.build`），只是输入换成从文本读回来的
    动作标签与六行元素——所以「渲染写歪」会直接变成「答案对不上」。
    问法编号给 0：它只影响措辞，不影响答案。
    """
    out: dict[str, list[JsonValue]] = {}
    for item in questions.build(screen.rows, screen.action, loc, 0):
        out.setdefault(item.family, []).append(item.answer)
    return out


def compare(sample: dict[str, JsonValue], expected: dict[str, list[JsonValue]]) -> list[str]:
    """输入：一条样本、按 state 重算出来的期望答案表。输出：差异列表（空 = 全对）。

    预期行为：判据在 `jc.workflow.compare_expected`，这里只把本 workflow 的题族表交进去——
    题族是各家的领域词表，通用层不认。
    """
    return compare_expected(sample, expected, questions.FAMILIES)


def roundtrip_diff(
    spec: ElementSpec, screen: parse.ParsedScreen, loc: dict[str, JsonValue]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：不一致项列表（空 = 逐格一致）。

    预期行为：屏号、动作标签与元素表每一行的四格都要对得上——比的是**键**，所以
    「渲染取了别的语言文案」不算差异（两边都会翻回同一个键）。
    """
    del loc
    checks: list[tuple[str, object, object]] = [
        ("屏号", screen.screen_id, spec.screen_id),
        ("动作", screen.action, spec.case.label),
    ]
    if len(screen.rows) != len(spec.rows):
        checks.append(("元素行数", len(screen.rows), len(spec.rows)))
    else:
        for index, (got, want) in enumerate(zip(screen.rows, spec.rows), start=1):
            checks.append(
                (
                    f"第 {index} 行四格",
                    (got.role, got.visibility, got.clickable, got.text),
                    (want.role, want.visibility, want.clickable, want.text),
                )
            )
    return [
        f"{name}：渲染 {got!r} vs 期望 {want!r}" for name, got, want in checks if got != want
    ]


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：清单反解后必须与真值规格逐格一致。"""

    name: str = "roundtrip"
    locale: str = "zh"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, ElementSpec):
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

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：规格类型不对直接报错。"""
        if not isinstance(spec, ElementSpec):
            return Verdict(False, reason=f"判别器拿到的规格类型不对：{type(spec).__name__}")
        loc = locales.load(self.locale)
        expected = questions.build(spec.rows, spec.case.label, loc, spec.phrasing)
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
