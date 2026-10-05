"""noul 题族的题面与判据必须同极性：问「有没有 X」时 `true` 那句不能写「没有 X」。

为什么要有：noul 的答案是一个布尔，`criteria` 的 `true` / `false` 两句是「取这个布尔值时，
世界是什么样」。题面若问「有没有 X？」，而 `true` 那句写的是「没有 X」，那么**按题面答「有」
的考生会被判成答错**——喂进训练集就是「问有没有、答 `false` 才是真有」这种反逻辑。

这个错没有别的东西会拦：判据自身是自洽的（两句互斥且穷尽），标签由代码算也自洽，
`jc audit` 反解得回来，语言闸与键对称闸只看结构。真机踩过：`siftrank` 的 `stable`
三个问法里前两个问的正是「有没有矛盾」，`hybrid` 的 en 侧 `room_free` 第 2 个问法同病
（见 `docs_local/人工抽检记录.md` 第十六轮）。

输入：`workflows/*/locales/*.toml`。
输出：无（断言）。
预期行为：题面是**纯存在性问句**（含「有没有 / 出现过 / Do any… / Is there…」，且不带澄清小句）
时，`criteria["true"]` 的**句首**不能是否定词。带澄清小句的问法（「…请判断『没有冲突』对不对」）
不算错——那种问法已经把「`true` 指什么」写进题面了。

已知边界：这是文案层面的启发式，只判「句首否定 + 纯存在性问句」这一个形状；
换个说法把极性写反（例如问「是否一致」而 `true` 写「有矛盾」）它抓不到，不假装能抓。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Final

#: 题面里的「存在性问句」标记。
ASK_EXISTS: Final[tuple[str, ...]] = (
    "有没有",
    "出现过",
    "是否存在",
    "Do any",
    "Does any",
    "Is there",
    "Are there",
)

#: 题面里自带澄清小句的标记——有这些就说明问法已经把「`true` 指什么」写进题面了。
CLARIFY: Final[tuple[str, ...]] = (
    "请判断",
    "对吗",
    "是不是",
    "Judge whether",
    "right?",
    "一致",
    "consistent",
    "相同",
    "same order",
)

#: `true` 那句**句首**的否定标记。
NEGATIVE: Final[tuple[str, ...]] = ("没有", "无——", "不是", "No -", "No—", "None", "Never")


def polarity_problem(question: str, true_text: str) -> bool:
    """输入：一句问法、判据 `true` 那句。输出：这一句问法是不是与 `true` 的极性相反。

    预期行为：两个条件同时成立才算——问法是纯存在性问句（不带澄清小句），且 `true` 那句**句首**
    是否定词。只看句首，是因为 `is not exactly equal` 这类句子中间的否定是描述的一部分，
    不是极性（`elementpick` 的 `shadow` 就长这样）。
    """
    if not any(mark in question for mark in ASK_EXISTS):
        return False
    if any(mark in question for mark in CLARIFY):
        return False
    return true_text.lstrip().startswith(NEGATIVE)


def offenders() -> list[str]:
    """输入：无。输出：极性相反的问法，每条一个字符串（路径 + 题族 + 问法 + 判据）。

    预期行为：扫 `workflows/*/locales/*.toml`，只看同时有 `q` 与 `criteria`、且判据键正好是
    `true` / `false` 的题族；一个题族里只要有一句问法出问题就报一次，带上那一句。
    """
    found: list[str] = []
    for path in sorted(Path("workflows").glob("*/locales/*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        fam = data.get("fam")
        if not isinstance(fam, dict):
            continue
        for name, spec in fam.items():
            if not isinstance(spec, dict):
                continue
            criteria = spec.get("criteria")
            questions = spec.get("q")
            if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                continue
            if not isinstance(questions, list):
                continue
            true_text = str(criteria["true"])
            for question in questions:
                if polarity_problem(str(question), true_text):
                    found.append(
                        f"{path.as_posix()} [{name}]：题面问「{question}」，"
                        f"而判据 true 写的是「{true_text}」——按题面答会落到 false 上"
                    )
                    break
    return found


def test_existence_question_with_negative_true_is_flagged() -> None:
    """输入：问「有没有矛盾」而 `true` 写「没有矛盾」。输出：无。预期行为：报出来。"""
    assert polarity_problem(
        "这些批次之间，有没有任何一对条目的先后顺序是相反的？",
        "没有——每一对条目在各批次里的先后都一致",
    )
    assert polarity_problem(
        "Do any two batches disagree on the relative order of some pair of items?",
        "No - every pair of items keeps the same relative order in every batch it appears in",
    )


def test_affirmative_true_is_allowed() -> None:
    """输入：问「有没有重复」而 `true` 写「有」。输出：无。预期行为：不报。"""
    assert not polarity_problem("这屏材料里有没有主题重复出现的片段？", "有")


def test_clarifying_clause_is_allowed() -> None:
    """输入：问法里自带澄清小句。输出：无。预期行为：不报——那种问法已把极性写进题面。"""
    assert not polarity_problem(
        "有没有两场会议同时占用同一间会议室？请判断「会议室没有冲突」这句话对不对。",
        "没有冲突",
    )
    assert not polarity_problem(
        "这些批次彼此一致吗——有没有哪一对条目的先后被两个批次写反了？",
        "没有——每一对条目在各批次里的先后都一致",
    )


def test_mid_sentence_negation_is_allowed() -> None:
    """输入：`true` 那句句首肯定、句中有否定。输出：无。预期行为：不报。"""
    assert not polarity_problem(
        "Is there an element whose name is only approximate?",
        "Yes: some element's text contains the action label as a substring but is not exactly equal",
    )


def test_non_existence_question_is_allowed() -> None:
    """输入：不是存在性问句的问法（判据写否定式）。输出：无。预期行为：不报。"""
    assert not polarity_problem("第 1 题，第 1 个候选答对了吗？", "没有答对")
    assert not polarity_problem("这条工单属于哪一类问题？", "没有这一类")


def test_no_family_asks_one_way_and_labels_the_other() -> None:
    """输入：无（扫全部语言包）。输出：无。

    预期行为：一家都不许出现「题面问有没有、`true` 却写没有」。有就把每一条都聚在一条断言里
    （路径 + 题族 + 那一句问法 + 判据原文），一次看全，改文案即可。
    """
    found = offenders()
    assert not found, "\n".join(found)
