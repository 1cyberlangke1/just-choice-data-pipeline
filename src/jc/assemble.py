"""组装器：把「题目 + 金答案」变成契约形态的样本。

这是**编号的唯一出口**：打乱选项与「选项名 → 编号」的转换都只在这里发生。
管线其余部分一律使用选项名，从而把编号相关的 bug 压到最小面积。

输入：样本 id、state、LabeledQuestion 列表、RNG。
输出：契约形态的样本 dict（`id` / `state` / `questions` / `answers`）。
"""

from __future__ import annotations

import random
from typing import Any

from . import contract
from .workflow import LabeledQuestion


def assign_family_ids(labeled: list[LabeledQuestion]) -> list[str]:
    """为每道题分配唯一题号：同族第一次用族名，之后加 `_2`、`_3`……

    输入：LabeledQuestion 列表（按出题顺序）。
    输出：与输入等长的题号列表。
    预期行为：顺序稳定；不改变题目内容。
    """
    used: dict[str, int] = {}
    ids: list[str] = []
    for item in labeled:
        used[item.family] = used.get(item.family, 0) + 1
        seq = used[item.family]
        ids.append(item.family if seq == 1 else f"{item.family}_{seq}")
    return ids


def _shuffled_criteria(names: list[str], rng: random.Random) -> list[str]:
    """打乱选项顺序（只动顺序，不动名字）。

    输入：选项名列表、RNG。
    输出：打乱后的新列表。预期行为：返回新对象，不修改入参。
    """
    order = list(names)
    rng.shuffle(order)
    return order


def assemble_sample(
    sample_id: str,
    state: Any,
    labeled: list[LabeledQuestion],
    rng: random.Random,
) -> dict[str, Any]:
    """组装一条契约样本。

    输入：样本 id、state、题目列表、RNG（决定 choice 选项打乱）。
    输出：`{"id", "state", "questions", "answers"}`。
    预期行为：
    - choice：先按 `shuffle` 决定是否打乱 `criteria` 的 key 顺序，
      再用 `list(criteria).index(选项名)` 得到编号（编号 = key 的书写顺序）；
    - score：等级数组保持原序，答案直接用等级下标；
    - noul：答案直接是布尔；
    - 题号由 `assign_family_ids` 唯一化，`answers` 的 key 与 `questions` 完全一致。
    """
    questions: dict[str, Any] = {}
    answers: dict[str, Any] = {}
    for qid, item in zip(assign_family_ids(labeled), labeled, strict=True):
        qtype = item.question["type"]
        question = dict(item.question)
        if qtype == contract.TYPE_CHOICE:
            criteria = question.get("criteria")
            if not isinstance(criteria, dict):
                raise ValueError(f"choice 题目缺少 criteria：{qid}")
            order = _shuffled_criteria(list(criteria), rng) if item.shuffle else list(criteria)
            if not isinstance(item.answer, str):
                raise ValueError(f"choice 的 answer 必须是选项名（字符串）：{qid} -> {item.answer!r}")
            if item.answer not in criteria:
                raise ValueError(f"choice 的 answer 不在 criteria 里：{qid} -> {item.answer!r}")
            question["criteria"] = {name: criteria[name] for name in order}
            answers[qid] = {"type": contract.TYPE_CHOICE, "choice": order.index(item.answer)}
        elif qtype == contract.TYPE_SCORE:
            if not contract.is_index(item.answer):
                raise ValueError(f"score 的 answer 必须是等级下标（整数）：{qid} -> {item.answer!r}")
            answers[qid] = {"type": contract.TYPE_SCORE, "score": int(item.answer)}
        elif qtype == contract.TYPE_NOUL:
            if not isinstance(item.answer, bool):
                raise ValueError(f"noul 的 answer 必须是布尔：{qid} -> {item.answer!r}")
            answers[qid] = {"type": contract.TYPE_NOUL, "noul": item.answer}
        else:
            raise ValueError(f"未知题型：{qtype!r}（题号 {qid}）")
        questions[qid] = question
    return {"id": sample_id, "state": state, "questions": questions, "answers": answers}
