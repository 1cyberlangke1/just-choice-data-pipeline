"""一致性校验：同一段素材上的同一道题，只能有一个答案。

照抄 Open-Jev 的 `validate_records`（见 docs 10.2 第 4 条）：**素材相同而标签不同**的两条
样本是自相矛盾的训练对——模型只会被这一对互相抵消。构造式数据天然不会出现（每条素材都唯一，
本项目实测 4800 条 maze / 1425 条 invoice / 642 条 hybrid 零重复），但 **LLM 渲染的数据会**：
模型完全可能把两套不同的注入特征写成同一段文字，那时这段文字就对应两个答案。

输入：合格样本列表（契约形态）。输出：冲突列表。
预期行为：只比**素材文本完全相同**的样本（不同素材上的同题不同答是正常的）；
比对用「问法 + 选项文字集合 + 金答案文字」，**不比编号**——编号是组装器洗过的，本来就该变。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .contract import gold_text

#: 冲突记录里带上多少素材原文（给人查线索用；完整素材在产物 jsonl 里）。
STATE_PREVIEW: int = 160


def state_text(state: Any) -> str:
    """输入：样本的 state。输出：可比文本。

    预期行为：非字符串（对象 / 数组形态的 state）用 JSON 序列化后再比——
    契约允许 state 是任意 JSON，比对要能把「同一个对象」认成同一个。
    """
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)


def question_key(
    question: dict[str, Any], answer: dict[str, Any]
) -> tuple[str, str, tuple[str, ...], str] | None:
    """输入：一道题、它的答案。输出：这道题的可比指纹；形态不对时 None。

    预期行为：指纹 = (题型, 问法, 排序后的选项文字, 金答案文字)。
    选项取文字而不是选项名（`m1` 这种槽名是 workflow 的内部编号）；
    排序是为了让「选项顺序被洗过」的两条样本仍然可比。
    """
    qtype = question.get("type")
    instructions = question.get("instructions")
    if not isinstance(qtype, str) or not isinstance(instructions, str):
        return None
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        options = tuple(sorted(str(value) for value in criteria.values()))
    elif isinstance(criteria, list):
        options = tuple(sorted(str(value) for value in criteria))
    else:
        options = ()
    return (qtype, instructions, options, gold_text(question, answer))


def conflicts(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """输入：合格样本序列。输出：冲突列表（按首次发现顺序，每个冲突点一条）。

    预期行为：把样本按「state 文本 + 题型 + 问法 + 选项文字」归堆，**一堆里出现两种以上
    金答案文字**就是一条冲突。一个冲突点只记一条（带每种答案各自的样本 id 样例），
    而不是每条相撞的样本对记一条——否则同一段素材有 200 条样本时会刷出上千行。
    记录里带 state 的 sha256 前 12 位（当键用）与素材原文片段（给人查）。
    """
    # probe -> {金答案文字: [样本 id，最多两个]}；`order` 保住首次出现的顺序
    table: dict[tuple[str, str, str, tuple[str, ...]], dict[str, list[str]]] = {}
    order: list[tuple[str, str, str, tuple[str, ...]]] = []
    meta: dict[tuple[str, str, str, tuple[str, ...]], tuple[str, str]] = {}
    for sample in samples:
        text = state_text(sample.get("state"))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        sample_id = str(sample.get("id", ""))
        for qid, question in sample.get("questions", {}).items():
            key = question_key(question, sample.get("answers", {}).get(qid, {}))
            if key is None:
                continue
            qtype, instructions, options, gold = key
            probe = (text, qtype, instructions, options)
            bucket = table.setdefault(probe, {})
            if not bucket:
                order.append(probe)
                meta[probe] = (digest, text[:STATE_PREVIEW])
            ids = bucket.setdefault(gold, [])
            if len(ids) < 2:
                ids.append(sample_id)
    out: list[dict[str, Any]] = []
    for probe in order:
        bucket = table[probe]
        if len(bucket) < 2:
            continue
        digest, preview = meta[probe]
        out.append(
            {
                "state_sha": digest,
                "question": probe[2],
                "type": probe[1],
                "options": list(probe[3]),
                "answers": {gold: ids for gold, ids in bucket.items()},
                "state_preview": preview,
            }
        )
    return out
