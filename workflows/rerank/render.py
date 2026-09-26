"""检索重排（rerank）workflow 的渲染：把真值规格写成一屏「查询 + 候选短名单」。

输入：`RerankSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「抬头（含查询号）→ 查询一行 → 候选若干块（每块：一行表头 + 若干行正文）」。

关键约束：每条候选正文里写出来的主题词与方面词**必须**能判出「与查询一致吗」——
所以「另一个主题 / 另一个方面」的挑法要保证**永远不等于**查询的那一个（见 `other_topic`）。
"""

from __future__ import annotations

from typing import Any, Final

from . import spec as spec_mod
from .spec import RerankCase, RerankSpec

#: 候选编号（顺序即 state 里的编号顺序，也是判别器认编号的词汇表）。
CAND_IDS: Final[tuple[str, ...]] = ("C1", "C2", "C3", "C4")


def cand_id(index: int) -> str:
    """输入：候选下标（0-based）。输出：候选编号（`C1`~`C4`）。"""
    return CAND_IDS[index]


def other_topic(topic: str, index: int) -> str:
    """输入：查询主题键、候选下标。输出：**另一个**主题键。

    预期行为：偏移量取 `index % (len(TOPICS) - 1) + 1`，落在 1..4，模 8 后绝不为 0，
    所以「另一个主题」永远撞不上查询主题——这是「主题不一致」那条判据能被读出来的前提。
    """
    offset = index % (len(spec_mod.TOPICS) - 1) + 1
    return spec_mod.TOPICS[(spec_mod.TOPICS.index(topic) + offset) % len(spec_mod.TOPICS)]


def other_aspect(aspect: str, index: int) -> str:
    """输入：查询方面键、候选下标。输出：**另一个**方面键。

    预期行为：同 `other_topic`，偏移量落在 1..2，模 3 后绝不为 0。
    """
    offset = index % (len(spec_mod.ASPECTS) - 1) + 1
    return spec_mod.ASPECTS[(spec_mod.ASPECTS.index(aspect) + offset) % len(spec_mod.ASPECTS)]


def cand_words(case: RerankCase, index: int) -> tuple[str, str]:
    """输入：case、候选下标。输出：这条候选正文里写的 (主题键, 方面键)。

    预期行为：主题 / 方面是否与查询一致，完全由画像里的两个布尔位决定；
    位为假时挑「另一个」词，保证读出来就是不等于查询那一个。
    """
    cand = case.shape.cands[index]
    topic = case.topic if cand.topic_ok else other_topic(case.topic, index)
    aspect = case.aspect if cand.aspect_ok else other_aspect(case.aspect, index)
    return topic, aspect


def ordered_traps(case: RerankCase) -> tuple[str, ...]:
    """输入：case。输出：按 `TRAPS_KEYS` 顺序归一后的噪声句键。"""
    return tuple(key for key in spec_mod.TRAPS_KEYS if key in case.traps)


def cand_lines(case: RerankCase, index: int, loc: dict[str, Any]) -> tuple[str, ...]:
    """输入：case、候选下标、语言字典。输出：这条候选的正文各行。

    预期行为：主题/方面一句 + 噪声句若干 + 示例一句（示例放最后，像一条收尾备注）。
    措辞（`case.wording`）只换模板，说的还是同一件事。
    """
    topic, aspect = cand_words(case, index)
    wording = case.wording
    lines = [
        loc["sentence"]["body"][wording].format(
            topic=loc["topic"][topic], aspect=loc["aspect"][aspect]
        )
    ]
    lines.extend(loc["sentence"]["trap"][key] for key in ordered_traps(case))
    example = "yes" if case.shape.cands[index].example else "no"
    lines.append(loc["sentence"]["example"][wording][example])
    return tuple(lines)


def build(spec: RerankSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整屏重排请求。

    预期行为：渲染器不做任何判断——相关度与名次全在 `spec` 的规则里算好了。
    """
    meta = loc["meta"]
    colon = meta["colon"]
    sep = meta["sep"]
    case = spec.case
    query = loc["query"]["text"].format(
        topic=loc["topic"][case.topic], aspect=loc["aspect"][case.aspect]
    )
    lines = [f"{meta['title']} {spec.query_id}"]
    lines.append(f"{meta['query_label']}{colon}{query}")
    lines.append(f"{meta['cand_label']}{colon}")
    for index, cid in enumerate(CAND_IDS):
        cand = case.shape.cands[index]
        header = (
            f"{cid}{sep}"
            f"{meta['source_label']}{colon}{loc['source'][cand.source]}{sep}"
            f"{meta['day_label']}{colon}{cand.day}"
        )
        lines.append(header)
        lines.extend(f"   {line}" for line in cand_lines(case, index, loc))
    return "\n".join(lines)
