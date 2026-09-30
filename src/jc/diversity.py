"""产物文本的多样性度量：把「这批数据是不是一坨复读」变成能核的数字。

输入：一批文本（逐条样本里模型写的自然语言正文，见 `Workflow.language_texts`）。
输出：`{n, distinct_1, distinct_2, self_overlap, length_mean, length_min, length_max}`。
预期行为：纯函数、零 token、不联网——判据是现算的，不是嘴上说的。

两把尺子：

- `distinct_n`：全部 n-gram 里**不同**的那些占多少。1.0 = 每个 n-gram 都只出现一次；
  同一句话被反复抄会把它压低。短文本天然偏低，所以它只适合横向比同一批里的不同字段。
- `self_overlap`：两两文本之间 n-gram 集合的 Jaccard **最大值**。均值会被长尾稀释，而
  「有一对几乎一模一样」才是真问题，所以这里报最坏的那一对。

切词一律按**字符**（中文没有空格，英文走同一把尺子才好跨语言比），空白字符不计入。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

from .jsontypes import JsonValue

#: 度量用的 n-gram 阶数：1 看用词、2 看搭配。再往上就只剩模板句的骨架了。
DEFAULT_N: Final[int] = 2

#: 算两两重合度用的阶数：3 比 2 稳，短句里 2-gram 太容易撞。
OVERLAP_N: Final[int] = 3


def tokens(text: str) -> str:
    """输入：一段文本。输出：去掉所有空白字符后的文本。

    预期行为：只做「空白不算词」这一件事，不转大小写、不去标点——标点也是模型的选择，
    把它抹掉会让「每句都用同一个句式」这种复读看起来更干净。
    """
    return "".join(ch for ch in text if not ch.isspace())


def ngrams(text: str, n: int) -> frozenset[str]:
    """输入：一段文本、阶数。输出：它的字符 n-gram 集合。

    预期行为：文本短于 n 时返回空集（这种文本不参与「谁和谁像」的判断，别让它污染结果）。
    """
    plain = tokens(text)
    if n <= 0 or len(plain) < n:
        return frozenset()
    return frozenset(plain[i : i + n] for i in range(len(plain) - n + 1))


def distinct_n(texts: Iterable[str], n: int = DEFAULT_N) -> float:
    """输入：一批文本、阶数。输出：不同 n-gram 数 / 总 n-gram 数（0.0 ~ 1.0）。

    预期行为：一条都没有（或都短于 n）时返回 0.0——「没东西可测」不当成满分。
    """
    total = 0
    seen: set[str] = set()
    for text in texts:
        grams = ngrams(text, n)
        total += len(grams)
        seen |= grams
    if total == 0:
        return 0.0
    return len(seen) / total


def self_overlap(texts: Sequence[str], n: int = OVERLAP_N) -> float:
    """输入：一批文本、阶数。输出：两两 Jaccard 重合度的最大值（0.0 ~ 1.0）。

    预期行为：不足两条文本时返回 0.0；任一条短于 n 就跳过这一对（空集与任何集合的
    Jaccard 都算 0，跳过比算 0 更接近「没得比」）。
    """
    sets = [ngrams(text, n) for text in texts]
    worst = 0.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            left, right = sets[i], sets[j]
            if not left or not right:
                continue
            union = len(left | right)
            if union:
                worst = max(worst, len(left & right) / union)
    return worst


def summarize(texts: Sequence[str]) -> dict[str, JsonValue]:
    """输入：一批文本。输出：一张度量表（条数、两条 distinct、最坏重合度、长度三档）。

    预期行为：空输入也给得出表（全 0），调用方不用自己判空。
    """
    lengths = [len(tokens(text)) for text in texts]
    return {
        "n": len(texts),
        "distinct_1": round(distinct_n(texts, 1), 4),
        "distinct_2": round(distinct_n(texts, DEFAULT_N), 4),
        "self_overlap": round(self_overlap(texts), 4),
        "length_mean": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
    }
