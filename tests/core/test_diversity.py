"""产物文本多样性度量的测试（通用层，与具体 workflow 无关）。

输入：`jc.diversity` 的纯函数。输出：断言结果。
预期行为：复读会被两把尺子同时抓到；空输入 / 短文本不假装满分；长度三档算得对。
"""

from __future__ import annotations

from jc import diversity


def test_tokens_drops_whitespace_only() -> None:
    """输入：带换行与空格的文本。输出：无。预期行为：只去空白，标点与大小写原样。"""
    assert diversity.tokens("A, b\n c\tD") == "A,bcD"


def test_ngrams_needs_enough_characters() -> None:
    """输入：长度不足 / 刚够的文本。输出：无。预期行为：不足给空集，刚够给一条。"""
    assert diversity.ngrams("ab", 3) == frozenset()
    assert diversity.ngrams("abc", 3) == frozenset({"abc"})


def test_distinct_n_is_one_when_every_gram_is_new() -> None:
    """输入：两条毫无重合的文本。输出：无。预期行为：distinct_2 恰好 1.0。"""
    assert diversity.distinct_n(["甲乙丙丁", "ABCD"], 2) == 1.0


def test_distinct_n_drops_when_the_same_sentence_repeats() -> None:
    """输入：同一句话抄四遍。输出：无。预期行为：distinct_2 塌到 1/4。"""
    assert diversity.distinct_n(["今天天气不错"] * 4, 2) == 0.25


def test_distinct_n_of_nothing_is_zero() -> None:
    """输入：空批 / 全短文本。输出：无。预期行为：0.0，不当满分。"""
    assert diversity.distinct_n([], 2) == 0.0
    assert diversity.distinct_n(["a", "b"], 2) == 0.0


def test_self_overlap_catches_the_worst_pair() -> None:
    """输入：三条里有一对几乎一样。输出：无。

    预期行为：报最坏的那一对（1.0），不被另外两条差异大的稀释。
    """
    texts = ["甲乙丙丁戊己", "甲乙丙丁戊己", "完全不同的另一段文字"]
    assert diversity.self_overlap(texts) == 1.0


def test_self_overlap_is_zero_for_one_text_or_no_common_grams() -> None:
    """输入：只有一条 / 两条毫无重合。输出：无。预期行为：0.0（没有可比的一对）。"""
    assert diversity.self_overlap(["只有一条"]) == 0.0
    assert diversity.self_overlap(["甲乙丙丁戊己", "ABCDEFGH"]) == 0.0


def test_summarize_counts_lengths_and_survives_empty_input() -> None:
    """输入：一批文本 / 空批。输出：无。预期行为：长度三档按去空白后的字符数算，空批全 0。"""
    table = diversity.summarize(["ab cd", "ef"])
    assert table["n"] == 2
    assert table["length_mean"] == 3.0
    assert table["length_min"] == 2
    assert table["length_max"] == 4
    empty = diversity.summarize([])
    assert empty == {
        "n": 0,
        "distinct_1": 0.0,
        "distinct_2": 0.0,
        "self_overlap": 0.0,
        "length_mean": 0.0,
        "length_min": 0,
        "length_max": 0,
    }
