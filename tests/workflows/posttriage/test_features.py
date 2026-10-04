"""社交帖子分诊的特征层测试：判据、诱饵、标签两条路径与计划。"""

from __future__ import annotations

import pytest

from tests.workflows.posttriage.conftest import text_of
from workflows.posttriage import features


def test_signal_sets_are_all_sixteen_combinations() -> None:
    """输入：无。输出：无。预期行为：16 组、空串在前、全命中在最后、顺序稳定。"""
    assert len(features.SIGNAL_SETS) == 16
    assert features.SIGNAL_SETS[0] == ""
    assert features.SIGNAL_SETS[1] == "S1"
    assert features.SIGNAL_SETS[-1] == "S1+S2+S3+S4"
    assert len(set(features.SIGNAL_SETS)) == 16


def test_every_signal_appears_eight_times() -> None:
    """输入：无。输出：无。

    预期行为：每条判据在 16 组里各出现 8 次——这是 `flag` 那一族的答案按构造就是 50% 的原因。
    """
    for signal in features.SIGNAL_IDS:
        hits = sum(1 for value in features.SIGNAL_SETS if signal in features.parse_signals(value))
        assert hits == 8, signal


def test_signal_set_is_purely_literal() -> None:
    """输入：一段帖子。输出：无。预期行为：命中组合按 `SIGNAL_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ("关注 $NVDA 这条线", ("S1",)),
        ("$A 只有一个字母，$ABCDEF 有六个", ()),  # 长度不在 2~5 之间：两条都不算
        ("现价 512.30", ("S2",)),
        ("涨了 7.5%", ("S3",)),
        ("只涨 8%", ("S3",)),
        ("一位小数的 123.4 不算价位", ()),
        ("见 https://example.com/chart", ("S4",)),
        ("见 www.example.com 上的讨论", ("S4",)),
        ("这一行里什么形状都没有", ()),
        ("$acme", ()),  # 小写 cashtag：看着像代码，按判据只认大写
        ("$NVDA 现价 512.30", ("S1", "S2")),
        ("12.50% 同时是两位小数与百分数", ("S2", "S3")),
        ("$NVDA 涨 7.5% 见 https://example.com", ("S1", "S3", "S4")),
    ]:
        try:
            assert features.signal_set(text) == expect
            assert features.signals_of(text) == "+".join(expect)
            assert features.parse_signals(features.signals_of(text)) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_lowercase_cashtag_is_not_a_hit() -> None:
    """输入：无。输出：无。

    预期行为：大写与小写只差一个字母，命中与否正好相反——这一条是 test_ood 那一档的全部意义，
    写成「见 $ 就当代码」的捷径会在这里当场露馅。
    """
    assert features.signal_set("$ACME") == ("S1",)
    assert features.signal_set("$acme") == ()


def test_decoy_example_is_a_decoy_and_not_a_hit() -> None:
    """输入：无。输出：无。

    预期行为：提示词里逐字给出的那个写法既匹配诱饵形状，又**不**触发 S1——这两件事缺一条，
    `decoy=lower` 那一档就废了（要么不算诱饵，要么直接变成真命中）。
    """
    assert features.DECOY_PATTERN.search(features.DECOY_EXAMPLE)
    assert features.has_decoy(features.DECOY_EXAMPLE)
    assert features.signal_set(features.DECOY_EXAMPLE) == ()


@pytest.mark.parametrize("text", ["$ACME", "$a", "$abcdef", "acme", "$1ab"])
def test_decoy_needs_lowercase_letters(text: str) -> None:
    """输入：几种「像但不是」诱饵的写法。输出：无。

    预期行为：大写、单字母、超过五个字母、没有 $ 的都不算诱饵——判据要求 $ 后面跟 2~5 个小写
    字母，写宽一点就会把真命中的大写 cashtag 也算成诱饵。
    """
    assert not features.has_decoy(text)


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且与真值只差一两处
    （差得越少越难分辨，题目才有区分度）。
    """
    case = features.PostTriageCase("S1+S3", features.DECOY_NONE)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.signals == ("S1", "S3")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.signals for combo in plan.distractors)
    assert all(len(set(combo) ^ set(plan.signals)) <= 2 for combo in plan.distractors)
    assert plan.distractors == tuple(
        sorted(plan.distractors, key=lambda combo: (len(combo), combo))
    )


@pytest.mark.parametrize(
    ("count", "expect"),
    [(0, "ignore"), (1, "watch"), (2, "report"), (3, "escalate"), (4, "escalate")],
)
def test_route_level_thresholds(count: int, expect: str) -> None:
    """输入：命中条数。输出：无。预期行为：0/1/2/3 条以上 分别对应四个档位。"""
    assert features.route_level(count) == expect


def test_two_label_paths_agree_on_every_faithful_text() -> None:
    """输入：全空间的 32 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——
    正文兑现了计划，两条路就不该打架。
    """
    cases = features.build_cases(512)
    assert len(cases) == 32
    for case in cases:
        text = text_of(case)
        assert features.signals_of(text) == case.signals, case.describe()
        assert features.observed_labels(text).signals == features.label_for(
            features.plan_for(case, 0)
        ).signals
        assert features.has_decoy(text) == (case.decoy == features.DECOY_LOWER)


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.PostTriageCase("", features.DECOY_NONE)
    assert case.describe() == "无信号-none"
    assert case.fingerprint() == "signals=|decoy=none"
    assert case.hits() == ()
    assert features.parse_signals("") == ()
