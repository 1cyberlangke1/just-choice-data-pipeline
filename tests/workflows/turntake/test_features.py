"""特征层测试：判据只认字面、命中组合按构造均衡、两条标签路径互不打架。"""

from __future__ import annotations

import pytest

from tests.workflows.turntake.conftest import text_of
from workflows.turntake import features


def test_hit_set_is_purely_literal() -> None:
    """输入：一段用户输入。输出：无。预期行为：命中组合按 `TAG_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ("我今天去超市买了点东西，然后", ("T1",)),
        ("i went to the store and bought stuff, then", ("T1",)),
        ("我想问一下，", ("T2",)),
        ("我还想问，", ("T2",)),
        ("那个方案我看了（还没看完", ("T3",)),
        ("the plan is fine (i think", ("T3",)),
        ("等一下，让我想想", ("T4",)),
        ("hold on, let me think", ("T4",)),
        # 这四条是判据「只认字面」的证据：说完了的输入一条都不该踩。
        ("好的，我明白了，谢谢", ()),
        ("这个方案我看完了，没问题。", ()),
        ("ok thanks, that is all", ()),
        ("the plan is fine (i think so)", ()),
        # 两条同时成立：连接词后面再跟一个逗号。
        ("我想说的是，然后，", ("T1", "T2")),
    ]:
        try:
            assert features.hit_set(text) == expect
            assert features.hits_of(text) == "+".join(expect)
            assert features.parse_hits(features.hits_of(text)) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_channels_are_three_distinct_shapes() -> None:
    """输入：无。输出：无。

    预期行为：三个通道键两两不同，且**全空间里每种通道都被枚举到**——留出维只砍取值，
    不该把别的取值一起砍掉（`build_cases` 默认维度表覆盖 `CHANNELS`）。
    """
    assert features.CHANNELS == ("im", "voice", "note")
    assert len(set(features.CHANNELS)) == len(features.CHANNELS)
    assert features.DIMS["channel"] == features.CHANNELS
    cases = features.build_cases(512)
    assert {case.channel for case in cases} == set(features.CHANNELS)


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且与真值只差一两处
    （差得越少越难分辨，题目才有区分度）。
    """
    case = features.TurnTakeCase("T1+T3", features.CHANNEL_IM)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.hits == ("T1", "T3")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.hits for combo in plan.distractors)
    assert all(len(set(combo) ^ set(plan.hits)) <= 2 for combo in plan.distractors)
    assert plan.distractors == tuple(
        sorted(plan.distractors, key=lambda combo: (len(combo), combo))
    )


@pytest.mark.parametrize(
    ("count", "expect"),
    [(0, "answer"), (1, "pause"), (2, "wait"), (3, "check"), (4, "check")],
)
def test_act_level_thresholds(count: int, expect: str) -> None:
    """输入：命中条数。输出：无。预期行为：0/1/2/3 条以上 分别对应四个档位。"""
    assert features.act_level(count) == expect


def test_two_label_paths_agree_on_every_faithful_text() -> None:
    """输入：全空间的 48 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——
    输入兑现了计划，两条路就不该打架。
    """
    cases = features.build_cases(512)
    assert len(cases) == 48  # 16 组命中 × 3 种通道
    for case in cases:
        text = text_of(case.tags())
        assert features.hits_of(text) == case.hits, case.describe()
        assert features.observed_labels(text).hits == features.label_for(
            features.plan_for(case, 0)
        ).hits


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.TurnTakeCase("", features.CHANNEL_IM)
    assert case.describe() == "无命中-im"
    assert case.fingerprint() == "hits=|channel=im"
    assert case.tags() == ()
    assert features.parse_hits("") == ()
