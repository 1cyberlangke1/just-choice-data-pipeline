"""内容审核的特征层测试：判据、诱饵、标签两条路径与计划。"""

from __future__ import annotations

import pytest

from tests.workflows.guardrail.conftest import text_of
from workflows.guardrail import features


def test_every_clause_appears_eight_times() -> None:
    """输入：无。输出：无。

    预期行为：每条条款在 16 组里各出现 8 次——这是 `clause` 那一族的答案按构造就是 50% 的原因。
    """
    for clause in features.CLAUSE_IDS:
        hits = sum(1 for value in features.HIT_SETS if clause in features.parse_hits(value))
        assert hits == 8, clause


def test_hit_set_is_purely_literal() -> None:
    """输入：一段正文。输出：无。预期行为：命中组合按 `CLAUSE_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ("联系电话 13812345678", ("P1",)),
        ("见 http://shop.example.com/order", ("P2",)),
        ("见 www.example.com 上的说明", ("P2",)),
        ("有问题发 liu.wei@example.com", ("P3",)),
        ("太棒了！！！", ("P4",)),
        ("这一行里什么形状都没有", ()),
        ("138-1234-5678", ()),  # 分段号码：每一段都不足 7 位
        ("2026-09-29 开会", ()),  # 日期也不该被当成命中
        ("13812345678 和 http://a.example.com", ("P1", "P2")),
        ("打 13812345678 或来信 liu.wei@example.com ！！！", ("P1", "P3", "P4")),
    ]:
        try:
            assert features.hit_set(text) == expect
            assert features.hits_of(text) == "+".join(expect)
            assert features.parse_hits(features.hits_of(text)) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_clause_boundaries_are_pinned() -> None:
    """输入：贴着判据边界的写法。输出：无。

    预期行为：「差一点」与「刚好够」判成不同结果。变异测试改宽一位（把 P1 的 `{7,}` 改成
    `{8,}`）时这里必须变红——只跑 `13812345678` 这种宽裕样例，边界改了也不会有人发现。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ("编号 123456", ()),  # P1：6 位，差一位不算命中
        ("编号 1234567", ("P1",)),  # P1：7 位就是边界，算命中
        ("见 http:/shop.example.com", ()),  # P2：少一个斜杠不算
        ("见 shop.example.com 的说明", ()),  # P2：没有 www. 不算
        ("发 liu.wei.example.com", ()),  # P3：没有 @ 不算
        ("太棒了！！", ()),  # P4：两个感叹号不算
        ("太棒了！！！", ("P4",)),  # P4：三个就是边界，算命中
    ]:
        try:
            assert features.hit_set(text) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_decoy_example_is_a_decoy_and_not_a_hit() -> None:
    """输入：无。输出：无。

    预期行为：提示词里逐字给出的**每一个**号码都既匹配诱饵形状，又**不**触发任何条款——这两件事
    缺一条，`decoy=dashed` 那一档就废了（要么不算诱饵，要么直接变成真命中）。池子里的值是要逐个
    发出去的，只验第一个等于没验。
    """
    assert features.DECOY_EXAMPLE == features.DECOY_POOL[0]
    assert len(set(features.DECOY_POOL)) == len(features.DECOY_POOL)
    for decoy in features.DECOY_POOL:
        assert features.DECOY_PATTERN.search(decoy), decoy
        assert features.has_decoy(decoy), decoy
        assert features.hit_set(decoy) == (), decoy


def test_pick_is_stable_and_uses_the_whole_pool() -> None:
    """输入：一组种子文本。输出：无。

    预期行为：`pick` 是纯函数（同种子同结果，换个种子换结果），而且**池子会被用满**——只抽到
    第一项的「池子」等于没换，复读问题原样回来。
    """
    pool = ("a", "b", "c", "d", "e")
    first = [features.pick(pool, f"seed-{index}", "P1") for index in range(50)]
    assert all(features.pick(pool, f"seed-{index}", "P1") == first[index] for index in range(50))
    assert set(first) == set(pool)
    assert features.decoy_for("seed-0") == features.pick(features.DECOY_POOL, "seed-0", "decoy")


@pytest.mark.parametrize("text", ["2026-09-29", "版本 1-2", "12-345", "138 1234 5678"])
def test_decoy_needs_three_digit_groups(text: str) -> None:
    """输入：几种「像但不是」诱饵的写法。输出：无。

    预期行为：日期、两位数版本号、空格分段的号码都不算诱饵——判据要求三段、每段 3~4 位。
    """
    assert not features.has_decoy(text)


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且与真值只差一两处
    （差得越少越难分辨，题目才有区分度）。
    """
    case = features.GuardrailCase("P1+P3", features.DECOY_NONE)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.hits == ("P1", "P3")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.hits for combo in plan.distractors)
    assert all(len(set(combo) ^ set(plan.hits)) <= 2 for combo in plan.distractors)
    assert plan.distractors == tuple(sorted(plan.distractors, key=lambda combo: (len(combo), combo)))


@pytest.mark.parametrize(
    ("count", "expect"),
    [(0, "pass"), (1, "flag"), (2, "review"), (3, "block"), (4, "block")],
)
def test_action_level_thresholds(count: int, expect: str) -> None:
    """输入：命中条数。输出：无。预期行为：0/1/2/3 条以上 分别对应四个档位。"""
    assert features.action_level(count) == expect


def test_two_label_paths_agree_on_every_faithful_text() -> None:
    """输入：全空间的 32 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——
    正文兑现了计划，两条路就不该打架。
    """
    cases = features.build_cases(512)
    assert len(cases) == 32
    for case in cases:
        text = text_of(case)
        assert features.hits_of(text) == case.hits, case.describe()
        assert features.observed_labels(text).hits == features.label_for(
            features.plan_for(case, 0)
        ).hits
        assert features.has_decoy(text) == (case.decoy == features.DECOY_DASHED)


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.GuardrailCase("", features.DECOY_NONE)
    assert case.describe() == "无命中-none"
    assert case.fingerprint() == "hits=|decoy=none"
    assert case.clauses() == ()
    assert features.parse_hits("") == ()
