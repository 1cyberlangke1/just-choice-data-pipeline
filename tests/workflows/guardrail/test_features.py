"""内容审核的特征层测试：判据、诱饵、标签两条路径与计划。"""

from __future__ import annotations

import pytest

from tests.workflows.guardrail.conftest import text_of
from workflows.guardrail import features


def test_hit_sets_are_all_sixteen_combinations() -> None:
    """输入：无。输出：无。预期行为：16 组、空串在前、全命中在最后、顺序稳定。"""
    assert len(features.HIT_SETS) == 16
    assert features.HIT_SETS[0] == ""
    assert features.HIT_SETS[1] == "P1"
    assert features.HIT_SETS[-1] == "P1+P2+P3+P4"
    assert len(set(features.HIT_SETS)) == 16


def test_every_clause_appears_eight_times() -> None:
    """输入：无。输出：无。

    预期行为：每条条款在 16 组里各出现 8 次——这是 `clause` 那一族的答案按构造就是 50% 的原因。
    """
    for clause in features.CLAUSE_IDS:
        hits = sum(1 for value in features.HIT_SETS if clause in features.parse_hits(value))
        assert hits == 8, clause


@pytest.mark.parametrize(
    ("text", "expect"),
    [
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
    ],
)
def test_hit_set_is_purely_literal(text: str, expect: tuple[str, ...]) -> None:
    """输入：一段正文。输出：无。预期行为：命中组合按 `CLAUSE_IDS` 顺序、只看字面。"""
    assert features.hit_set(text) == expect
    assert features.hits_of(text) == "+".join(expect)
    assert features.parse_hits(features.hits_of(text)) == expect


def test_decoy_example_is_a_decoy_and_not_a_hit() -> None:
    """输入：无。输出：无。

    预期行为：提示词里逐字给出的那个号码既匹配诱饵形状，又**不**触发 P1——这两件事缺一条，
    `decoy=dashed` 那一档就废了（要么不算诱饵，要么直接变成真命中）。
    """
    assert features.DECOY_PATTERN.search(features.DECOY_EXAMPLE)
    assert features.has_decoy(features.DECOY_EXAMPLE)
    assert features.hit_set(features.DECOY_EXAMPLE) == ()


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


def test_build_cases_order_is_stable() -> None:
    """输入：同一份维度表两次。输出：无。预期行为：顺序逐格一致、覆盖全部命中组合。"""
    first = features.build_cases(512)
    second = features.build_cases(512)
    assert [case.as_dict() for case in first] == [case.as_dict() for case in second]
    assert {case.hits for case in first} == set(features.HIT_SETS)


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.GuardrailCase("", features.DECOY_NONE)
    assert case.describe() == "无命中-none"
    assert case.fingerprint() == "hits=|decoy=none"
    assert case.clauses() == ()
    assert features.parse_hits("") == ()
