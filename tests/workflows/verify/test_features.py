"""完成声明核验的特征层：维度表、禁用对、四条判据。"""

from __future__ import annotations

from collections import Counter
from itertools import product

import pytest

from jc.cover import missing_pairs
from workflows.verify import features
from workflows.verify.features import Facts, TagSet


def _shares(values: list[str | int | bool]) -> dict[str, float]:
    """输入：一串取值。输出：{取值: 占比百分数}（保留一位，按取值名排序）。"""
    counter = Counter(values)
    return {
        str(key): round(count / len(values) * 100, 1)
        for key, count in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def test_full_space_is_the_cartesian_product_minus_forbidden_pairs() -> None:
    """输入：无。输出：无。

    预期行为：全枚举结果**恰好**是笛卡尔积剔掉两个禁用对之后的 300 组——本 workflow
    的约束只有那两对，这条把「没有别的隐形约束」钉死。
    """
    cases = features.build_cases(1000)
    allowed = {
        row
        for row in product(
            features.CLAIM_KINDS,
            features.TONES,
            features.CHECKS,
            features.COVERAGES,
            features.ARTIFACTS,
        )
        if features.allow_pair("checks", row[2], "coverage", row[3])
    }
    assert len(cases) == 300
    assert {tuple(case.as_dict().values()) for case in cases} == allowed


def test_forbidden_pairs_only_touch_checks_and_coverage() -> None:
    """输入：无。输出：无。

    预期行为：禁用对只有两对、且都发生在 `checks` 与 `coverage` 之间——OOD 留出别的维度
    时才不用像 shell 那样把另一维收窄（见 `workflow.ood_dims` 的说明）。
    """
    dims = {dim for pair in features.FORBIDDEN for dim, _ in pair}
    assert dims == {"checks", "coverage"}
    assert len(features.FORBIDDEN) == 2
    assert all(len(pair) == 2 for pair in features.FORBIDDEN)


def test_pairwise_cover_still_covers_every_pair() -> None:
    """输入：无。输出：无。

    预期行为：把上限压到全空间以下时走成对覆盖，行数远少于 300，但任意两维的每个取值对
    至少出现一次——这是 `jc.cover` 在本 workflow 的维度表上的交叉验证。
    """
    cases = features.build_cases(48)
    assert 0 < len(cases) < 300
    assert not missing_pairs(
        features.DIMS, [case.as_dict() for case in cases], allow_pair=features.allow_pair
    )


def test_build_cases_is_deterministic() -> None:
    """输入：同一上限两次。输出：无。预期行为：逐组一致（case 库要能逐字节重建）。"""
    assert [c.as_dict() for c in features.build_cases(48)] == [
        c.as_dict() for c in features.build_cases(48)
    ]


def test_forbidden_pairs_never_show_up() -> None:
    """输入：无。输出：无。预期行为：枚举出来的每一组都相容（没跑检查就不会声称覆盖）。"""
    for case in features.build_cases(1000):
        assert features.compatible(case)
        assert features.allow_pair("checks", case.checks, "coverage", case.coverage)


@pytest.mark.parametrize(
    ("checks", "coverage", "expected"),
    [
        ("all_fail", "covered", "refuted"),
        ("all_fail", "uncovered", "refuted"),
        ("not_run", "uncovered", "unverified"),
        ("all_pass", "uncovered", "unverified"),
        ("partial_pass", "uncovered", "unverified"),
        ("all_pass", "covered", "verified"),
        ("all_pass", "partial", "partial"),
        ("partial_pass", "covered", "partial"),
        ("partial_pass", "partial", "partial"),
    ],
)
def test_verdict_rules(checks: str, coverage: str, expected: str) -> None:
    """输入：自报的检查与覆盖。输出：无。预期行为：结论与文档里那条判据一致。"""
    assert features.verdict_from(Facts(checks, coverage, "committed")) == expected


@pytest.mark.parametrize(
    ("checks", "coverage", "artifact", "expected"),
    [
        ("not_run", "uncovered", "committed", "run_the_checks"),
        ("not_run", "uncovered", "uncommitted", "run_the_checks"),
        ("all_fail", "covered", "committed", "fix_the_failing_checks"),
        ("partial_pass", "covered", "committed", "fix_the_failing_checks"),
        ("all_pass", "partial", "committed", "tighten_scope"),
        ("all_pass", "uncovered", "uncommitted", "tighten_scope"),
        ("all_pass", "covered", "uncommitted", "commit_the_change"),
        ("all_pass", "covered", "committed", "nothing_missing"),
    ],
)
def test_gap_rules(checks: str, coverage: str, artifact: str, expected: str) -> None:
    """输入：自报事实。输出：无。

    预期行为：按「检查 → 覆盖 → 落盘」取第一个没过的关卡——`partial_pass` 也算检查没过
    （只过一部分同样得回去修），所以它落到 `fix_the_failing_checks` 而不是「什么都不缺」。
    """
    assert features.gap_from(Facts(checks, coverage, artifact)) == expected


@pytest.mark.parametrize(
    ("checks", "coverage", "artifact", "expected"),
    [
        ("all_pass", "covered", "committed", 3),
        ("all_pass", "covered", "uncommitted", 2),
        ("partial_pass", "covered", "committed", 2),
        ("all_pass", "partial", "committed", 2),
        ("partial_pass", "partial", "committed", 1),
        ("not_run", "uncovered", "committed", 1),
        ("partial_pass", "partial", "uncommitted", 0),
    ],
)
def test_confidence_rules(checks: str, coverage: str, artifact: str, expected: int) -> None:
    """输入：自报事实。输出：无。预期行为：三条门槛过了几条就是几档。"""
    assert features.confidence_from(Facts(checks, coverage, artifact)) == expected


def test_gates_and_confidence_agree_on_every_case() -> None:
    """输入：全空间。输出：无。

    预期行为：三条门槛、`confidence`、以及各自那条判据用的是**同一份事实**——
    渲染器写门槛行、审计重算结论，都靠这条不变量对齐。
    """
    for case in features.build_cases(1000):
        facts = case.facts
        gates = features.gates_from(facts)
        assert sum(gates) == features.confidence_from(facts)
        assert gates[0] is (facts.checks == "all_pass")
        assert gates[1] is (facts.coverage == "covered")
        assert gates[2] is (facts.artifact == "committed")
        assert features.committed_from(facts) is gates[2]
        assert len(gates) == features.GATE_COUNT


def test_labels_are_a_pure_function_of_the_tags() -> None:
    """输入：全空间。输出：无。

    预期行为：同一组特征必得同一套标签；两个由文本承载的标签直接来自注入特征。
    """
    for case in features.build_cases(1000):
        labels = case.label()
        assert labels == features.label_for(case)
        assert labels.kind == case.claim_kind
        assert labels.tone == case.tone
        assert labels.committed is (case.artifact == "committed")


def test_every_label_value_is_an_option() -> None:
    """输入：全空间。输出：无。

    预期行为：choice 题族的选项表与判据能产出的取值**一一对应**（不多不少），
    `confidence` 四档也都能取到——多一个选项是送分题，少一个是死选项。
    """
    seen: dict[str, set[str]] = {family: set() for family in features.CHOICE_OPTIONS}
    confidences: set[int] = set()
    for case in features.build_cases(1000):
        labels = case.label()
        for family in features.CHOICE_OPTIONS:
            seen[family].add(str(getattr(labels, family)))
        confidences.add(labels.confidence)
    for family, options in features.CHOICE_OPTIONS.items():
        assert seen[family] == set(options), family
    assert confidences == set(range(features.GATE_COUNT + 1))


def test_documented_distributions_hold() -> None:
    """输入：全空间。输出：无。

    预期行为：实测分布与 `label_for` 文档里写的那三个数字**一致**——文档里的偏斜是
    设计取舍（照实标出不掩盖），但数字一旦漂了，说明判据被悄悄改过，得有人知道。
    """
    labels = [case.label() for case in features.build_cases(1000)]
    assert _shares([label.verdict for label in labels]) == {
        "partial": 30.0,
        "refuted": 30.0,
        "unverified": 30.0,
        "verified": 10.0,
    }
    assert _shares([label.gap for label in labels]) == {
        "commit_the_change": 5.0,
        "fix_the_failing_checks": 60.0,
        "nothing_missing": 5.0,
        "run_the_checks": 10.0,
        "tighten_scope": 20.0,
    }
    assert _shares([label.confidence for label in labels]) == {
        "0": 25.0,
        "1": 45.0,
        "2": 25.0,
        "3": 5.0,
    }


def test_committed_is_a_coin_flip() -> None:
    """输入：全空间。输出：无。

    预期行为：noul 那一族正好五五开——`artifact` 不与禁用对相交，所以它的两个取值在
    全空间里一样多（均衡闸对 noul 的参考值就是 50%）。
    """
    counts = Counter(case.label().committed for case in features.build_cases(1000))
    assert counts[True] == counts[False] == 150


def test_label_flip_axes_skips_forbidden_neighbours() -> None:
    """输入：两组特征。输出：无。

    预期行为：只改一维的邻居里，撞上禁用对的那些被跳过——「没跑检查」这一组翻不动
    覆盖维，而「覆盖完整」那一组翻不动检查维（翻成 `not_run` 就被禁了）。
    """
    stuck = TagSet("tests_added", "plain", "not_run", "uncovered", "committed")
    axes = features.label_flip_axes(stuck)
    assert stuck not in axes
    assert len(axes) == 10
    assert {axis.coverage for axis in axes} == {"uncovered"}
    free = TagSet("tests_added", "plain", "all_pass", "covered", "committed")
    flipped = {
        axis.checks
        for axis in features.label_flip_axes(free)
        if axis.checks != free.checks
    }
    assert flipped == {"partial_pass", "all_fail"}
