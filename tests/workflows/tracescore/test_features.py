"""代理轨迹自评的特征层测试：信号、入口、计分口径、标签两条路径与计划。"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.workflows.tracescore.conftest import text_of
from workflows.tracescore import features


def test_every_tag_appears_eight_times() -> None:
    """输入：无。输出：无。

    预期行为：每条信号在 16 组里各出现 8 次——这是 `flag` 那一族的答案按构造就是 50% 的原因。
    """
    for tag in features.TAG_IDS:
        hits = sum(1 for value in features.HIT_SETS if tag in features.parse_hits(value))
        assert hits == 8, tag


def test_hit_set_is_purely_literal() -> None:
    """输入：一段轨迹。输出：无。预期行为：命中组合按 `TAG_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ("跑了测试，两处失败", ("S1",)),
        ("npm test 挂了", ("S1",)),
        ("又试了一次同样的命令", ("S2",)),
        ("the retry did not help, same error again", ("S2",)),
        ("输出显示路径解析错了", ("S3",)),
        ("the output shows two failures", ("S3",)),
        ("又看了一遍那个文件", ("S4",)),
        ("reopened the same file", ("S4",)),
        # 这两条是判据「只认字面」的证据：换个说法、复合词都不算。
        ("测试跑得不错", ()),
        ("我重读了一遍需求文档", ()),
        ("lint 通过了，代码风格没问题", ("S1",)),
        ("输出显示还是失败", ("S2", "S3")),
        # 英文散文形态：判据原文里列了它们，正则就必须认（en 侧首轮 3 条拒稿的根因）。
        ("ran the tests, the tests pass now", ("S1",)),
        ("ran it once more, still broken", ("S2",)),
    ]:
        try:
            assert features.hit_set(text) == expect
            assert features.hits_of(text) == "+".join(expect)
            assert features.parse_hits(features.hits_of(text)) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_sources_are_three_distinct_shapes() -> None:
    """输入：无。输出：无。

    预期行为：三个入口键两两不同，且**全空间里每种入口都被枚举到**——留出维只砍取值，
    不该把别的取值一起砍掉（`build_cases` 默认维度表覆盖 `SOURCES`）。
    """
    assert features.SOURCES == ("cli", "editor", "chat")
    assert len(set(features.SOURCES)) == len(features.SOURCES)
    assert features.DIMS["source"] == features.SOURCES
    cases = features.build_cases(512)
    assert {case.source for case in cases} == set(features.SOURCES)


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且与真值只差一两处
    （差得越少越难分辨，题目才有区分度）。
    """
    case = features.TraceScoreCase("S1+S3", features.SOURCE_CLI)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.hits == ("S1", "S3")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.hits for combo in plan.distractors)
    assert all(len(set(combo) ^ set(plan.hits)) <= 2 for combo in plan.distractors)
    assert plan.distractors == tuple(
        sorted(plan.distractors, key=lambda combo: (len(combo), combo))
    )


def test_score_weighs_the_four_conditions() -> None:
    """输入：16 组命中组合。输出：无。

    预期行为：基础 1 分，`S1` / `S3` 命中各加 1，`S2` / `S4` **未**命中各加 1——所以同样命中
    两条的组合分数可以差到 4 分（`S1+S3` 是 5 分、`S2+S4` 是 1 分），`count` 与 `score` 不是
    一一对应。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for hits, score in [
        ("", 3),
        ("S1", 4),
        ("S2", 2),
        ("S3", 4),
        ("S4", 2),
        ("S1+S2", 3),
        ("S1+S3", 5),
        ("S1+S4", 3),
        ("S2+S3", 3),
        ("S2+S4", 1),
        ("S3+S4", 3),
        ("S1+S2+S3", 4),
        ("S1+S2+S4", 2),
        ("S1+S3+S4", 4),
        ("S2+S3+S4", 2),
        ("S1+S2+S3+S4", 3),
    ]:
        try:
            assert features.score_for(features.parse_hits(hits)) == score
            assert features.label_for(features.plan_for(
                features.TraceScoreCase(hits, features.SOURCE_CLI), 0
            )).score == score
        except AssertionError as exc:
            bad.append(f"{(hits, score)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_score_distribution_is_the_binomial_shape() -> None:
    """输入：16 组命中组合。输出：无。

    预期行为：五档占比 1:4:6:4:1——四条计分条件在 16 组上各占一半，所以分档与「命中条数」同分布
    （与 `logtriage` 的 `count` 同款）。
    """
    counts = Counter(
        features.score_for(features.parse_hits(value)) for value in features.HIT_SETS
    )
    assert counts == Counter({3: 6, 2: 4, 4: 4, 1: 1, 5: 1})


@pytest.mark.parametrize(
    ("score", "route"),
    [(1, "escalate"), (2, "escalate"), (3, "replan"), (4, "proceed"), (5, "proceed")],
)
def test_route_thresholds(score: int, route: str) -> None:
    """输入：1~5 分。输出：无。预期行为：4 分及以上继续、正好 3 分换方案、2 分及以下问人。"""
    assert features.route_for(score) == route


def test_route_distribution_is_five_six_five() -> None:
    """输入：16 组命中组合。输出：无。

    预期行为：三档在 16 组上占比 5:6:5——三档而不是四档，是因为分档口径只有「继续 / 换方案 /
    停下问人」这三种真实动作。
    """
    counts = Counter(
        features.route_for(features.score_for(features.parse_hits(value)))
        for value in features.HIT_SETS
    )
    assert counts == Counter({"replan": 6, "proceed": 5, "escalate": 5})
    assert set(features.ROUTE_LEVELS) == set(counts)


def test_two_label_paths_agree_on_every_faithful_text() -> None:
    """输入：全空间的 48 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——轨迹兑现了
    计划，两条路就不该打架。
    """
    cases = features.build_cases(512)
    assert len(cases) == 48  # 16 组命中 × 3 种入口
    for case in cases:
        text = text_of(case)
        assert features.hits_of(text) == case.hits, case.describe()
        assert features.observed_labels(text).hits == features.label_for(
            features.plan_for(case, 0)
        ).hits


def test_case_fingerprint_and_describe() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.TraceScoreCase("", features.SOURCE_CLI)
    assert case.describe() == "无命中-cli"
    assert case.fingerprint() == "hits=|source=cli"
    assert case.tags() == ()
    assert features.parse_hits("") == ()
