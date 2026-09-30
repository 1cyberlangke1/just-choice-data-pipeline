"""日志行分诊的特征层测试：判据、通道、标签两条路径与计划。"""

from __future__ import annotations

import pytest

from tests.workflows.logtriage.conftest import text_of
from workflows.logtriage import features


def test_hit_sets_are_all_sixteen_combinations() -> None:
    """输入：无。输出：无。预期行为：16 组、空串在前、全命中在最后、顺序稳定。"""
    assert len(features.HIT_SETS) == 16
    assert features.HIT_SETS[0] == ""
    assert features.HIT_SETS[1] == "N1"
    assert features.HIT_SETS[-1] == "N1+N2+N3+N4"
    assert len(set(features.HIT_SETS)) == 16


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("心跳探测 /healthz 正常", ("N1",)),
        ("readiness probe failed once", ("N1",)),
        ("web 服务正在启动，开始监听 8080", ("N2",)),
        ("web service starting, listening on 8080", ("N2",)),
        ("DEBUG 处理请求耗时 12ms", ("N3",)),
        ("trace span exported", ("N3",)),
        ("上游超时，正在重试第 2 次", ("N4",)),
        ("upstream timed out, retrying attempt 2", ("N4",)),
        # 这两条是判据「只认字面」的证据：异常类名与复合词都不算。
        ("Traceback (most recent call last)", ()),
        ("container restarted by kubelet", ()),
        ("订单 88213 支付失败：上游返回 500", ()),
        ("心跳正常，正在重试第 2 次", ("N1", "N4")),
    ],
)
def test_hit_set_is_purely_literal(text: str, expect: tuple[str, ...]) -> None:
    """输入：一行日志。输出：无。预期行为：命中组合按 `TAG_IDS` 顺序、只看字面。"""
    assert features.hit_set(text) == expect
    assert features.hits_of(text) == "+".join(expect)
    assert features.parse_hits(features.hits_of(text)) == expect


def test_sources_are_three_distinct_shapes() -> None:
    """输入：无。输出：无。

    预期行为：三个通道键两两不同，且**全空间里每种通道都被枚举到**——留出维只砍取值，
    不该把别的取值一起砍掉（`build_cases` 默认维度表覆盖 `SOURCES`）。
    """
    assert features.SOURCES == ("app", "infra", "access")
    assert len(set(features.SOURCES)) == len(features.SOURCES)
    assert features.DIMS["source"] == features.SOURCES
    cases = features.build_cases(512)
    assert {case.source for case in cases} == set(features.SOURCES)


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且与真值只差一两处
    （差得越少越难分辨，题目才有区分度）。
    """
    case = features.LogTriageCase("N1+N3", features.SOURCE_APP)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.hits == ("N1", "N3")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.hits for combo in plan.distractors)
    assert all(len(set(combo) ^ set(plan.hits)) <= 2 for combo in plan.distractors)
    assert plan.distractors == tuple(sorted(plan.distractors, key=lambda combo: (len(combo), combo)))


@pytest.mark.parametrize(
    ("count", "expect"),
    [(0, "alert"), (1, "aggregate"), (2, "hold"), (3, "silence"), (4, "silence")],
)
def test_handling_level_thresholds(count: int, expect: str) -> None:
    """输入：命中条数。输出：无。预期行为：0/1/2/3 条以上 分别对应四个档位。"""
    assert features.handling_level(count) == expect


def test_two_label_paths_agree_on_every_faithful_text() -> None:
    """输入：全空间的 48 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——
    日志兑现了计划，两条路就不该打架。
    """
    cases = features.build_cases(512)
    assert len(cases) == 48  # 16 组命中 × 3 种通道
    for case in cases:
        text = text_of(case)
        assert features.hits_of(text) == case.hits, case.describe()
        assert features.observed_labels(text).hits == features.label_for(
            features.plan_for(case, 0)
        ).hits


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.LogTriageCase("", features.SOURCE_APP)
    assert case.describe() == "无命中-app"
    assert case.fingerprint() == "hits=|source=app"
    assert case.tags() == ()
    assert features.parse_hits("") == ()
