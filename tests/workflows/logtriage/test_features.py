"""日志行分诊的特征层测试：判据、通道、标签两条路径与计划。"""

from __future__ import annotations

import pytest

from tests.workflows.logtriage.conftest import text_of
from workflows.logtriage import features


def test_hit_set_is_purely_literal() -> None:
    """输入：一行日志。输出：无。预期行为：命中组合按 `TAG_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
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
    ]:
        try:
            assert features.hit_set(text) == expect
            assert features.hits_of(text) == "+".join(expect)
            assert features.parse_hits(features.hits_of(text)) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


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


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.LogTriageCase("", features.SOURCE_APP)
    assert case.describe() == "无命中-app"
    assert case.fingerprint() == "hits=|source=app"
    assert case.tags() == ()
    assert features.parse_hits("") == ()
