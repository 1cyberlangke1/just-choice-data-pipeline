"""特征层测试：判据只认字面、命中组合按构造均衡、两条标签路径互不打架。"""

from __future__ import annotations


from tests.workflows.sqlreview.conftest import packet_of
from workflows.sqlreview import features


def test_hit_set_is_purely_literal() -> None:
    """输入：一屏评审记录里的一行。输出：无。预期行为：命中组合按 `TAG_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ("statement: DELETE FROM sessions;", ("Q1",)),
        ("statement: update orders set paid = true;", ("Q1",)),
        ("statement: UPDATE orders SET status = 'paid' WHERE id = 42;", ()),
        ("statement: DELETE FROM sessions WHERE id = 3;", ()),
        ("statement: SELECT count(*) FROM orders;", ()),
        ("plan: Seq Scan on orders  (cost=0.00..18334.00 rows=1000000 width=12)", ("Q2",)),
        ("plan: Parallel Seq Scan on events  (cost=0.00..900.10 rows=12345678 width=8)", ("Q2",)),
        ("plan: Index Scan using orders_pkey on orders  (cost=0.43..8.45 rows=1 width=12)", ()),
        ("plan: Seq Scan on events  (cost=0.00..90.10 rows=2400 width=8)", ()),
        ("plan: Seq Scan on orders  (cost=0.00..18334.00 rows=999999 width=12)", ()),
        ("ddl: ALTER TABLE orders ADD COLUMN paid_at timestamptz;", ("Q3",)),
        ("ddl: DROP INDEX idx_orders_status;", ("Q3",)),
        ("ddl: ALTER TABLE orders ADD COLUMN IF NOT EXISTS paid_at timestamptz;", ()),
        ("ddl: DROP TABLE IF EXISTS tmp_orders;", ()),
        ("ddl: CREATE INDEX CONCURRENTLY idx_orders_paid ON orders (paid);", ()),
        ("note: 这个字段的回填脚本待确认，先记一笔", ("Q4",)),
        ("note: TODO 找上游确认这里的字段类型", ("Q4",)),
        ("note: 这个口径需要复核一遍", ("Q4",)),
        ("note: 这次只改一行、走主键索引，直接执行没问题", ()),
        ("note: NOTODO 只是一个人名，不是没写完的痕迹", ()),
        # 这几条是判据「只认字面、各占一行」的证据：形状落在**别的行**上时不该算这一条。
        ("statement: Seq Scan on orders  (cost=0.00..18334.00 rows=1000000 width=12)", ()),
        ("plan: UPDATE orders SET status = 'paid';", ()),
        ("note: DROP TABLE orders;", ()),
        ("statement: 这条语句没有 WHERE，也无所谓", ()),
        ("ddl: 结构变更那一条写在别处", ()),
    ]:
        try:
            assert features.hit_set(text) == expect
            assert features.hits_of(text) == "+".join(expect)
            assert features.parse_hits(features.hits_of(text)) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_key_lines_are_the_four_structure_keys() -> None:
    """输入：无。输出：无。

    预期行为：四个结构键是这一族 state 格式的一部分（键名用英文标识符），`missing_keys` 认的是
    「单独占一行、后面跟冒号」——少一行就定位不到钉在行上的那几条判据。
    """
    assert features.KEY_LINES == ("statement", "plan", "ddl", "note")
    packet = packet_of(())
    assert features.missing_keys(packet) == ()
    assert features.missing_keys(packet.replace("note:", "备注：")) == ("note",)


def test_workloads_are_three_distinct_shapes() -> None:
    """输入：无。输出：无。

    预期行为：三个变更场景键两两不同，且**全空间里每种场景都被枚举到**——留出维只砍取值，
    不该把别的取值一起砍掉（`build_cases` 默认维度表覆盖 `WORKLOADS`）。
    """
    assert features.WORKLOADS == ("oltp", "analytics", "migration")
    assert len(set(features.WORKLOADS)) == len(features.WORKLOADS)
    assert features.DIMS["workload"] == features.WORKLOADS
    cases = features.build_cases(512)
    assert {case.workload for case in cases} == set(features.WORKLOADS)


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且与真值只差一两处
    （差得越少越难分辨，题目才有区分度）。
    """
    case = features.SqlReviewCase("Q1+Q3", features.WORKLOAD_OLTP)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.hits == ("Q1", "Q3")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.hits for combo in plan.distractors)
    assert all(len(set(combo) ^ set(plan.hits)) <= 2 for combo in plan.distractors)
    assert plan.count == 2


def test_two_label_paths_agree_on_every_faithful_packet() -> None:
    """输入：全空间的 48 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——
    评审记录兑现了计划，两条路就不该打架。
    """
    cases = features.build_cases(512)
    assert len(cases) == 48  # 16 组命中 × 3 种场景
    for case in cases:
        packet = packet_of(case.tags())
        assert features.hits_of(packet) == case.hits, case.describe()
        assert (
            features.observed_labels(packet).hits
            == features.label_for(features.plan_for(case, 0)).hits
        )


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.SqlReviewCase("", features.WORKLOAD_OLTP)
    assert case.describe() == "无命中-oltp"
    assert case.fingerprint() == "hits=|workload=oltp"
    assert case.tags() == ()
    assert features.parse_hits("") == ()
