"""规则表与 case 矩阵：每条规则都对着题面写的那张表验一遍。"""

from __future__ import annotations

import pytest

from tests.workflows.routing.conftest import _make_case
from workflows.routing import spec as sp


def test_severity_matrix_covers_every_combination() -> None:
    """输入：影响面 × 紧急度的全部组合。输出：无。

    预期行为：矩阵恰好九格、每格都在 `SEVERITIES` 里——矩阵漏一格会让题面里的判据与
    代码算出来的答案对不上，而读者是照着题面算的。
    """
    assert len(sp.SEVERITY_TABLE) == len(sp.IMPACTS) * len(sp.URGENCIES)
    for impact in sp.IMPACTS:
        for urgency in sp.URGENCIES:
            assert sp.severity_of(impact, urgency) in sp.SEVERITIES


def test_severity_matrix_is_the_one_written_in_the_prompt() -> None:
    """输入：九格矩阵。输出：无。预期行为：与题面原文逐格一致（这里写死一遍当锚点）。"""
    assert sp.severity_of("single_user", "can_wait") == "s4"
    assert sp.severity_of("single_user", "blocked") == "s4"
    assert sp.severity_of("single_user", "deadline") == "s3"
    assert sp.severity_of("team", "can_wait") == "s3"
    assert sp.severity_of("team", "blocked") == "s2"
    assert sp.severity_of("team", "deadline") == "s2"
    assert sp.severity_of("all_users", "can_wait") == "s3"
    assert sp.severity_of("all_users", "blocked") == "s1"
    assert sp.severity_of("all_users", "deadline") == "s1"


def test_severity_of_unknown_combination_speaks_human() -> None:
    """输入：矩阵外的组合。输出：无。预期行为：报人话，不抛裸 KeyError。"""
    with pytest.raises(ValueError, match="严重度矩阵"):
        sp.severity_of("nobody", "blocked")


@pytest.mark.parametrize(
    ("issue", "compliance", "want"),
    [
        ("outage", "none", "incident_response"),
        ("outage", "gdpr", "incident_response"),
        ("outage", "security", "security_incident"),
        ("data_loss", "none", "incident_response"),
        ("data_loss", "gdpr", "privacy_ops"),
        ("data_loss", "security", "security_incident"),
        ("bug", "none", "engineering"),
        ("bug", "gdpr", "engineering"),
        ("bug", "security", "security_incident"),
        ("howto", "none", "support_l1"),
        ("howto", "gdpr", "support_l1"),
        ("howto", "security", "support_l1"),
        ("feature", "none", "product_backlog"),
        ("feature", "gdpr", "privacy_ops"),
        ("feature", "security", "product_backlog"),
    ],
)
def test_queue_rules_match_the_prompt(issue: str, compliance: str, want: str) -> None:
    """输入：十五个 (问题类型 × 合规标记) 组合。输出：无。

    预期行为：每个组合都落在题面那六条有序规则算出来的队列上——这张表是**手抄题面**的，
    所以题面与代码哪边改了都会当场露馅。
    """
    assert sp.queue_of(issue, compliance) == want


def test_every_queue_is_reachable() -> None:
    """输入：全空间。输出：无。预期行为：六个队列都至少被一个组合用到（没有死选项）。"""
    used = {
        sp.queue_of(issue, compliance)
        for issue in sp.ISSUES
        for compliance in sp.COMPLIANCES
    }
    assert used == set(sp.QUEUES)


@pytest.mark.parametrize(
    ("compliance", "want"),
    [("none", "none"), ("gdpr", "privacy_lead"), ("security", "security_lead")],
)
def test_escalation_follows_the_compliance_flag(compliance: str, want: str) -> None:
    """输入：三个合规标记。输出：无。预期行为：升级对象只看合规标记。"""
    assert sp.escalation_of(compliance) == want


def test_priority_table_shape() -> None:
    """输入：优先级表。输出：无。

    预期行为：s1 任何等级都是 4 档、s4 任何等级都是 0 档、中间两档按客户等级 1/2/3——
    与题面原文一致；档位都在 `PRIORITY_LEVELS` 里。
    """
    for tier in sp.TIERS:
        assert sp.priority_of("s1", tier) == 4
        assert sp.priority_of("s4", tier) == 0
    assert [sp.priority_of("s2", tier) for tier in sp.TIERS] == [1, 2, 3]
    assert [sp.priority_of("s3", tier) for tier in sp.TIERS] == [1, 2, 3]
    for severity in sp.SEVERITIES:
        for tier in sp.TIERS:
            assert sp.priority_of(severity, tier) in sp.PRIORITY_LEVELS


def test_sla_breach_is_strict() -> None:
    """输入：恰好卡在时限上的等待时长。输出：无。

    预期行为：**不算**超时——题面写的是「严格大于」，代码必须照办，
    否则所有「刚好卡点」的工单都会被标错。
    """
    assert sp.sla_breach_of(30, 30) is False
    assert sp.sla_breach_of(31, 30) is True


def test_sla_limits_shrink_as_severity_and_tier_grow() -> None:
    """输入：时限表。输出：无。

    预期行为：`SEVERITIES` 从 s1（最严重）排到 s4，所以同一等级下时限**递增**；
    `TIERS` 从免费版排到企业版，所以同一严重度下时限**递减**。
    """
    for tier in sp.TIERS:
        limits = [sp.sla_limit_of(item, tier) for item in sp.SEVERITIES]
        assert limits == sorted(limits)
    for severity in sp.SEVERITIES:
        by_tier = [sp.sla_limit_of(severity, item) for item in sp.TIERS]
        assert by_tier == sorted(by_tier, reverse=True)


def test_critical_area_only_covers_two_domains() -> None:
    """输入：四个产品域。输出：无。预期行为：只有支付与身份算关键域。"""
    assert [sp.critical_area_of(area) for area in sp.AREAS] == [True, False, False, True]


def test_needs_human_rule() -> None:
    """输入：几组（严重度 / 合规标记 / 客户等级）。输出：无。

    预期行为：s1/s2 一律要人工；安全报告只在企业版要人工；其余不要。
    """
    assert sp.needs_human_of("s1", "none", "free") is True
    assert sp.needs_human_of("s2", "none", "free") is True
    assert sp.needs_human_of("s3", "security", "enterprise") is True
    assert sp.needs_human_of("s3", "security", "pro") is False
    assert sp.needs_human_of("s4", "none", "enterprise") is False
    assert sp.needs_human_of("s3", "gdpr", "enterprise") is False


def test_data_risk_rule() -> None:
    """输入：几组（问题类型 / 合规标记）。输出：无。预期行为：数据丢失或数据删除请求。"""
    assert sp.data_risk_of("data_loss", "none") is True
    assert sp.data_risk_of("howto", "gdpr") is True
    assert sp.data_risk_of("howto", "none") is False
    assert sp.data_risk_of("bug", "security") is False


def test_times_differ_by_exactly_the_wait() -> None:
    """输入：四个产品域 × 三个等待档。输出：无。

    预期行为：当前时间 − 创建时间恰好等于该档的分钟数——判别器就是从这两格相减反推的，
    差一分钟就会让 `sla_breach` 判错。
    """
    for area in sp.AREAS:
        for wait in sp.WAITS:
            created, current = sp.times_of(area, wait)
            start = int(created[:2]) * 60 + int(created[3:])
            end = int(current[:2]) * 60 + int(current[3:])
            assert end - start == sp.elapsed_of(wait)


def test_ticket_id_is_stable_and_does_not_leak_the_split() -> None:
    """输入：同一个 case 用不同切分名 / 序号造两次。输出：无。

    预期行为：工单号只由素材决定——把切分或序号拼进工单号，等于把「这条是训练还是 OOD」
    写进 state，模型可以直接背下来。
    """
    case = _make_case()
    first = sp.ticket_id_of(case)
    assert first == sp.ticket_id_of(case)
    assert first.startswith(sp.TICKET_PREFIX)
    assert first != sp.ticket_id_of(_make_case(area="storage"))
    train_id = sp.sample_spec(None, case, "train", 3, 0).ticket_id
    ood_id = sp.sample_spec(None, case, "test_ood", 999, 2).ticket_id
    assert train_id == ood_id


def test_build_cases_enumerates_the_full_cross_product() -> None:
    """输入：全取值表。输出：无。预期行为：条数等于七维之积（本 workflow 不做成立性过滤）。"""
    total = 1
    for pool in (sp.AREAS, sp.ISSUES, sp.TIERS, sp.IMPACTS, sp.URGENCIES, sp.COMPLIANCES, sp.WAITS):
        total *= len(pool)
    assert len(sp.build_cases(sp.RoutingConfig())) == total == 4860


def test_ood_cases_always_carry_an_unseen_value() -> None:
    """输入：`ood_only` 的全表。输出：无。预期行为：每条都至少有一维落在训练侧没见过的取值上。"""
    cases = sp.build_cases(sp.RoutingConfig(split="test_ood", ood_only=True))
    assert cases
    assert all(case.unseen_dims() for case in cases)


def test_id_and_ood_case_sets_are_disjoint() -> None:
    """输入：训练侧取值表切出来的 case 与 OOD 的 case。输出：无。

    预期行为：两个集合不相交——OOD 的定义就是「至少含一个训练侧没见过的取值」，
    而训练侧枚举时那些取值一个都不在池子里，所以这条是**按构造成立**的。
    """
    held_out = {
        ("area", "identity"),
        ("issue", "feature"),
        ("compliance", "security"),
    }
    id_cases = sp.build_cases(
        sp.RoutingConfig(
            areas=tuple(sorted(sp.SEEN_AREAS)),
            issues=tuple(sorted(sp.SEEN_ISSUES)),
            compliances=tuple(sorted(sp.SEEN_COMPLIANCES)),
        )
    )
    ood_cases = sp.build_cases(sp.RoutingConfig(split="test_ood", ood_only=True))
    for case in id_cases:
        assert not set(case.as_dict().items()) & held_out
    assert not set(id_cases) & set(ood_cases)


def test_config_rejects_empty_pools_and_unknown_values() -> None:
    """输入：空取值表 / 不存在的取值。输出：无。预期行为：构造期就报人话。"""
    with pytest.raises(ValueError, match="areas"):
        sp.RoutingConfig(areas=())
    with pytest.raises(ValueError, match="不存在的取值"):
        sp.RoutingConfig(waits=("forever",))


def test_sample_spec_derives_everything_from_the_case() -> None:
    """输入：一个 case。输出：无。预期行为：规格里的每一项都与规则函数一致。"""
    case = _make_case(area="identity", issue="feature", tier="enterprise", compliance="gdpr")
    spec = sp.sample_spec(None, case, "train", 1, 0)
    severity = sp.severity_of(case.impact, case.urgency)
    assert spec.severity == severity
    assert spec.queue == sp.queue_of(case.issue, case.compliance) == "privacy_ops"
    assert spec.escalation == "privacy_lead"
    assert spec.priority == sp.priority_of(severity, case.tier)
    assert spec.critical_area is True
    assert spec.sla_limit == sp.sla_limit_of(severity, case.tier)
    assert spec.elapsed == sp.elapsed_of(case.wait)
    assert spec.sla_breach == sp.sla_breach_of(spec.elapsed, spec.sla_limit)
