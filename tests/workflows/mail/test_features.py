"""特征层测试：维度表、标签规则、组合枚举（纯函数，不需要后端）。"""

from __future__ import annotations

from collections import Counter

import pytest

from jc.cover import missing_pairs
from workflows.mail import features, questions


def test_link_split_roundtrip() -> None:
    """链接维拆出来的 (域名, 链接文字) 必须能拼回去，且没有链接时锚点是 n_a。"""
    assert features.split_link(features.LINK_NONE) == ("none", "n_a")
    for value in features.LINK:
        target, anchor = features.split_link(value)
        if value == features.LINK_NONE:
            continue
        assert f"{target}_{anchor}" == value
        assert target in features.LINK_TARGETS
        assert anchor in features.ANCHORS


def test_link_split_rejects_unknown() -> None:
    """未知链接取值必须报错，不静默兜底。"""
    with pytest.raises(ValueError, match="未知的链接取值"):
        features.split_link("whatever")


def test_label_is_pure_and_repeatable() -> None:
    """同一组特征算两次必须一模一样，且 TagSet.label() 与 label_for() 一致。"""
    case = features.TagSet("forged_brand", "urgency", "credentials", "ip_mismatch", "curt")
    assert case.label() == features.label_for(case)
    assert case.label() == case.label()


def test_tone_is_pure_register_and_moves_no_label() -> None:
    """语气维只描述语域：换语气不许改变任何标签，取值也不许与「手法」重合。

    输入：无（全枚举语气取值）。输出：无。
    预期行为：原先的 `tone=threatening` 既把「施压」与 `lure=fear` 重复表达了一遍，
    又给 severity 加了一分，于是 `lure=none` + `tone=threatening` 的清单自相矛盾
    （一边要求「通篇没有任何施压」，一边要求「用警告与后果的措辞」）。
    现在施压信号只由 `lure` 维承担，这一维必须与内容正交。
    """
    assert not set(features.TONE) & set(features.LURE)
    base = features.TagSet("known_vendor", "none", "none", "official_honest", "formal")
    for tone in features.TONE:
        other = features.TagSet("known_vendor", "none", "none", "official_honest", tone)
        assert other.label() == base.label()


def test_phishing_requires_both_a_request_and_a_suspicious_signal() -> None:
    """钓鱼判定是合取：只索取不算，只可疑也不算，两者同时成立才算。"""
    only_ask = features.TagSet("known_vendor", "none", "credentials", "official_honest", "formal")
    only_suspicious = features.TagSet("forged_brand", "none", "none", "official_honest", "formal")
    both = features.TagSet("forged_brand", "none", "credentials", "official_honest", "formal")
    assert only_ask.label().is_phishing is False
    assert only_suspicious.label().is_phishing is False
    assert both.label().is_phishing is True

def test_spam_needs_a_promo_lure_or_a_stranger_asking_for_something() -> None:
    """输入：四种贴着判据的组合。输出：无。

    预期行为：`is_spam` 的两条路各钉一头——① 促销诱因（中奖 / 好奇）本身就算，与发件人是谁
    无关；② 陌生个人**并且**提了要求才算，陌生个人但没提要求不算。变异测试把第二支的
    `ask != none` 翻成 `== none`，全空间 2640 条里 528 条标签会变，而改前**一条测试都不红**
    （`tmp/probe_mutation3.py` / `tmp/probe_mail_spam.py`）。
    """
    promo = features.TagSet("known_vendor", "reward", "none", "official_honest", "formal")
    stranger_asks = features.TagSet(
        "unknown_personal", "none", "credentials", "official_honest", "formal"
    )
    stranger_quiet = features.TagSet("unknown_personal", "none", "none", "official_honest", "formal")
    known_quiet = features.TagSet("known_vendor", "none", "none", "official_honest", "formal")
    assert promo.label().is_spam is True
    assert stranger_asks.label().is_spam is True
    assert stranger_quiet.label().is_spam is False
    assert known_quiet.label().is_spam is False



def test_atomic_answers_are_read_off_the_tags() -> None:
    """四个原子题的答案就是对应维度的取值（不是二次推断出来的）。"""
    case = features.TagSet("unknown_personal", "reward", "attachment", "shortener_mismatch", "casual")
    labels = case.label()
    assert labels.requested == "attachment"
    assert labels.pressure == "reward"
    assert labels.link_text == "mismatched"
    assert labels.sender_identity == "no_claim"


def test_severity_and_action_are_in_range() -> None:
    """严重度必须是 0~4，处置必须是四个已知取值之一（全枚举检查）。"""
    for case in features.build_cases(10**9):
        labels = case.label()
        assert 0 <= labels.severity <= 4
        assert labels.action in {"deliver", "review", "quarantine", "block"}


def test_label_distribution_is_balanced() -> None:
    """全枚举下每个答案取值都要拿到像样的份额。

    阈值取 15%（远松于 1/选项数）：这条测试要抓的是「某个取值几乎取不到」这种
    结构性失衡，不是抓统计噪声。choice 题在均匀 tag 空间里天然均衡，
    只有 noul（子集判定）需要盯；`severity` 的 0 级天然是少数，
    由 `test_severity_floor_stays_a_minority` 单独盯。
    """
    counter: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for case in features.build_cases(10**9):
        labels = case.label()
        # 按**实际会出的题**统计：没有链接的邮件不出 link_text，就不该把它算进去
        for name in questions.families_for(case):
            if name == "severity":
                continue  # 有序量表另有一条专门的测试盯，这里只查分类题的份额
            counter[f"{name}={getattr(labels, name)}"] += 1
            totals[name] += 1
    thin = {
        key: (count, totals[key.split("=")[0]])
        for key, count in counter.items()
        if count / totals[key.split("=")[0]] < 0.15
    }
    assert not thin, f"这些取值占比不足 15%：{thin}"


def test_severity_floor_stays_a_minority() -> None:
    """0 级要求所有风险信号都不出现，天然是少数——盯住这个数别让它继续掉。

    输入：全枚举。输出：无。
    预期行为：0 级占比落在 8%~15%：低于 8% 说明风险信号铺得太满（判据写歪了），
    高于 15% 说明判据形同虚设。每一级也都得取得到。
    """
    cases = features.build_cases(10**9)
    counter = Counter(case.label().severity for case in cases)
    total = len(cases)
    assert set(counter) == {0, 1, 2, 3, 4}, f"严重度有取不到的档位：{sorted(counter)}"
    floor_share = counter[0] / total
    assert 0.08 <= floor_share <= 0.15, f"0 级占比 {floor_share:.3f} 越界"


def test_action_never_understates_a_high_severity() -> None:
    """处置不许低于风险等级该有的动作（全枚举检查）。

    输入：无（全量 case 空间）。输出：无。
    预期行为：3 级至少 `review`、4 级至少 `quarantine`。这条盯的是一处真实分叉——
    严重度是**风险量表**，钓鱼 / 垃圾是**分类判定**，两者会分叉：`sender=forged_brand`
    + `lure=urgency` + `ask=none` + `link=lookalike_mismatch` 够不上钓鱼（没索取），
    却实打实高风险。旧规则给它 `severity=4` 配 `action=deliver`（「风险极高」配
    「正常投递，不做额外处理」），全量扫描抓到 128 组这种组合。
    """
    floor = {3: "review", 4: "quarantine"}
    for case in features.build_cases(10**9):
        labels = case.label()
        required = floor.get(labels.severity, "deliver")
        assert features.ACTION_RANK[labels.action] >= features.ACTION_RANK[required], (
            case,
            labels,
        )


def test_no_request_but_high_risk_mail_is_not_delivered() -> None:
    """「什么都没索取」不等于「可以照常投递」：伪链 + 施压照样要复核。

    输入：两组「不索取任何东西」的高风险特征。输出：无。
    预期行为：风险 3 级（伪链 + 施压）→ `review`；风险 4 级（再加冒名）→ `quarantine`。
    两者都够不上钓鱼（钓鱼是合取：要索取 + 可疑），但都不许 `deliver`。
    """
    risky = features.TagSet("known_vendor", "urgency", "none", "lookalike_mismatch", "formal")
    worse = features.TagSet("forged_brand", "urgency", "none", "lookalike_mismatch", "formal")
    assert risky.label().is_phishing is False
    assert risky.label().severity == 3
    assert risky.label().action == "review"
    assert worse.label().is_phishing is False
    assert worse.label().severity == 4
    assert worse.label().action == "quarantine"


def test_pairwise_cover_is_complete() -> None:
    """成对覆盖必须覆盖任意两维的任意取值组合（这是 case 矩阵的承诺）。

    输入：无。输出：无。
    预期行为：被 `FORBIDDEN` 否掉的取值对不算「该覆盖而没覆盖」，
    同时它也不许真的出现在结果里——两头都要查。
    """
    cases = features.build_cases(32)
    rows = [case.as_dict() for case in cases]
    assert missing_pairs(features.DIMS, rows, allow_pair=features.allow_pair) == set()
    assert all(features.compatible(case) for case in cases)
    assert not any(row["ask"] == "click_link" and row["link"] == "none" for row in rows)


def test_full_enumeration_is_the_whole_space() -> None:
    """上限足够大时必须全枚举：数量 = 各维取值数之积 减去被禁的组合。"""
    cases = features.build_cases(10**9)
    expected = 1
    for values in features.DIMS.values():
        expected *= len(values)
    # 被禁的是「ask=click_link 配 link=none」这一对，其余三维自由组合 → 3×5×4 种
    blocked = 1
    for name, values in features.DIMS.items():
        if name not in {"ask", "link"}:
            blocked *= len(values)
    assert len(cases) == expected - blocked
    assert all(features.compatible(case) for case in cases)


def test_label_flip_axes_skips_impossible_neighbours() -> None:
    """翻转成无解组合的邻居必须被跳过，不能给出写不出来的清单。

    输入：一组「没有链接」的特征。输出：无。
    预期行为：把索取翻成 `click_link` 就无解（清单会同时要求点击链接与没有链接），
    这个邻居必须消失；其余索取档位照常给出。
    """
    case = features.TagSet("known_vendor", "none", "none", "none", "formal")
    neighbours = features.label_flip_axes(case)
    assert all(features.compatible(other) for other in neighbours)
    asks = [other.ask for other in neighbours if other.link == "none"]
    assert "click_link" not in asks
    assert "attachment" in asks


def test_label_flip_axes_are_single_dimension_neighbours() -> None:
    """`label_flip_axes` 给出的邻居必须只差一维、且不含自己。"""
    case = features.TagSet("known_vendor", "none", "none", "none", "formal")
    neighbours = features.label_flip_axes(case)
    assert neighbours
    for other in neighbours:
        assert other != case
        diff = [k for k, v in other.as_dict().items() if case.as_dict()[k] != v]
        assert len(diff) == 1


def test_link_address_pools_are_usable_and_unique() -> None:
    """链接地址池：非空、去重、IP 全落在 RFC 5737 的文档网段里。

    输入：无。输出：无。
    预期行为：池子里的取值不许重复（重复等于池子比看起来小）；IP 只许取 192.0.2.0/24、
    198.51.100.0/24、203.0.113.0/24 这三个文档网段——合成数据不该指向真实主机。
    """
    assert len(set(features.LINK_PATHS)) == len(features.LINK_PATHS)
    assert len(set(features.LINK_IPS)) == len(features.LINK_IPS)
    assert all(path and "/" not in path for path in features.LINK_PATHS)
    for ip in features.LINK_IPS:
        assert ip.startswith(("192.0.2.", "198.51.100.", "203.0.113.")), ip


def test_link_address_pickers_wrap_around() -> None:
    """输入：越界的样本序号。输出：无。预期行为：按池长回绕，不抛异常、且序号相邻时取值不同。"""
    assert features.link_path_at(0) == features.LINK_PATHS[0]
    assert features.link_path_at(len(features.LINK_PATHS)) == features.LINK_PATHS[0]
    assert features.link_ip_at(len(features.LINK_IPS) + 3) == features.LINK_IPS[3]
    assert features.link_path_at(0) != features.link_path_at(1)
    assert features.link_ip_at(0) != features.link_ip_at(1)
