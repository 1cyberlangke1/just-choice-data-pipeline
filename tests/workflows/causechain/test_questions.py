"""题族测试：五族题的形状、答案与「题面点名的服务」这一条审计口径。"""

from __future__ import annotations

import pytest

from workflows.causechain import features, locales, questions


#: 跳数档位 ↔ 探针末位数字的对应（`h0` 问 0 跳、`h1` 问 1 跳、`h2` 问 2 跳以上）。
HOP_DIGITS = {features.HOP_ZERO: "0", features.HOP_ONE: "1", features.HOP_MORE: "2"}


def test_family_order_follows_the_declared_order() -> None:
    """输入：全空间 100 组 case。输出：无。

    预期行为：出题的**首次出现顺序**与 `FAMILIES` 一致；`hop` 只在跳数探针上出、`isroot`
    只在布尔探针上出，所以任何一条样本最多三族。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        out = questions.build(features.plan_for(case, 0), loc, 0)
        seen = list(dict.fromkeys(item.family for item in out))
        assert seen == [family for family in questions.FAMILIES if family in seen], case.describe()
        assert "root" in seen and "span" in seen
        assert ("hop" in seen) == (case.probe in ("h0", "h1", "h2"))
        assert ("isroot" in seen) == (case.probe in ("ryes", "rno"))


def test_root_family_lists_all_six_services() -> None:
    """输入：全空间 100 组 case。输出：无。

    预期行为：六个服务全出、选项描述两两不同、答案是注入的那个根因——选项就是服务位，
    所以这一族不会因为「答案退化成某一个」而失去区分度。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        item = questions.build(features.plan_for(case, 0), loc, 0)[0]
        assert item.family == "root"
        assert set(item.question["criteria"]) == set(features.SERVICES)
        assert len(set(item.question["criteria"].values())) == len(features.SERVICES)
        assert item.answer == case.root


def test_hop_family_matches_the_probed_distance() -> None:
    """输入：全空间 100 组 case。输出：无。

    预期行为：题面点名的就是计划里那个探针服务，答案是它在依赖图上的档位；探针末位数字与
    档位一一对应（`h0`/`h1`/`h2` ↔ zero/one/more），所以「问几跳」与「答几档」不会错位。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        labels = features.label_for(plan)
        hop = [item for item in questions.build(plan, loc, 0) if item.family == "hop"]
        if case.probe not in ("h0", "h1", "h2"):
            assert hop == [], case.describe()
            continue
        assert len(hop) == 1
        item = hop[0]
        assert questions.named_service(str(item.question["instructions"])) == labels.probed
        assert item.answer == labels.probed_hop_class
        assert HOP_DIGITS[item.answer] == case.probe[1]
        assert item.question["criteria"][item.answer] == questions.hop_text(loc, item.answer)
        assert set(item.question["criteria"]) == set(features.HOP_CLASSES)


def test_isroot_family_splits_true_and_false_by_probe() -> None:
    """输入：全空间 100 组 case。输出：无。

    预期行为：`ryes` 问的就是根因自己（答 True）、`rno` 问的是别的出错服务（答 False）；
    两种探针各自可行到上限，所以这一族的两档接近一半一半。
    """
    loc = locales.load("zh")
    counted: dict[bool, int] = {True: 0, False: 0}
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        labels = features.label_for(plan)
        found = [item for item in questions.build(plan, loc, 0) if item.family == "isroot"]
        if case.probe not in ("ryes", "rno"):
            assert found == [], case.describe()
            continue
        assert len(found) == 1
        item = found[0]
        assert questions.named_service(str(item.question["instructions"])) == labels.probed
        assert item.answer is (case.probe == "ryes")
        assert item.answer == labels.probed_is_root
        counted[item.answer] += 1
    assert counted == {True: 24, False: 20}


def test_span_family_is_the_truth_plus_three_near_misses() -> None:
    """输入：全空间 100 组 case。输出：无。

    预期行为：四个选项两两不同、真值永远是 `c0`、三个干扰集合与真值只差一两处（差得越少
    越难分辨，题目才有区分度），而且没有一个干扰集合等于真值。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        labels = features.label_for(features.plan_for(case, 0))
        item = next(
            q for q in questions.build(features.plan_for(case, 0), loc, 0) if q.family == "span"
        )
        criteria = item.question["criteria"]
        assert item.answer == "c0"
        assert len(criteria) == 4
        assert len(set(criteria.values())) == 4
        assert criteria["c0"] == questions.span_text(loc, labels.errored)
        for combo in labels.distractors:
            assert combo != labels.errored
            assert len(set(combo) ^ set(labels.errored)) <= 2


def test_count_family_is_the_number_of_errored_services() -> None:
    """输入：全空间 100 组 case。输出：无。

    预期行为：档位表长度等于 `features.COUNT_LEVELS`、描述两两不同、答案下标 = 出错服务数 − 1；
    每条样本都出这一族（它不吃探针）。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        labels = features.label_for(plan)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "count")
        levels = item.question["criteria"]
        assert isinstance(levels, list)
        assert len(levels) == features.COUNT_LEVELS
        assert len(set(levels)) == features.COUNT_LEVELS
        assert item.answer == len(labels.errored) - 1
        assert levels[item.answer] == questions.count_text(loc, labels.count_index)


def test_count_text_rejects_an_out_of_range_index() -> None:
    """输入：越界下标。输出：无。预期行为：当场报人话——静默给空串会把「档位抄错」藏起来。"""
    loc = locales.load("zh")
    with pytest.raises(ValueError, match="越界"):
        questions.count_text(loc, features.COUNT_LEVELS)
    with pytest.raises(ValueError, match="越界"):
        questions.count_text(loc, -1)


@pytest.mark.parametrize(
    ("instructions", "expect"),
    [
        ("S1 离根因几跳？", "S1"),
        ("S3 是这次故障的根因吗？", "S3"),
        ("这次故障的根因是哪个服务？", None),
        ("S1 和 S3 哪个是根因？", None),
        ("没有点名任何服务", None),
    ],
)
def test_named_service_reads_the_only_service_in_the_prompt(
    instructions: str, expect: str | None
) -> None:
    """输入：几种题面。输出：无。

    预期行为：题面里恰好一个服务编号时给出它，一个都没有、或点了多个就给 None——审计拿它
    把「这次问的是谁」从产物里还原出来，认不准的时候必须老实说认不出。
    """
    assert questions.named_service(instructions) == expect


def test_span_text_uses_the_canonical_order_and_a_none_phrase() -> None:
    """输入：几种服务集合。输出：无。预期行为：按 `SERVICES` 的规范顺序拼，空集合给那句「一个都没受影响」。"""
    loc = locales.load("zh")
    assert questions.span_text(loc, ("S3", "S1")) == "S1、S3"
    assert questions.span_text(loc, ()) == str(loc["fam"]["span"]["none"])
    assert questions.span_text(loc, ("S1",)) != questions.span_text(loc, ())


def test_root_text_uses_the_role_of_that_graph() -> None:
    """输入：同一服务在两张力不同的图里。输出：无。预期行为：描述随角色变，六个选项仍两两不同。"""
    loc = locales.load("zh")
    assert questions.root_text(loc, "G1", "S1") != questions.root_text(loc, "G1", "S3")
    assert questions.root_text(loc, "G1", "S1") == questions.root_text(loc, "G2", "S1")
    texts = {questions.root_text(loc, "G1", service) for service in features.SERVICES}
    assert len(texts) == len(features.SERVICES)
