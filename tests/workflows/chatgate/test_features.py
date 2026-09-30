"""chatgate 的特征层测试：判据、毛病、标签两条路径、计划，以及**标签分布**。"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.workflows.chatgate.conftest import candidates_of, message_of
from workflows.chatgate import features

#: 全空间 45 组 case（5 档命中条数 × 3 种场景 × 3 个候选位置）。
CASES = features.build_cases(512)


def test_case_space_is_the_full_cartesian_product() -> None:
    """输入：无。输出：无。预期行为：45 组，三个维度都铺满。"""
    assert len(CASES) == 45
    assert {case.degree for case in CASES} == set(features.DEGREES)
    assert {case.scene for case in CASES} == set(features.SCENES)
    assert {case.slot for case in CASES} == set(features.SLOTS)


def test_label_distribution_is_uniform_by_construction() -> None:
    """输入：全空间 45 组 case。输出：无。

    预期行为：命中条数**每档恰好 9 个**、候选位置**每个恰好 15 个**、每条判据被选中的比例落在
    一半附近。这条是**本 workflow 自己钉的**——通用均衡闸看的是打乱之后的选项位置，量不到标签
    分布（见 `features` 开头那段「为什么这条要自己钉」）。
    """
    assert set(Counter(case.degree for case in CASES).values()) == {9}
    assert set(Counter(case.slot for case in CASES).values()) == {15}
    assert set(Counter(case.scene for case in CASES).values()) == {15}
    hits = Counter(tag for case in CASES for tag in case.tags())
    for tag in features.TAG_IDS:
        rate = hits[tag] / len(CASES)
        assert 0.4 <= rate <= 0.6, f"{tag}: {rate}"


def test_tags_for_is_reproducible_and_not_a_constant() -> None:
    """输入：同一组入参两次。输出：无。

    预期行为：同一个 case 两次给同一组判据；`degree=2` 不是永远「前两条」——那样模型不用读消息
    就能答。
    """
    assert features.tags_for(2, "group", 1) == features.tags_for(2, "group", 1)
    pairs = {features.tags_for(2, scene, slot) for scene in features.SCENES for slot in features.SLOTS}
    assert len(pairs) == 6
    assert features.tags_for(0, "group", 0) == ()
    assert features.tags_for(4, "group", 0) == features.TAG_IDS


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("你定吧，我都行。", ("C1",)),
        ("这事你定，我不插手。", ("C1",)),
        ("今天之内能给我吗？", ("C2",)),
        ("30 分钟内回我一下。", ("C2",)),
        # 催促语不算：它们没给时间界。
        ("尽快给我，马上要。", ()),
        ("先借我两百块。", ("C3",)),
        ("帮我担保一下这笔贷款。", ("C3",)),
        ("把你的验证码发我。", ("C4",)),
        ("身份证号发我核对下。", ("C4",)),
        ("项目文档我更新好了，你有空翻一下。", ()),
        ("你定时间，今天之内告诉我就行。", ("C1", "C2")),
    ],
)
def test_hit_set_is_purely_literal(text: str, expect: tuple[str, ...]) -> None:
    """输入：一条消息。输出：无。预期行为：命中组合按 `TAG_IDS` 顺序、只看字面。"""
    assert features.hit_set(text) == expect
    assert features.hits_of(text) == "+".join(expect)


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("没问题，包在我身上。", ("F1",)),
        ("放心，我来搞定。", ("F1",)),
        ("把你的卡号发我核对一下。", ("F2",)),
        ("验证码发我。", ("F2",)),
        ("再说吧，看情况。", ("F3",)),
        ("改天再说。", ("F3",)),
        ("收到，我下午看一遍再回你。", ()),
        ("好的，我看下就回。", ()),
    ],
)
def test_flaw_set_is_purely_literal(text: str, expect: tuple[str, ...]) -> None:
    """输入：一条候选。输出：无。预期行为：空元组就是「这条可以发」。"""
    assert features.flaw_set(text) == expect


@pytest.mark.parametrize(
    ("count", "expect"),
    [(0, "direct"), (1, "confirm"), (2, "hold"), (3, "escalate"), (4, "escalate")],
)
def test_handling_level_thresholds(count: int, expect: str) -> None:
    """输入：命中条数。输出：无。预期行为：0/1/2/3 条以上 分别对应四个档位。"""
    assert features.handling_level(count) == expect


def test_two_label_paths_agree_on_every_faithful_text() -> None:
    """输入：全空间 45 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——文本兑现了
    计划，两条路就不该打架。
    """
    for case in CASES:
        plan = features.plan_for(case, 0)
        text = message_of(plan.hits)
        candidates = candidates_of(plan.reply.flaws)
        assert features.hits_of(text) == "+".join(case.tags()), case.describe()
        assert features.observed_labels(text, tuple(candidates)) == features.label_for(plan)


def test_plan_is_reproducible_and_spreads_the_flaws() -> None:
    """输入：同一个 case 的两种问法。输出：无。

    预期行为：计划与进程无关；该发的那条一个毛病都不踩、另外两条的毛病互不相同且非空；
    两种问法抽到不同的毛病分配（不然两种问法渲染出结构相同的样本）。
    """
    case = CASES[7]
    first = features.plan_for(case, 0)
    assert first == features.plan_for(case, 0)
    assert first.reply.clean == case.slot
    assert first.reply.flaws[case.slot] == ()
    dirty = [flaws for index, flaws in enumerate(first.reply.flaws) if index != case.slot]
    assert all(dirty) and len(set(dirty)) == len(dirty)
    layouts = {features.plan_for(case, phrasing).reply.flaws for phrasing in (0, 1, 2, 3)}
    assert len(layouts) > 1


@pytest.mark.parametrize(
    ("clean", "flaws"),
    [
        (0, ((), ())),  # 候选条数不对
        (0, ((), (), ())),  # 该发的那条有毛病
        (0, ((), ("F1",), ())),  # 另一条没毛病
        (0, ((), ("F1",), ("F1",))),  # 两条毛病一样
        (0, ((), ("F9",), ("F3",))),  # 不认识的毛病编号
    ],
)
def test_reply_plan_rejects_bad_plans(clean: int, flaws: tuple[tuple[str, ...], ...]) -> None:
    """输入：几种坏计划。输出：无。预期行为：构造期就抛错，不等到出题。"""
    with pytest.raises(ValueError):
        features.ReplyPlan(clean=clean, flaws=flaws)


def test_clean_indexes_reports_every_clean_candidate() -> None:
    """输入：三种候选组合。输出：无。

    预期行为：返回**全部**干净候选的下标——「一条都不干净」与「干净的不止一条」都是构造 bug，
    要让审计看得见，而不是硬凑一个答案。
    """
    assert features.clean_indexes(("收到，我看一下。", "没问题", "再说吧")) == (0,)
    assert features.clean_indexes(("嗯嗯", "再说吧", "看情况")) == ()
    assert features.clean_indexes(("收到", "没问题", "好的")) == (0, 2)


def test_build_cases_order_is_stable() -> None:
    """输入：同一份维度表两次。输出：无。预期行为：顺序逐格一致。"""
    assert [case.as_dict() for case in features.build_cases(512)] == [
        case.as_dict() for case in features.build_cases(512)
    ]


def test_dimension_table_matches_the_declared_keys() -> None:
    """输入：无。输出：无。预期行为：维度表的键与 case 的三个字段一一对应。"""
    assert sorted(features.DIMS) == ["degree", "scene", "slot"]
    assert features.DIMS["degree"] == features.DEGREES
    assert features.DIMS["scene"] == features.SCENES
    assert features.DIMS["slot"] == features.SLOTS
