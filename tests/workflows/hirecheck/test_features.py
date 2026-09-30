"""hirecheck 特征层：四条判据的正则、定级规则、旋钮池与计划推导。"""

from __future__ import annotations

import random

import pytest

from tests.workflows.hirecheck.conftest import packet_of
from workflows.hirecheck import features

#: 每条判据的「命中样本」与「不命中样本」——逐条验「只认自己那一行、只认那个形状」。
PATTERN_CASES: tuple[tuple[str, str, str], ...] = (
    ("H1", "years: 1", "years: 3"),
    ("H2", "skills: python, go", "skills: python, kubernetes"),
    ("H3", "permit: none", "permit: citizen"),
    ("H4", "degree: associate", "degree: bs"),
)


@pytest.mark.parametrize(("tag", "hit", "miss"), PATTERN_CASES)
def test_each_pattern_hits_only_its_own_line(tag: str, hit: str, miss: str) -> None:
    """输入：每条判据的命中 / 不命中写法。输出：无。

    预期行为：命中写法只算出它自己那一条；不命中写法一条都算不出来。判据是**按行**认的，所以
    单独一行就能验——这是「四条判据各占一整行」那条设计的直接后果。
    """
    assert features.hit_set(hit) == (tag,)
    assert features.hit_set(miss) == ()


def test_patterns_do_not_shadow_each_other() -> None:
    """输入：六行骨架把四条判据全取到。输出：无。预期行为：命中组合就是四条全中、顺序按 `TAG_IDS`。"""
    assert features.hit_set(packet_of(features.TAG_IDS)) == features.TAG_IDS


@pytest.mark.parametrize("value", ["years: 12", "years: 20", "years: 23"])
def test_two_digit_years_do_not_trip_h1(value: str) -> None:
    """输入：两位数的年限行。输出：无。

    预期行为：`12` / `20` / `23` 都不算命中——`(?!\\d)` 挡住多位数（它们的首位落在 0~2 里），
    这条正则只认 0 / 1 / 2 三个一位数。
    """
    assert features.hit_set(value) == ()


@pytest.mark.parametrize("value", ["years: 0", "years: 1", "years: 2"])
def test_single_digit_years_trip_h1(value: str) -> None:
    """输入：一位数的年限行。输出：无。预期行为：0 / 1 / 2 都算命中。"""
    assert features.hit_set(value) == ("H1",)


def test_negative_lookaheads_only_look_within_the_line() -> None:
    """输入：把该出现的词挪到**别的行**。输出：无。

    预期行为：`skills:` 那一行没有 `kubernetes` 就算命中（哪怕下一行写着 `kubernetes:`）；`degree:`
    那一行不是那三个代号就算命中（哪怕下一行是 `bs:`）。`[^\\n]*` 跨不过换行，这就是「每条判据各占
    一整行」要的性质。
    """
    assert features.hit_set("skills: python, go\nkubernetes: yes") == ("H2",)
    assert features.hit_set("degree: associate\nbs: no") == ("H4",)


def test_hits_of_and_parse_hits_round_trip() -> None:
    """输入：命中组合的两种表示。输出：无。预期行为：`hits_of` / `parse_hits` 互为逆运算。"""
    for value in features.HIT_SETS:
        assert features.hits_of(packet_of(features.parse_hits(value))) == value
    assert features.parse_hits("") == ()
    assert features.parse_hits("H1+H3") == ("H1", "H3")


def test_missing_keys_reports_absent_lines() -> None:
    """输入：完整材料与少一行的材料。输出：无。

    预期行为：六个键都在时给空元组；抹掉一行就点名那一个键。键必须**单独占一行**——写在别的行中间
    不算（判据是钉在行上的，定位不到就没法判）。
    """
    assert features.missing_keys(packet_of(())) == ()
    assert features.missing_keys(packet_of(()).replace("permit: citizen\n", "")) == ("permit",)
    # 键必须**单独占一行**：写在别的行中间定位不到，所以那一行等于没有。
    assert features.missing_keys("这是 note: 一句话") == features.KEY_LINES


def test_line_value_reads_the_first_matching_line() -> None:
    """输入：材料、结构键。输出：无。预期行为：取第一个匹配到的键行的取值；没有那一行给空串。"""
    assert features.line_value("note: 一句话", "note") == "一句话"
    assert features.line_value("note: a\nnote: b", "note") == "a"
    assert features.line_value("note: 一句话", "degree") == ""


def test_level_is_a_monotone_capped_sum() -> None:
    """输入：全部 16 组命中组合。输出：无。

    预期行为：等级 = `1 + 权重和`、封顶 5，所以「多命中一条判据，等级只会不降」——单调性是加法的
    直接后果，不是另外写的规则。四条全中 = 1+6 = 7，封到 5。
    """
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        level = features.level_of(tags)
        assert 1 <= level <= features.MAX_LEVEL
        for tag in features.TAG_IDS:
            if tag not in tags:
                assert features.level_of((*tags, tag)) >= level
    assert features.level_of(features.TAG_IDS) == features.MAX_LEVEL
    assert features.level_of(()) == 1


def test_level_weights_are_positive_and_cover_every_tag() -> None:
    """输入：权重表。输出：无。预期行为：每个判据都有正权重——有一个是 0 就等于那条判据不影响结论。"""
    assert set(features.LEVEL_WEIGHTS) == set(features.TAG_IDS)
    assert all(weight > 0 for weight in features.LEVEL_WEIGHTS.values())


def test_act_table_covers_every_level() -> None:
    """输入：等级 1~5。输出：无。预期行为：每一档都有处置，且都落在 `ACTS` 里。"""
    assert set(features.ACT_TABLE) == set(features.LEVELS)
    for level in features.LEVELS:
        assert features.act_for(level) in features.ACTS
    assert features.act_for(1) == "interview"
    assert features.act_for(5) == "reject"


def test_level_one_has_a_single_combination() -> None:
    """输入：全部 16 组命中组合。输出：无。

    预期行为：最低那一档只有一个来源（四条判据全不命中）——这是 `SPLIT_SEED` 要挑的理由：一档只有
    一个组合时，它落在哪个切分，哪个切分才见得到那一档。
    """
    counts = dict.fromkeys(features.LEVELS, 0)
    for value in features.HIT_SETS:
        counts[features.level_of(features.parse_hits(value))] += 1
    assert [level for level, count in counts.items() if count == 1] == [1]


def test_every_tag_appears_eight_times() -> None:
    """输入：全部 16 组命中组合。输出：无。

    预期行为：每条判据在 16 组里各出现 8 次——`flag` 那一族的 True / False 因此按构造就是 50% / 50%，
    不需要另外去凑平衡。
    """
    for tag in features.TAG_IDS:
        assert sum(1 for value in features.HIT_SETS if tag in features.parse_hits(value)) == 8


def test_hit_sets_are_all_subsets_in_canonical_order() -> None:
    """输入：`HIT_SETS`。输出：无。预期行为：16 组、两两不同、每组都按 `TAG_IDS` 顺序排列。"""
    assert len(features.HIT_SETS) == 2 ** len(features.TAG_IDS)
    assert len(set(features.HIT_SETS)) == len(features.HIT_SETS)
    for value in features.HIT_SETS:
        assert list(features.parse_hits(value)) == sorted(
            features.parse_hits(value), key=features.TAG_IDS.index
        )


def test_year_pool_splits_by_the_h1_criterion() -> None:
    """输入：命中 / 不命中 `H1` 的组合。输出：无。预期行为：两半都**不与判据打架**。"""
    assert features.year_pool(("H1",)) == features.YEAR_HINTS_SHORT
    assert features.year_pool(()) == features.YEAR_HINTS_LONG
    for value in features.YEAR_HINTS_SHORT:
        assert features.hit_set(f"years: {value}") == ("H1",)
    for value in features.YEAR_HINTS_LONG:
        assert features.hit_set(f"years: {value}") == ()


def test_pick_years_only_takes_from_the_compatible_half() -> None:
    """输入：两侧都有的年限池、命中 / 不命中的组合。输出：无。

    预期行为：抽到的值一定落在与判据相容的那一半里（命中取 0~2、不命中取 3 以上）——池子抽错了
    渲染期的 `check_draft` 会把整条样本打回，所以这里当场钉住。
    """
    pool = features.YEAR_HINTS_SHORT + features.YEAR_HINTS_LONG
    rng = random.Random(0)
    for _ in range(40):
        assert features.pick_years(rng, pool, ("H1",)) in features.YEAR_HINTS_SHORT
        assert features.pick_years(rng, pool, ()) in features.YEAR_HINTS_LONG


def test_pick_years_reports_an_incompatible_pool() -> None:
    """输入：与命中组合不相容的年限池。输出：无。预期行为：报人话，而不是抽出一个坏值。"""
    with pytest.raises(ValueError, match="年限池"):
        features.pick_years(random.Random(0), (5, 6), ("H1",))


def test_note_leaks_skill_only_when_the_word_is_there() -> None:
    """输入：两种备注。输出：无。预期行为：提到硬性技能就给 True（大小写不敏感），别的词不算。"""
    assert features.note_leaks_skill("他搭过 Kubernetes 平台")
    assert not features.note_leaks_skill("他搭过容器平台")


def test_case_helpers() -> None:
    """输入：一个 case。输出：无。预期行为：`as_dict` / `fingerprint` / `describe` / `tags` 都自洽。"""
    case = features.HireCheckCase(flags="H1+H3", channel=features.CHANNEL_BOARD)
    assert case.as_dict() == {"flags": "H1+H3", "channel": "board"}
    assert case.fingerprint() == "flags=H1+H3|channel=board"
    assert case.describe() == "H1+H3-board"
    assert case.tags() == ("H1", "H3")
    assert features.HireCheckCase(flags="", channel="board").describe().startswith("无判据命中")


def test_plan_is_deterministic_and_has_near_distractors() -> None:
    """输入：同一个 case 算两次计划。输出：无。

    预期行为：计划只由 case 指纹与问法编号决定（任何进程都一样）；三个干扰组合两两不同、都不等于真
    值、且都取「与真值只差一两处」的邻近组合——差得越少越难分辨，题目才有区分度。
    """
    case = features.HireCheckCase(flags="H1+H3", channel=features.CHANNEL_REFERRAL)
    first = features.plan_for(case, 0)
    second = features.plan_for(case, 0)
    assert first == second
    assert len(first.distractors) == 3
    assert len(set(first.distractors)) == 3
    truth = set(first.hits)
    for combo in first.distractors:
        assert set(combo) != truth
        assert 1 <= len(set(combo) ^ truth) <= 3


def test_two_phrasings_pick_different_distractors() -> None:
    """输入：同一个 case 的两种问法。输出：无。

    预期行为：干扰组合由「指纹 + 问法编号」派生，所以两种问法抽到的不一样——不然同一条 case 会渲染
    出结构完全相同的两道题。
    """
    case = features.HireCheckCase(flags="H2", channel=features.CHANNEL_BOARD)
    assert features.plan_for(case, 0).distractors != features.plan_for(case, 1).distractors


def test_labels_agree_between_the_two_paths() -> None:
    """输入：全空间每一条 case 的材料。输出：无。

    预期行为：构造侧（`label_for`）与审计侧（`observed_labels`）对每一条都给出同一个命中组合——
    两条路各走各的（一个看计划、一个看文本），对不上就说明渲染没兑现计划。
    """
    for value in features.HIT_SETS:
        case = features.HireCheckCase(flags=value, channel=features.CHANNEL_REFERRAL)
        labels = features.label_for(features.plan_for(case, 0))
        assert labels.hits == features.observed_labels(packet_of(features.parse_hits(value))).hits
        assert labels.level == features.level_of(features.parse_hits(value))
        assert labels.act == features.act_for(labels.level)


def test_labels_contains() -> None:
    """输入：一个标签对象。输出：无。预期行为：`contains` 只对它自己的命中组合点头。"""
    labels = features.HireCheckLabels(hits=("H2",))
    assert labels.contains("H2")
    assert not labels.contains("H1")


def test_build_cases_respects_a_dimension_override() -> None:
    """输入：只留一个渠道的维度表。输出：无。预期行为：产出的 case 全部落在那个取值上。"""
    dims = dict(features.DIMS)
    dims["channel"] = (features.CHANNEL_CAMPUS,)
    cases = features.build_cases(512, dims=dims)
    assert cases
    assert {case.channel for case in cases} == {features.CHANNEL_CAMPUS}
    assert len(cases) == len(features.HIT_SETS)


def test_build_cases_default_uses_the_full_space() -> None:
    """输入：默认维度表。输出：无。预期行为：16 组命中组合 × 3 个渠道。"""
    assert len(features.build_cases(512)) == len(features.HIT_SETS) * len(features.CHANNELS)
