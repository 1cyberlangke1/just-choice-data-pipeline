"""testimpact 的特征层测试：触及组合、并集语义、档位阈值、两条标签路径。"""

from __future__ import annotations

from itertools import combinations

import pytest

from tests.workflows.testimpact.conftest import diff_for
from workflows.testimpact import features
from workflows.testimpact.features import ImpactCase


def test_touch_sets_are_the_sixteen_allowed_combinations() -> None:
    """输入：无。输出：无。

    预期行为：16 组 = `P1`~`P3`、`P5` 的全部非空子集（15 组）+ 单独触及盲区（1 组）；没有重复。
    """
    others = [pid for pid in features.PATH_IDS if pid != features.BLIND_PATH]
    expected = {
        "+".join(combo)
        for size in range(1, len(others) + 1)
        for combo in combinations(others, size)
    } | {features.BLIND_PATH}
    assert set(features.TOUCH_SETS) == expected
    assert len(features.TOUCH_SETS) == 16
    assert len(set(features.TOUCH_SETS)) == 16


def test_each_path_appears_the_expected_number_of_times() -> None:
    """输入：无。输出：无。

    预期行为：`P1`~`P3`、`P5` 各出现 8 次、盲区 1 次——`flag` 那一族的 True 占比就是靠这个
    落在 9/16（每条路径的覆盖集都只有一条路径，再加「单独触及盲区」要跑全套），而不是靠运气。
    """
    counts = {pid: 0 for pid in features.PATH_IDS}
    for touch in features.TOUCH_SETS:
        for pid in features.parse_touch(touch):
            counts[pid] += 1
    assert counts == {"P1": 8, "P2": 8, "P3": 8, "P4": 1, "P5": 8}


@pytest.mark.parametrize(
    ("touch", "need"),
    [
        ("", ()),
        ("P1", ("unit",)),
        ("P2", ("integration",)),
        ("P3", ("e2e",)),
        ("P5", ()),
        ("P1+P2", ("unit", "integration")),
        ("P1+P5", ("unit",)),
        ("P1+P2+P3", ("unit", "integration", "e2e")),
        ("P4", ("unit", "integration", "e2e")),
    ],
)
def test_need_is_a_union_over_the_touched_paths(touch: str, need: tuple[str, ...]) -> None:
    """输入：触及组合。输出：无。

    预期行为：需要跑的套件是**并集**（`P1+P5` 只要一套，文档路径不挂套件）；触及盲区（`P4`）
    直接给全部三套——影响面未知，只能全套跑。
    """
    assert features.need_suites(features.parse_touch(touch)) == need


def test_blind_path_and_docs_path_differ_only_by_kind() -> None:
    """输入：无。输出：无。

    预期行为：盲区（`P4`）与文档路径（`P5`）的覆盖列都是空的，但类型不同、结论相反——这一条
    是本 workflow 的核心读法，改覆盖表时先看它。
    """
    assert features.COVERAGE[features.BLIND_PATH] == ()
    assert features.PATH_KINDS[features.BLIND_PATH] == features.KIND_CODE
    assert features.PATH_KINDS["P5"] == features.KIND_DOCS
    assert features.COVERAGE["P5"] == ()
    assert features.need_suites([features.BLIND_PATH]) == features.SUITES
    assert features.need_suites(["P5"]) == ()


@pytest.mark.parametrize(
    ("count", "level"),
    [(0, "smoke"), (1, "targeted"), (2, "pair"), (3, "full")],
)
def test_level_thresholds(count: int, level: str) -> None:
    """输入：需要跑的套数。输出：无。预期行为：0/1/2/3 套各对应一档，四档全可达。"""
    assert features.level_for(features.SUITES[:count]) == level


def test_every_level_and_count_is_reachable_in_the_full_space() -> None:
    """输入：无。输出：无。

    预期行为：16 组触及组合把四个档位与四种套数都取到至少一次——取不到就说明某一族题目按构造
    出不来（均衡闸与审计都会报），这一条在小批量之前先把它拦住。
    """
    cases = features.build_cases(512, dims={"touch": features.TOUCH_SETS, "flavor": ("fix",)})
    assert {case.level() for case in cases} == set(features.LEVELS)
    assert {len(case.need()) for case in cases} == {0, 1, 2, 3}


@pytest.mark.parametrize(
    ("touch", "expect"),
    [
        ("", ()),
        ("P1", ("P1",)),
        ("P1+P3", ("P1", "P3")),
        ("P3+P1", ("P1", "P3")),
        ("P5+P2", ("P2", "P5")),
        ("P9", ("P9",)),
        ("P2+P9", ("P2", "P9")),
    ],
)
def test_touch_key_is_canonical_and_keeps_unknown_ids(touch: str, expect: tuple[str, ...]) -> None:
    """输入：触及组合串。输出：无。

    预期行为：按 `PATH_IDS` 顺序规范化（`P3+P1` 与 `P1+P3` 同串），认不出的编号留在后面——
    审计要靠它报「改了覆盖表之外的路径」。
    """
    assert features.touch_key(features.parse_touch(touch)) == "+".join(expect)


@pytest.mark.parametrize(
    ("diff", "expect"),
    [
        ("", ()),
        ("diff --git a/src/api/handlers.py b/src/api/handlers.py\n", ("P1",)),
        (
            "diff --git a/src/api/handlers.py b/src/api/handlers.py\n"
            "diff --git a/src/db/schema.py b/src/db/schema.py\n",
            ("P1", "P2"),
        ),
        # 只有 `---` / `+++` 不算：落点只认 `diff --git` 那一行（散文式描述读不出落点）。
        ("--- a/src/api/handlers.py\n+++ b/src/api/handlers.py\n", ()),
        ("改了 src/api/handlers.py 里的边界判断。\n", ()),
        ("diff --git a/src/other/thing.py b/src/other/thing.py\n", ("src/other/thing.py",)),
    ],
)
def test_touched_in_reads_only_the_diff_header(diff: str, expect: tuple[str, ...]) -> None:
    """输入：改动正文。输出：无。

    预期行为：只认 `diff --git a/<路径> b/<路径>` 那一行的新路径；`---` / `+++` 与散文一律不算；
    覆盖表之外的文件原样报出来。
    """
    assert features.touched_in(diff) == expect


def test_two_label_paths_agree_on_every_faithful_diff() -> None:
    """输入：16 组触及组合各造一份兑现计划的改动。输出：无。

    预期行为：构造侧（从计划推）与审计侧（从文本重推）逐组一致——两条独立路径互相印证，渲染
    要是没兑现计划，这一条会红。
    """
    for touch in features.TOUCH_SETS:
        case = ImpactCase(touch=touch, flavor="fix")
        plan = features.plan_for(case, 0)
        labels = features.label_for(plan)
        assert features.observed_labels(diff_for(case)) == labels


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一 case 两次、不同问法各一次。输出：无。

    预期行为：计划只由「case 指纹 + 问法编号」决定（与进程、语言无关）；三个干扰集合与真值不同、
    彼此不同，且都取「只差一两套」的邻近集合。
    """
    case = ImpactCase(touch="P1+P2", flavor="feat")
    first = features.plan_for(case, 1)
    second = features.plan_for(case, 1)
    assert first == second
    truth = set(first.need)
    assert len(first.distractors) == 3
    assert len(set(first.distractors)) == 3
    for combo in first.distractors:
        assert set(combo) != truth
        assert len(set(combo) ^ truth) <= 2


def test_build_cases_order_is_stable() -> None:
    """输入：同一份维度表算两次。输出：无。预期行为：逐项一致（case 库要能逐字节重建）。"""
    first = features.build_cases(512)
    second = features.build_cases(512)
    assert [case.as_dict() for case in first] == [case.as_dict() for case in second]
    assert len(first) == len(features.TOUCH_SETS) * len(features.ALL_FLAVORS)


def test_case_describe_and_fingerprint() -> None:
    """输入：一个 case。输出：无。预期行为：描述可读、指纹稳定、派生属性与纯函数一致。"""
    case = ImpactCase(touch="P1+P3", flavor="revert")
    assert case.describe() == "P1+P3-revert"
    assert case.fingerprint() == "touch=P1+P3|flavor=revert"
    assert case.touched() == ("P1", "P3")
    assert case.need() == ("unit", "e2e")
    assert case.level() == "pair"
    assert case.blind() is False
    assert ImpactCase(touch="P4", flavor="fix").blind() is True
