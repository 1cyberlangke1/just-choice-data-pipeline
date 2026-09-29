"""行级规则检查的特征层测试：规则判据、行计划、标签与 case 空间的均衡性。"""

from __future__ import annotations

import pytest

from workflows.lint import features

#: 全部 case（不剔除任何取值）：5 条规则 × 3 条体裁 × 9 个命中行数。
ALL_CASES = features.build_cases(10_000)

#: 全部 case × 全部问法。
ALL_PLANS = [
    features.plan_for(case, phrasing)
    for case in ALL_CASES
    for phrasing in (0, 1)
]


def test_long_uses_character_count() -> None:
    """输入：正好 60 与 61 个字符的行。输出：无。预期行为：只有 61 个字符那行命中。"""
    assert features.LONG_LIMIT == 60
    assert not features.matches(features.RULE_LONG, "x" * features.LONG_LIMIT)
    assert features.matches(features.RULE_LONG, "x" * (features.LONG_LIMIT + 1))


@pytest.mark.parametrize(
    ("rule", "line", "expected"),
    [
        (features.RULE_SECRET, 'API_KEY = "sk-live-9f2b2c"', True),
        (features.RULE_SECRET, 'DB_PASSWORD: "hunter2-x"', True),
        (features.RULE_SECRET, "API_KEY = os.environ[\"API_KEY\"]", False),
        (features.RULE_SECRET, "API_KEY = get_secret()", False),
        (features.RULE_SECRET, 'API_KEY = "ab"', False),
        (features.RULE_SECRET, "timeout = 30", False),
        (features.RULE_TODO, "# TODO: retry the upload", True),
        (features.RULE_TODO, "// fixme: lock order", True),
        (features.RULE_TODO, "xxx = 1", True),
        (features.RULE_TODO, "the task list is empty", False),
        # 判据是「独立的词」，所以小写的 todo 照样算命中——这条钉住大小写不敏感。
        (features.RULE_TODO, "the todo list is empty", True),
        (features.RULE_DEBUG, 'print("loading")', True),
        (features.RULE_DEBUG, "console.log(payload)", True),
        (features.RULE_DEBUG, 'logger.debug("retry")', True),
        # `debugger` 自己就是一条调试语句，不需要括号——真机第一批里模型写 `debugger;` 被拒过。
        (features.RULE_DEBUG, "debugger; // pause here", True),
        (features.RULE_DEBUG, "// the debugger is attached", True),
        # Go 那一家：`fmt.Println` / `fmt.Printf` 才是模型真会写的调试输出，判据与正则都收
        # （真机 lint 批次里 `fmt.Println(status)` 被判「不命中」而拒稿，2026-09-29）。
        (features.RULE_DEBUG, "fmt.Println(status)", True),
        (features.RULE_DEBUG, 'fmt.Printf("%d\\n", count)', True),
        (features.RULE_DEBUG, "fmt.Print(count)", True),
        # `Sprintf` 只是拼字符串，不是调试输出——别把 Print 家族一锅端。
        (features.RULE_DEBUG, 'label = fmt.Sprintf("%d", count)', False),
        (features.RULE_DEBUG, 'logger.info("cache warm")', False),
        (features.RULE_DEBUG, "printer = 1", False),
        (features.RULE_PATH, 'CONFIG = "/etc/app/config.yml"', True),
        (features.RULE_PATH, 'ROOT = "C:\\projects\\app"', True),
        (features.RULE_PATH, 'CONFIG = "etc/app/config.yml"', False),
        (features.RULE_PATH, 'CONFIG = os.environ["CONFIG"]', False),
    ],
)
def test_pattern_rules(rule: str, line: str, expected: bool) -> None:
    """输入：规则名、一行文本、期望。输出：无。预期行为：命中与否与期望一致。"""
    assert features.matches(rule, line) is expected


def test_unknown_rule_is_an_error() -> None:
    """输入：不存在的规则名。输出：无。预期行为：抛 KeyError，不静默返回 False。"""
    with pytest.raises(KeyError, match="没有规则"):
        features.matches("nope", "anything")


def test_match_any_takes_the_first_rule_in_table_order() -> None:
    """输入：同时命中两条规则的行。输出：无。预期行为：按 `RULES` 顺序给第一条。"""
    line = 'API_KEY = "sk-live-9f2b2c"  # TODO'
    assert features.matches(features.RULE_SECRET, line)
    assert features.matches(features.RULE_TODO, line)
    assert features.match_any(line) == features.RULE_SECRET
    assert features.match_any("import os") is None


@pytest.mark.parametrize(
    ("hits", "expected"),
    [(0, "pass"), (1, "pass"), (2, "pass"), (3, "review"), (5, "review"), (6, "block"), (8, "block")],
)
def test_gate_bands(hits: int, expected: str) -> None:
    """输入：命中行数、期望档位。输出：无。预期行为：三档边界正好落在 2 / 5 / 6。"""
    assert features.gate_of(hits) == expected


@pytest.mark.parametrize("hits", [-1, features.SIZE + 1])
def test_gate_rejects_out_of_range(hits: int) -> None:
    """输入：越界的命中行数。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="命中行数"):
        features.gate_of(hits)


def test_plan_is_deterministic_and_sorted() -> None:
    """输入：同一个 case。输出：无。

    预期行为：同一 (case, 问法) 必得同一份计划；命中行下标升序、不重复、落在 0~7 里。
    """
    case = features.LintCase(features.RULE_TODO, "code", 3)
    first = features.plan_for(case, 0)
    assert first == features.plan_for(case, 0)
    assert first.hit_rows == tuple(sorted(first.hit_rows))
    assert len(set(first.hit_rows)) == 3
    assert all(0 <= row < features.SIZE for row in first.hit_rows)
    assert first.hits == 3
    assert sum(first.matched()) == 3


def test_plan_differs_across_phrasings() -> None:
    """输入：同一个 case 的两个问法。输出：无。

    预期行为：行计划不一样——一样的话这两种问法会渲染出逐字节相同的片段，样本就成了重复。
    """
    case = features.LintCase(features.RULE_PATH, "log", 4)
    assert features.plan_for(case, 0).hit_rows != features.plan_for(case, 1).hit_rows


@pytest.mark.parametrize("hits", features.HITS)
def test_label_matched_length_matches_plan(hits: int) -> None:
    """输入：每个命中行数。输出：无。预期行为：逐行标记长度是 8、True 的个数等于 hits。"""
    labels = features.label_for(features.plan_for(features.LintCase("debug", "doc", hits), 0))
    assert len(labels.matched) == features.SIZE
    assert sum(labels.matched) == hits
    assert labels.hits == hits
    assert labels.gate == features.gate_of(hits)


def test_spot_is_only_set_for_a_single_hit() -> None:
    """输入：0~8 行的计划。输出：无。预期行为：只有恰好一行命中时 `spot` 才有值。"""
    for hits in features.HITS:
        labels = features.label_for(features.plan_for(features.LintCase("path", "code", hits), 0))
        if hits == 1:
            assert labels.spot is not None
            assert labels.matched[labels.spot - 1] is True
        else:
            assert labels.spot is None


def test_full_space_has_one_case_per_combination() -> None:
    """输入：全空间。输出：无。预期行为：5×3×9 = 135 组，顺序稳定且互不重复。"""
    assert len(ALL_CASES) == len(features.RULES) * len(features.GENRES) * len(features.HITS)
    assert len({case.describe() for case in ALL_CASES}) == len(ALL_CASES)
    assert features.build_cases(10_000) == ALL_CASES


def test_restricted_dims_shrink_the_space() -> None:
    """输入：剔除一个体裁的维度表。输出：无。预期行为：组合数按比例缩小。"""
    dims = dict(features.DIMS)
    dims["genre"] = ("code", "log")
    cases = features.build_cases(10_000, dims=dims)
    assert len(cases) == len(features.RULES) * 2 * len(features.HITS)
    assert all(case.genre in {"code", "log"} for case in cases)


def test_hits_values_are_equally_common() -> None:
    """输入：全空间。输出：无。预期行为：9 个命中行数各出现 15 次（均衡由构造保证）。"""
    tally = dict.fromkeys(features.HITS, 0)
    for case in ALL_CASES:
        tally[case.hits] += 1
    assert set(tally.values()) == {len(ALL_CASES) // len(features.HITS)}


def test_line_answers_are_half_true() -> None:
    """输入：全空间 × 全部问法。输出：无。预期行为：`line` 族 True 的占比正好 1/2。"""
    total = 0
    true_count = 0
    for plan in ALL_PLANS:
        for hit in plan.matched():
            total += 1
            true_count += int(hit)
    assert total > 0
    assert true_count * 2 == total


def test_gate_answers_split_into_three_equal_bands() -> None:
    """输入：全空间 × 全部问法。输出：无。预期行为：三档各占 1/3。"""
    tally: dict[str, int] = dict.fromkeys(features.GATES, 0)
    for plan in ALL_PLANS:
        tally[features.gate_of(plan.hits)] += 1
    assert len(set(tally.values())) == 1


def test_spot_positions_cover_all_eight_rows() -> None:
    """输入：全部「恰好一行命中」的计划。输出：无。预期行为：八个行号都出现得到。"""
    spots = {
        features.label_for(plan).spot
        for plan in ALL_PLANS
        if plan.hits == 1
    }
    assert spots == set(range(1, features.SIZE + 1))


def test_every_rule_and_genre_is_reachable() -> None:
    """输入：全空间。输出：无。预期行为：五条规则、三条体裁都到得了。"""
    assert {case.rule for case in ALL_CASES} == set(features.RULES)
    assert {case.genre for case in ALL_CASES} == set(features.GENRES)


def test_dims_table_covers_the_declared_values() -> None:
    """输入：无。输出：无。预期行为：维度表与三条取值表一一对得上。"""
    assert set(features.DIMS) == {"rule", "genre", "hits"}
    assert features.DIMS["rule"] == features.RULES
    assert features.DIMS["genre"] == features.GENRES
    assert tuple(features.DIMS["hits"]) == features.HITS
    assert set(features.PATTERNS) == set(features.RULES) - {features.RULE_LONG}
