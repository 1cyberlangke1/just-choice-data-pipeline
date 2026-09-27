"""真值规格：规则判据、门控分档、case 矩阵与行计划。"""

from __future__ import annotations

import pytest

from workflows.lint import spec as sp
from workflows.lint.spec import LintCase, LintConfig

# ---------- 规则判据 ----------


@pytest.mark.parametrize(
    ("rule", "line", "expected"),
    [
        ("secret", "API_KEY = \"sk-live-9f2b\"", True),
        # 走环境变量不算硬编码——背景行正是靠这条规矩躲开的
        ("secret", "API_KEY = os.environ[\"API_KEY\"]", False),
        ("secret", "password = get_secret()", False),
        ("todo", "# TODO: 补上单元测试", True),
        ("todo", "// FIXME: 边界条件", True),
        ("todo", "# 这里以后再补", False),
        ("debug", "print(\"loading\")", True),
        ("debug", "logger.debug(\"cache miss\")", True),
        ("debug", "logger.info(\"ok\")", False),
        ("path", "CONFIG = \"/etc/app/config.yml\"", True),
        ("path", "ROOT = \"C:\\\\projects\\\\app\"", True),
        ("path", "CONFIG = \"config.yml\"", False),
    ],
)
def test_matches_known_rules(rule: str, line: str, expected: bool) -> None:
    """输入：规则名、一行文本、期望。输出：无。预期行为：判据与文案表里的实例一致。"""
    assert sp.matches(rule, line) is expected


def test_matches_long_is_a_length_rule() -> None:
    """输入：长度恰好等于 / 超过阈值的两行。输出：无。预期行为：严格大于才算命中。"""
    assert sp.matches("long", "x" * sp.LONG_LIMIT) is False
    assert sp.matches("long", "x" * (sp.LONG_LIMIT + 1)) is True


def test_matches_unknown_rule_reports_name() -> None:
    """输入：不存在的规则名。输出：无。预期行为：报人话并列出可用规则，不静默返回 False。"""
    with pytest.raises(KeyError, match="没有规则"):
        sp.matches("nope", "x")


# ---------- 门控分档 ----------


def test_gate_bands_are_three_equal_thirds() -> None:
    """输入：0~8 全部命中档。输出：无。

    预期行为：三档各占 1/3——这是 `gate` 那一族答案均衡的**唯一**来源（`hits` 均匀取 0~8，
    9 个档恰好三等分）。改了边界就必须同时改题面文案，这条会先红。
    """
    bands = [sp.gate_of(hits) for hits in sp.HITS]
    assert bands.count("pass") == 3
    assert bands.count("review") == 3
    assert bands.count("block") == 3
    assert bands[sp.GATE_PASS_MAX] == "pass"
    assert bands[sp.GATE_PASS_MAX + 1] == "review"
    assert bands[sp.GATE_REVIEW_MAX + 1] == "block"


@pytest.mark.parametrize("hits", [-1, sp.SIZE + 1])
def test_gate_rejects_out_of_range(hits: int) -> None:
    """输入：越界的命中行数。输出：无。预期行为：报人话。"""
    with pytest.raises(ValueError, match="命中行数"):
        sp.gate_of(hits)


# ---------- case 与素材 ----------


def test_material_fingerprint_covers_every_dimension() -> None:
    """输入：四个只差一维的 case。输出：无。

    预期行为：指纹逐维不同——指纹是切分分桶的键，漏一维会让「只有该维不同」的 case
    被当成同一份素材，切分之间立刻撞车。
    """
    base = LintCase("secret", "code", 1, 0)
    variants = [
        LintCase("todo", "code", 1, 0),
        LintCase("secret", "log", 1, 0),
        LintCase("secret", "code", 2, 0),
        LintCase("secret", "code", 1, 1),
    ]
    prints = {sp.material_of(base)} | {sp.material_of(case) for case in variants}
    assert len(prints) == 5


def test_unseen_dims_names_holdout_values() -> None:
    """输入：各含一个留出取值的 case。输出：无。预期行为：点名到具体维度，顺序固定。"""
    assert LintCase("long", "code", 1, 0).unseen_dims() == ("rule",)
    assert LintCase("secret", "doc", 1, 0).unseen_dims() == ("genre",)
    assert LintCase("secret", "code", 1, 1).unseen_dims() == ("phrasing",)
    assert LintCase("long", "doc", 8, 1).unseen_dims() == ("rule", "genre", "phrasing")
    assert LintCase("secret", "code", 8, 0).unseen_dims() == ()


def test_case_describe_is_readable() -> None:
    """输入：一个 case。输出：无。预期行为：描述里四个维度都在。"""
    text = LintCase("todo", "log", 3, 1).describe()
    assert text == "todo/log/3/1"


def test_case_as_dict_flattens_dimensions() -> None:
    """输入：一个 case。输出：无。预期行为：四个维度都在，值是原始类型（hits 是整数）。"""
    data = LintCase("path", "doc", 5, 1).as_dict()
    assert data == {"rule": "path", "genre": "doc", "hits": 5, "phrasing": 1}


def test_build_cases_is_the_full_product() -> None:
    """输入：全开的采样参数。输出：无。预期行为：四张表的全组合，顺序稳定。"""
    cases = sp.build_cases(LintConfig())
    assert len(cases) == len(sp.RULES) * len(sp.GENRES) * len(sp.HITS) * len(sp.PHRASINGS)
    assert cases[0] == LintCase(sp.RULES[0], sp.GENRES[0], 0, 0)
    assert cases == sp.build_cases(LintConfig())


def test_build_cases_ood_only_keeps_cases_with_an_unseen_value() -> None:
    """输入：`ood_only=True`。输出：无。

    预期行为：留下的每条都至少含一个留出取值，且条数 = 全量 − 同分布量
    （同分布量就是「训练侧见过的取值」的全组合）。
    """
    ood = sp.build_cases(LintConfig(ood_only=True))
    assert ood and all(case.unseen_dims() for case in ood)
    id_count = len(sp.SEEN_RULES) * len(sp.SEEN_GENRES) * len(sp.HITS) * len(sp.SEEN_PHRASINGS)
    assert len(ood) == len(sp.build_cases(LintConfig())) - id_count


def test_empty_pool_reports_the_field() -> None:
    """输入：空取值表。输出：无。预期行为：报人话并点名是哪个字段。"""
    with pytest.raises(ValueError, match="rules"):
        LintConfig(rules=())


# ---------- 行计划 ----------


def test_plan_hit_count_matches_the_case() -> None:
    """输入：0~8 全部命中档。输出：无。

    预期行为：计划里的命中行数就是 case 的 `hits`，且行槽两两不重复（重复会让同一行
    出现两次，`spot` 的选项描述撞车）。
    """
    for hits in sp.HITS:
        plan = sp.build_plan(LintCase("todo", "code", hits, 0))
        assert plan.hits == hits
        assert len(plan.entries) == sp.SIZE
        slots = [(entry.kind, entry.slot) for entry in plan.entries]
        assert len(set(slots)) == sp.SIZE
        assert plan.matched() == tuple(entry.kind == "hit" for entry in plan.entries)


def test_plan_is_derived_from_the_fingerprint_only() -> None:
    """输入：同一 case 造两次、不同切分与序号。输出：无。

    预期行为：计划只由素材指纹决定——切分与序号都不参与，否则「同素材同题必须同答」
    这条一致性校验会挂。
    """
    case = LintCase("debug", "log", 4, 1)
    first = sp.build_plan(case)
    assert first == sp.build_plan(case)
    assert sp.sample_spec(case, "train", 0).plan == sp.sample_spec(case, "test_ood", 7).plan


def test_plan_positions_differ_between_neighbouring_cases() -> None:
    """输入：只差命中行数的两个 case。输出：无。

    预期行为：行模式不是同一套位置换个数——那会让「命中行永远在前几行」成为捷径。
    这里只钉住「不是恒等映射」，不假设具体分布。
    """
    one = sp.build_plan(LintCase("secret", "code", 1, 0))
    two = sp.build_plan(LintCase("secret", "code", 2, 0))
    assert one.matched() != two.matched()[: sp.SIZE]


def test_spec_carries_split_and_seq() -> None:
    """输入：一个 case、切分名、序号。输出：无。预期行为：规格如实带上它们。"""
    spec = sp.sample_spec(LintCase("path", "doc", 0, 1), "val", 3)
    assert spec.split == "val" and spec.seq == 3
    assert spec.case == LintCase("path", "doc", 0, 1)
