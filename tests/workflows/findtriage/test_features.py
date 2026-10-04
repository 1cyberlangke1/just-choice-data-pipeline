"""改动引入性判定的特征层测试：取值表、被禁取值对、计划兑现与口径表。"""

from __future__ import annotations

import collections
import itertools

from workflows.findtriage import features as F
from workflows.findtriage.features import RecordCase

CASES = F.build_cases(512)


def test_symbol_pool_is_near_form_safe() -> None:
    """输入：无。输出：无。

    预期行为：池子里每个名字都**不以 `s` 结尾**，所以 `near_form` 补出来的名字既不在池子里、
    也不等于别的池子成员——诱饵行「像而不同」全靠这条不变量。
    """
    assert len(set(F.SYMBOLS)) == len(F.SYMBOLS)
    for symbol in F.SYMBOLS:
        assert not symbol.endswith("s"), symbol
        assert F.near_form(symbol) not in F.SYMBOLS, symbol


def test_cases_exclude_the_forbidden_pair() -> None:
    """输入：全枚举。输出：无。

    预期行为：`cite=missing` 与 `evidence=in_row` 这一对不出现（指向的行都不在记录里了，
    就不可能有「指向那一行的符号」），而且它**不算没覆盖**——所以组合数是 3×3×2×3×2−2×3×2。
    """
    assert len(CASES) == 3 * 3 * 2 * 3 * 2 - 2 * 3 * 2
    assert all(F.compatible(case) for case in CASES)
    assert all(not (case.cite == "missing" and case.evidence == "in_row") for case in CASES)
    assert not F.allow_pair("cite", "missing", "evidence", "in_row")
    assert not F.allow_pair("evidence", "in_row", "cite", "missing")
    assert F.allow_pair("cite", "missing", "evidence", "elsewhere")


def test_every_pair_of_values_is_covered() -> None:
    """输入：全枚举。输出：无。预期行为：任意两维的任意允许取值对至少出现一次。"""
    names = list(F.DIMS)
    seen: set[tuple[str, str, str, str]] = set()
    for case in CASES:
        values = case.as_dict()
        for left, right in itertools.combinations(names, 2):
            seen.add((left, str(values[left]), right, str(values[right])))
    for left, right in itertools.combinations(names, 2):
        for lv in F.DIMS[left]:
            for rv in F.DIMS[right]:
                if not F.allow_pair(left, lv, right, rv):
                    continue
                key = (left, str(lv), right, str(rv))
                assert key in seen, key


def test_every_dimension_is_realized_by_the_plan() -> None:
    """输入：全枚举 × 三个问法。输出：无。

    预期行为：五个注入维度都**按构造**兑现——表长、指向那一行的处境、证据符号的落点、
    修法点不点名、有没有近形诱饵行，逐条核对，一个 case 都不放过。
    """
    for case in CASES:
        for phrasing in (0, 1, 2):
            plan = F.plan_for(case, phrasing)
            labels = F.label_for(plan)
            lines = [row.line for row in plan.rows]
            symbols = [row.symbol for row in plan.rows]
            assert len(plan.rows) == case.size, (case, phrasing)
            assert len(set(lines)) == case.size
            assert len(set(symbols)) == case.size
            if case.cite == "missing":
                assert plan.cited_index is None
                assert plan.cited_line not in set(lines)
            else:
                row = plan.rows[plan.cited_index]
                assert plan.cited_line == row.line
                touched = row.touch in F.TOUCHED_KINDS
                assert touched is (case.cite == "touched"), (case, phrasing)
            inside = set(symbols)
            if case.evidence == "in_row":
                assert plan.evidence_symbol == plan.rows[plan.cited_index].symbol
            elif case.evidence == "elsewhere":
                assert plan.evidence_symbol in inside
                if plan.cited_index is not None:
                    assert plan.evidence_symbol != plan.rows[plan.cited_index].symbol
            else:
                assert plan.evidence_symbol not in inside
            assert labels.actionable is (case.fix == "concrete")
            assert labels.decoy is (case.decoy == 1)


def test_plan_is_a_pure_function_of_case_and_phrasing() -> None:
    """输入：同一个 case 取两次计划。输出：无。预期行为：完全相同（派生而不是存下来的）。"""
    case = CASES[0]
    assert F.plan_for(case, 1) == F.plan_for(case, 1)


def test_the_fix_always_names_the_quoted_symbol() -> None:
    """输入：全枚举 × 三个问法。输出：无。

    预期行为：修法点名的符号**就是**发现引用的证据符号——审查意见的修法针对的是它自己引用
    的那个东西。这条不是风格：真机批次里反解正是拿「修法点的符号与发现引用的不是同一个」
    当成了「这条修法落不了地」，把 `actionable` 判反（见范式 §9.78 第三条）。
    """
    for case in CASES:
        plan = F.plan_for(case, 0)
        assert plan.fix_symbol == plan.evidence_symbol, case


def test_labels_follow_the_printed_rules() -> None:
    """输入：几组构造出来的三元组。输出：无。

    预期行为：口径**从上往下第一条命中**——空口（证据查不到）先判丢弃，其次不归本次改动、
    引入性说不清、修法落不了地都只记录，其余才必须修。顺序写反了这条会红。
    """
    assert F.triage_for("yes", "none", True) == "drop"
    assert F.triage_for("no", "strong", True) == "log_only"
    assert F.triage_for("unknown", "strong", True) == "log_only"
    assert F.triage_for("yes", "strong", False) == "log_only"
    assert F.triage_for("yes", "strong", True) == "must_fix"
    assert F.triage_for("yes", "weak", True) == "must_fix"


def test_triage_buckets_are_all_reachable_on_both_sides() -> None:
    """输入：全枚举 × 三个问法。输出：无。

    预期行为：三档处置都到得了，而且**没有一档吃掉一大半**——分布大致落在
    37.5% / 50% / 12.5% 上（口径表的排序就是为这个选的，见 `features` 模块注释）。
    """
    tally = collections.Counter(
        F.label_for(F.plan_for(case, phrasing)).triage
        for case in CASES
        for phrasing in (0, 1, 2)
    )
    total = sum(tally.values())
    assert set(tally) == set(F.TRIAGES)
    for bucket in F.TRIAGES:
        assert tally[bucket] / total >= 0.1, (bucket, tally)
    assert tally["log_only"] / total <= 0.6, tally


def test_introduced_and_supported_are_recomputed_from_the_table() -> None:
    """输入：几个计划。输出：无。

    预期行为：`introduced_for` / `supported_for` 只吃「表 + 行号 + 符号」，所以审计拿产物里
    解析出来的表也能算出同一个答案——这条保证「审计重算」与「合成期算标签」是同一套口径。
    """
    case = RecordCase(cite="touched", evidence="elsewhere", fix="concrete", size=4, decoy=1)
    plan = F.plan_for(case, 0)
    assert F.introduced_for(plan.rows, plan.cited_line) == "yes"
    assert F.introduced_for(plan.rows, plan.cited_line + 999) == "unknown"
    assert F.supported_for(plan.rows, plan.cited_line, plan.evidence_symbol) == "weak"
    assert F.supported_for(plan.rows, plan.cited_line, "no_such_symbol") == "none"
    assert F.supported_for(plan.rows, plan.cited_line, plan.rows[plan.cited_index].symbol) == "strong"


def test_cite_key_and_near_form() -> None:
    """输入：几个取值。输出：无。预期行为：选项名与近形符号的拼法固定。"""
    assert F.cite_key(None) == F.NO_CITE
    assert F.cite_key(1) == "s1"
    assert F.near_form("parse_timeout") == "parse_timeouts"


def test_owner_for_follows_the_priority_table() -> None:
    """输入：几组构造出来的记录表 + 行号 + 符号。输出：无。

    预期行为：归属口径**从上往下第一条命中**——行号查不到先回记录维护者，其次符号查不到回
    规则维护者，再次本次碰过交作者，其余交架构组。顺序写反了这条会红。
    """
    rows = (
        F.RecordRow(line=10, touch="added", symbol="parse_timeout"),
        F.RecordRow(line=20, touch="untouched", symbol="retry_budget"),
    )
    assert F.owner_for(rows, 999, "parse_timeout") == F.OWNER_RECORD
    assert F.owner_for(rows, 10, "no_such_symbol") == F.OWNER_RULE
    assert F.owner_for(rows, 10, "parse_timeout") == F.OWNER_AUTHOR
    assert F.owner_for(rows, 20, "retry_budget") == F.OWNER_ARCH


def test_every_owner_is_reachable_across_the_case_space() -> None:
    """输入：全枚举 × 三个问法。输出：无。

    预期行为：四个去处**每一个都真的取到过**，而且没有哪一档吃掉一大半——有一档取不到，
    那一族的选项就是死选项；有一档吃掉大半，这一族就没有区分度。
    """
    tally = collections.Counter(
        F.label_for(F.plan_for(case, phrasing)).owner
        for case in CASES
        for phrasing in (0, 1, 2)
    )
    total = sum(tally.values())
    assert set(tally) == set(F.OWNERS), tally
    for owner in F.OWNERS:
        assert tally[owner] / total >= 0.05, (owner, tally)
    assert max(tally.values()) / total <= 0.7, tally


def test_dimension_tables_are_declared_consistently() -> None:
    """输入：无。输出：无。预期行为：`DIMS` 的键与取值表一一对应，`RULES` 顺序固定。"""
    assert list(F.DIMS) == ["cite", "evidence", "fix", "size", "decoy"]
    assert F.DIMS["cite"] == F.CITES
    assert F.DIMS["evidence"] == F.EVIDENCES
    assert F.DIMS["fix"] == F.FIXES
    assert F.DIMS["size"] == F.SIZES
    assert F.DIMS["decoy"] == F.DECOYS
    assert F.RULES == ("baseless", "attribution", "unknown", "fix", "else")
    assert F.RECORD_COLUMNS == ("line", "touch", "symbol")
