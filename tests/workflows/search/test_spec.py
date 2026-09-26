"""结果页规格的单元测试：角色表、配方过滤、版式几何、日期顺序、旋钮。"""

from __future__ import annotations

import datetime as dt
import random
from collections import Counter

import pytest

from workflows.search import locales, questions
from workflows.search import spec as sp
from workflows.search import workflow as wf


def truth_rows(spec: sp.SearchSpec, locale: str = "zh") -> tuple[sp.TruthRow, ...]:
    """输入：规格、语言。输出：构造侧的「可判定事实」（借出题模块那份实现）。"""
    return questions._truth_rows(spec, locales.load(locale))


def test_wrong_kind_never_equals_the_right_one() -> None:
    """输入：全部文档类型。输出：无。预期行为：换出来的类型一定不是原来那个。"""
    for kind in sp.KINDS:
        assert sp.wrong_kind_of(kind) != kind
        assert sp.wrong_kind_of(kind) in sp.KINDS


def test_mix_fits_rejects_distractors_that_beat_the_best() -> None:
    """输入：全部 (配方, 最佳结果角色) 组合。输出：无。

    预期行为：配方里任何一条的相关度都不得高于最佳结果——否则最佳结果不是最优，
    `best_result` 的答案就不唯一了。
    """
    for mix in sp.MIXES:
        for best in sp.BEST_FLAVORS:
            top = sp.FLAVOR_BY_NAME[best].relevance
            worst = max(sp.FLAVOR_BY_NAME[name].relevance for name in mix)
            assert sp.mix_fits(mix, best) == (worst <= top), (mix, best)
    # `stale` 那一档必须真有配方可用，否则它一条 case 都造不出来。
    assert [mix for mix in sp.MIXES if sp.mix_fits(mix, "stale")]


def test_build_cases_geometry_is_sane() -> None:
    """输入：全取值表的 case 矩阵。输出：无。

    预期行为：被点名的两条与最佳结果位置两两不同、都在这一页的编号范围内；
    同一条 case 不会重复出现。
    """
    cases = sp.build_cases(sp.SearchConfig())
    assert cases
    for case in cases:
        assert case.count == len(case.mix) + 3
        assert case.best_at not in case.pair
        assert max(*case.pair, case.best_at, case.probe) <= case.count
        assert case.probe >= 1
    assert len(set(cases)) == len(cases)


def test_ood_only_keeps_cases_with_an_unseen_value() -> None:
    """输入：全取值表 + `ood_only`。输出：无。

    预期行为：留下来的每条 case 都至少有一个维度取的是训练侧没见过的值，
    而且**真的有**这样的 case（否则 OOD 是空的，白跑一批）。
    """
    cases = sp.build_cases(sp.SearchConfig(ood_only=True))
    assert cases
    for case in cases:
        assert case.unseen_dims()
    dims = Counter(name for case in cases for name in case.unseen_dims())
    assert set(dims) == {"topic", "best_flavor", "mix", "pair_kind"}


def test_id_splits_never_see_the_held_out_values() -> None:
    """输入：同分布三个切分。输出：无。

    预期行为：训练侧一个留出取值都不出现——这是「值留出」的定义。
    """
    workflow = wf.SearchWorkflow()
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            assert not case.unseen_dims(), (split, case.describe())


def test_best_position_is_the_ranking_winner_on_every_id_case() -> None:
    """输入：同分布的全部 case。输出：无。

    预期行为：按题面写的排序规则算出来的最优，永远落在 case 指定的那个位置上——
    构造与判据走偏的话，`best_result` 的答案就会变成「另一个位置」。
    """
    workflow = wf.SearchWorkflow()
    for locale in locales.available():
        for split in ("train", "val", "test_iid"):
            for case in workflow.build_cases(workflow.config_for(split)):
                spec = sp.sample_spec(random.Random(0), case, split, 0, 0)
                rows = truth_rows(spec, locale)
                assert sp.best_of(rows).index == case.best_at, (locale, case.describe())


def test_same_authority_competitors_are_older_than_the_best() -> None:
    """输入：同分布的全部 case。输出：无。

    预期行为：与最佳结果**同相关度、同权威**的竞争者，更新日期一定更早——
    排序规则的第三档（日期）因此是真在起作用，而不是摆设。
    """
    workflow = wf.SearchWorkflow()
    for case in workflow.build_cases(workflow.config_for("train")):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        rows = truth_rows(spec)
        top = sp.best_of(rows)
        for row in rows:
            same_class = (
                sp.relevance_of(row) == sp.relevance_of(top) and row.authority == top.authority
            )
            if same_class and row.index != top.index:
                assert row.updated < top.updated, case.describe()


def test_lower_authority_competitor_can_be_newer() -> None:
    """输入：含 `old_hit` 的配方。输出：无。

    预期行为：权威低一档的竞争者**可以**比最佳结果新——排序规则的第二档（权威）
    因此也是真在起作用，光看日期猜不出答案。
    """
    case = sp.SearchCase("refund", "full", sp.MIXES[0], (1, 2), 3, "none", 1, 0)
    spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
    rows = {row.index: row for row in truth_rows(spec)}
    top = sp.best_of(tuple(rows.values()))
    newer = [row for row in rows.values() if row.updated > top.updated]
    assert newer, "这条配方里没有比最佳结果更新的干扰项，这条测试就没意义了"
    assert all(row.authority != top.authority for row in newer)


def test_every_flavor_matches_its_recipe() -> None:
    """输入：全取值表的 case。输出：无。

    预期行为：渲染出来的每一条，重算出来的相关度与要点覆盖都等于角色表里的主张——
    模板改歪了会在这里当场炸，不会静默产出一条错数据。
    """
    loc = locales.load("zh")
    workflow = wf.SearchWorkflow()
    for case in workflow.build_cases(workflow.config_for("train")):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        points = tuple(loc["topic"][spec.topic.name]["points"])
        by_index = {row.index: row for row in truth_rows(spec)}
        for result in spec.results:
            if result.flavor not in sp.FLAVOR_BY_NAME:  # 被点名的那两条：角色名是 `pair_<关系>`
                continue
            flavor = sp.FLAVOR_BY_NAME[result.flavor]
            row = by_index[result.index]
            assert sp.relevance_of(row) == flavor.relevance, (case.describe(), row.index)
            assert sp.covers_of(row, points) == flavor.covers, (case.describe(), row.index)


def test_shift_months_clamps_to_the_month_end() -> None:
    """输入：月末日期。输出：偏移后的日期。

    预期行为：日号夹到该月最后一天（1 月 31 日推一个月是 2 月 29 日），
    不借 `timedelta` 硬加天数——那样推出来的月份会随大小月漂。
    """
    assert sp.shift_months(dt.date(2024, 1, 31), 1) == dt.date(2024, 2, 29)
    assert sp.shift_months(dt.date(2024, 3, 31), -1) == dt.date(2024, 2, 29)
    assert sp.shift_months(dt.date(2024, 12, 15), 1) == dt.date(2025, 1, 15)
    assert sp.shift_months(dt.date(2024, 1, 15), -1) == dt.date(2023, 12, 15)


def test_day_jitter_is_stable_per_case() -> None:
    """输入：同一个 case。输出：两次算出来的抖动。预期行为：相等（不随进程变）。"""
    case = sp.SearchCase("refund", "full", sp.MIXES[1], (1, 2), 3, "conflict", 1, 0)
    assert sp._day_jitter(case) == sp._day_jitter(case)
    other = sp.SearchCase("refund", "full", sp.MIXES[1], (1, 2), 3, "conflict", 2, 0)
    assert 0 <= sp._day_jitter(other) < 25


def test_best_authority_only_covers_the_declared_flavors() -> None:
    """输入：不能当最佳结果的角色。输出：异常。预期行为：报错点名可用取值。"""
    with pytest.raises(KeyError, match="other_topic"):
        sp._best_authority("other_topic")


def test_config_reports_bad_knobs_by_name() -> None:
    """输入：各种坏旋钮。输出：异常。预期行为：报错点名是哪个字段、可用取值是什么。"""
    with pytest.raises(ValueError, match="topics 不能是空表"):
        sp.SearchConfig(topics=())
    with pytest.raises(ValueError, match="不存在的主题"):
        sp.SearchConfig(topics=("nope",))
    with pytest.raises(ValueError, match="不能当最佳结果"):
        sp.SearchConfig(best_flavors=("other_topic",))
    with pytest.raises(ValueError, match="不存在的关系"):
        sp.SearchConfig(pair_kinds=("nope",))
    with pytest.raises(ValueError, match="不存在的角色"):
        sp.SearchConfig(mixes=(("nope",),))


def test_build_cases_drops_layouts_that_do_not_fit() -> None:
    """输入：一个位置超出页长的版式。输出：无。

    预期行为：直接丢掉这个组合，而不是造出一条引用不存在编号的 case。
    """
    config = sp.SearchConfig(
        topics=("refund",),
        best_flavors=("full",),
        mixes=(sp.MIXES[1],),
        layouts=(((1, 2), 9),),
        pair_kinds=("none",),
        query_styles=(0,),
    )
    assert sp.build_cases(config) == []


def test_spec_to_dict_is_json_serialisable() -> None:
    """输入：一条规格。输出：摘要字典。预期行为：字段齐全、可 JSON 序列化。"""
    import json

    spec = sp.sample_spec(random.Random(0), sp.SearchCase("refund", "full", sp.MIXES[1], (1, 2), 3, "conflict", 1, 0), "train", 0, 0)
    payload = sp.spec_to_dict(spec)
    json.dumps(payload, ensure_ascii=False)
    assert payload["best_at"] == 3
    assert len(payload["results"]) == spec.case.count
