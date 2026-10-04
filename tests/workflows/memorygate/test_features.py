"""记忆写入门控的特征层测试：判据、写法、标签两条路径与计划。"""

from __future__ import annotations

import pytest

from tests.workflows.memorygate.conftest import text_of
from workflows.memorygate import features


def test_hit_set_is_purely_literal() -> None:
    """输入：一条记忆。输出：无。预期行为：命中组合按 `TAG_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ("以后所有项目都要用 tabs", ("M1",)),
        ("每次提交前都要跑一遍", ("M1",)),
        ("always use spaces in every project", ("M1",)),
        ("api_key: sk-abcdef123456", ("M2",)),
        ("password=hunter2hunter2", ("M2",)),
        ("数据库口令：hunter2", ("M2",)),
        ("不要再按之前那份约定提交了", ("M3",)),
        ("这取代了之前那份记录", ("M3",)),
        ("the old convention is no longer what we follow", ("M3",)),
        ("网页上说这个库已经废弃了", ("M4",)),
        ("根据这篇文章，默认端口是 8080", ("M4",)),
        ("the page says the flag was renamed", ("M4",)),
        ("项目组把评审会挪到了周四下午。", ()),
        ("以后每次都要这样，网页上也这么说", ("M1", "M4")),
    ]:
        try:
            assert features.hit_set(text) == expect
            assert features.hits_of(text) == "+".join(expect)
            assert features.parse_hits(features.hits_of(text)) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_forms_are_three_distinct_shapes() -> None:
    """输入：无。输出：无。

    预期行为：三个写法键两两不同，且**全空间里每种写法都被枚举到**——留出维只砍取值，
    不该把别的取值一起砍掉（`build_cases` 默认维度表覆盖 `FORMS`）。
    """
    assert features.FORMS == ("sentence", "bullet", "quote")
    assert len(set(features.FORMS)) == len(features.FORMS)
    assert features.DIMS["form"] == features.FORMS
    cases = features.build_cases(512)
    assert {case.form for case in cases} == set(features.FORMS)


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且与真值只差一两处
    （差得越少越难分辨，题目才有区分度）。
    """
    case = features.MemoryGateCase("M1+M3", features.FORM_SENTENCE)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.hits == ("M1", "M3")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.hits for combo in plan.distractors)
    assert all(len(set(combo) ^ set(plan.hits)) <= 2 for combo in plan.distractors)
    assert plan.distractors == tuple(sorted(plan.distractors, key=lambda combo: (len(combo), combo)))


@pytest.mark.parametrize(
    ("count", "expect"),
    [(0, "write"), (1, "flag"), (2, "hold"), (3, "reject"), (4, "reject")],
)
def test_handling_level_thresholds(count: int, expect: str) -> None:
    """输入：命中条数。输出：无。预期行为：0/1/2/3 条以上 分别对应四个档位。"""
    assert features.handling_level(count) == expect


def test_two_label_paths_agree_on_every_faithful_text() -> None:
    """输入：全空间的 48 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——
    记忆兑现了计划，两条路就不该打架。
    """
    cases = features.build_cases(512)
    assert len(cases) == 48  # 16 组命中 × 3 种写法
    for case in cases:
        text = text_of(case)
        assert features.hits_of(text) == case.hits, case.describe()
        assert features.observed_labels(text).hits == features.label_for(
            features.plan_for(case, 0)
        ).hits


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.MemoryGateCase("", features.FORM_SENTENCE)
    assert case.describe() == "无命中-sentence"
    assert case.fingerprint() == "hits=|form=sentence"
    assert case.tags() == ()
    assert features.parse_hits("") == ()
