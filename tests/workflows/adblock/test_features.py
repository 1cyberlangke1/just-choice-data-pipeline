"""特征层测试：判据只认字面、命中组合按构造均衡、两条标签路径互不打架。"""

from __future__ import annotations

import pytest

from tests.workflows.adblock.conftest import text_of
from workflows.adblock import features


def test_hit_set_is_purely_literal() -> None:
    """输入：一段 HTML 片段。输出：无。预期行为：命中组合按 `TAG_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ('<div class="ad-slot">正文</div>', ("A1",)),
        ('<div id="adsbygoogle">正文</div>', ("A1",)),
        ('<div class="sponsored-post">正文</div>', ("A1",)),
        ('<span class="note">本栏目为赞助内容</span>', ("A2",)),
        ('<span class="note">Sponsored by Acme</span>', ("A2",)),
        ('<span class="note">Advertisement</span>', ("A2",)),
        ('<img src="https://googlesyndication.com/x.gif" alt="">', ("A3",)),
        ('<div data-ad-slot="123"></div>', ("A3",)),
        ('<img src="https://doubleclick.net/pixel.gif" alt="">', ("A3",)),
        ('<div style="display:none"></div>', ("A4",)),
        ('<div aria-hidden="true"></div>', ("A4",)),
        ('<img src="x.gif" width="1" height="1" alt="">', ("A4",)),
        # 这几条是判据「只认字面、且各占一处」的证据：正常元素一条都不该踩。
        ('<div class="header">导航</div>', ()),
        ('<div class="download-btn">下载</div>', ()),
        ('<div class="read-more">阅读全文</div>', ()),
        ('<img src="https://example.com/photo.jpg" alt="">', ()),
        ('<div style="width:100px"></div>', ()),
    ]:
        try:
            assert features.hit_set(text) == expect
            assert features.hits_of(text) == "+".join(expect)
            assert features.parse_hits(features.hits_of(text)) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_surfaces_are_three_distinct_shapes() -> None:
    """输入：无。输出：无。

    预期行为：三个页面类型键两两不同，且**全空间里每种页面都被枚举到**——留出维只砍取值，
    不该把别的取值一起砍掉（`build_cases` 默认维度表覆盖 `SURFACES`）。
    """
    assert features.SURFACES == ("article", "feed", "video")
    assert len(set(features.SURFACES)) == len(features.SURFACES)
    assert features.DIMS["surface"] == features.SURFACES
    cases = features.build_cases(512)
    assert {case.surface for case in cases} == set(features.SURFACES)


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且与真值只差一两处
    （差得越少越难分辨，题目才有区分度）。
    """
    case = features.AdBlockCase("A1+A3", features.SURFACE_ARTICLE)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.hits == ("A1", "A3")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.hits for combo in plan.distractors)
    assert all(len(set(combo) ^ set(plan.hits)) <= 2 for combo in plan.distractors)
    assert plan.distractors == tuple(
        sorted(plan.distractors, key=lambda combo: (len(combo), combo))
    )


@pytest.mark.parametrize(
    ("count", "expect"),
    [(0, "keep"), (1, "collapse"), (2, "hide"), (3, "remove"), (4, "remove")],
)
def test_act_level_thresholds(count: int, expect: str) -> None:
    """输入：命中条数。输出：无。预期行为：0/1/2/3 条以上 分别对应四个档位。"""
    assert features.act_level(count) == expect


def test_two_label_paths_agree_on_every_faithful_snippet() -> None:
    """输入：全空间的 48 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——
    片段兑现了计划，两条路就不该打架。
    """
    cases = features.build_cases(512)
    assert len(cases) == 48  # 16 组命中 × 3 种页面
    for case in cases:
        snippet = text_of(case.tags())
        assert features.hits_of(snippet) == case.hits, case.describe()
        assert features.observed_labels(snippet).hits == features.label_for(
            features.plan_for(case, 0)
        ).hits


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.AdBlockCase("", features.SURFACE_ARTICLE)
    assert case.describe() == "无命中-article"
    assert case.fingerprint() == "hits=|surface=article"
    assert case.tags() == ()
    assert features.parse_hits("") == ()
