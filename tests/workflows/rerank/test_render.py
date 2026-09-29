"""文本外壳测试：提示词、草稿校验、state 往返、语言识别。"""

from __future__ import annotations

import pytest

from workflows.rerank import features, locales, render
from tests.workflows.rerank.conftest import FakeBackend, draft_data
from jc.workflow import DraftError


def _plan(terms: int = 3, relevant: int = 3, best: str = "first") -> features.RerankPlan:
    """输入：三个特征取值。输出：名次计划。预期行为：测试里随手造计划的便捷入口。"""
    return features.plan_for(features.RerankCase(terms, relevant, best), 0)


def test_cand_ref_is_zero_based() -> None:
    """输入：0 起的下标。输出：无。预期行为：0 → 候选 1（差一格是最容易犯的错）。"""
    loc = locales.load("zh")
    assert render.cand_ref(loc, 0) == "候选 1"
    assert render.cand_ref(loc, 4) == "候选 5"


def test_prompt_spells_out_every_candidate_requirement() -> None:
    """输入：一份含满覆盖 / 部分覆盖 / 干净三种候选的计划。输出：无。

    预期行为：提示词里逐条写明第几条该沾几个词——这是「受控合成」的规格，不能省。
    """
    plan = _plan(terms=3, relevant=3, best="first")
    prompt = render.build_prompt(plan, locales.load("zh"))
    assert "3 个有区分度的关键词" in prompt
    assert f"第 {plan.best_pos + 1} 条：必须把 3 个关键词全部写进去" in prompt
    clean = [index + 1 for index, value in enumerate(plan.overlaps) if value == 0]
    for row in clean:
        assert f"第 {row} 条：一个关键词都不许出现" in prompt
    partials = [index + 1 for index, value in enumerate(plan.overlaps) if 0 < value < 3]
    for rank, row in enumerate(partials):
        # 指名道姓：部分覆盖的那几条必须点出「terms 里第几个」，不能只说「其中 1 个」。
        assert f"第 {row} 条：只包含关键词 {rank % 3 + 1}（terms 里第 {rank % 3 + 1} 个）" in prompt


def test_check_parts_accepts_a_faithful_draft() -> None:
    """输入：一份完全兑现计划的草稿。输出：无。预期行为：不抛异常。"""
    plan = _plan()
    data = draft_data(plan)
    loc = locales.load("zh")
    render.check_parts(
        str(data["query"]), list(data["terms"]), list(data["candidates"]), plan, loc
    )


def test_prompt_restates_the_coverage_vector_as_a_self_check() -> None:
    """输入：一份计划。输出：无。

    预期行为：提示词末尾把覆盖率清单原样再摆一遍（真机第一批五条拒稿是同一个理由，这条
    自查就是照那个改的）；清单本身是机械算的，不靠人写。
    """
    plan = _plan(terms=3, relevant=3, best="first")
    loc = locales.load("zh")
    vector = render.coverage_vector(plan, loc)
    assert vector == "、".join(str(value) for value in plan.overlaps)
    assert vector in render.build_prompt(plan, loc)


@pytest.mark.parametrize(
    ("switch", "match"),
    [
        ("unrealized", "该沾上"),
        ("extra_hit", "该沾上"),
        ("duplicate", "候选重复"),
        ("drop_candidate", "候选该有 5 条"),
        ("bad_terms", "关键词太短"),
        ("term_missing_from_query", "查询里没出现关键词"),
        ("fence_line", "带围栏"),
    ],
)
def test_check_parts_rejects_broken_drafts(switch: str, match: str) -> None:
    """输入：每种「故意写歪」。输出：无。预期行为：当场抛 `ValueError`，理由对得上。"""
    plan = _plan(terms=3, relevant=2, best="first")
    data = draft_data(plan, **{switch: True})
    loc = locales.load("zh")
    with pytest.raises(ValueError, match=match):
        render.check_parts(
            str(data["query"]), list(data["terms"]), list(data["candidates"]), plan, loc
        )


def test_broken_drafts_are_marked_retryable() -> None:
    """输入：一条「候选重复」的草稿。输出：无。

    预期行为：抛的是 `jc.workflow.DraftError`（带 `retryable_for_case`），管线据此重渲染一次。
    真机上这个 workflow 的拒稿**全是**「渲染期写歪」（PLAN 10.80），单发等于白丢一半样本。
    """
    plan = _plan(terms=3, relevant=2, best="first")
    data = draft_data(plan, duplicate=True)
    loc = locales.load("zh")
    with pytest.raises(DraftError) as caught:
        render.check_parts(
            str(data["query"]), list(data["terms"]), list(data["candidates"]), plan, loc
        )
    assert getattr(caught.value, "retryable_for_case", False) is True


def test_check_draft_wraps_check_parts() -> None:
    """输入：草稿模型。输出：无。预期行为：走同一份判据（不另写一套）。"""
    plan = _plan(terms=2, relevant=1, best="last")
    good = draft_data(plan)
    loc = locales.load("zh")
    render.check_draft(
        render.RerankDraft(
            query=str(good["query"]),
            terms=[str(item) for item in good["terms"]],
            candidates=[str(item) for item in good["candidates"]],
        ),
        plan,
        loc,
    )
    with pytest.raises(ValueError, match="候选重复"):
        bad = draft_data(plan, duplicate=True)
        render.check_draft(
            render.RerankDraft(
                query=str(bad["query"]),
                terms=[str(item) for item in bad["terms"]],
                candidates=[str(item) for item in bad["candidates"]],
            ),
            plan,
            loc,
        )


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_state_round_trips_through_parse(lang: str) -> None:
    """输入：两种语言各渲染一次。输出：无。

    预期行为：查询、关键词、候选逐条原样反解回来；排序口径那两行也认得出。
    """
    plan = _plan(terms=3, relevant=2, best="middle")
    data = draft_data(plan)
    draft = render.RerankDraft(
        query=str(data["query"]),
        terms=[str(item) for item in data["terms"]],
        candidates=[str(item) for item in data["candidates"]],
    )
    loc = locales.load(lang)
    state = render.to_state(draft, plan, loc)
    parsed = render.parse_with(state, loc)
    assert parsed.query == draft.query.strip()
    assert parsed.terms == tuple(draft.terms)
    assert parsed.candidates == tuple(draft.candidates)
    assert parsed.rule == render.rule_rows(loc)
    assert render.detect_locale(state) == lang


def test_detect_locale_does_not_let_one_language_steal_the_other() -> None:
    """输入：中英各一段 state。输出：无。

    预期行为：各认各的——编号行与语言无关，靠它认语言会让先试的那一种抢走全部 state
    （这条坑在 lint 上踩过）。
    """
    plan = _plan(terms=2, relevant=1, best="first")
    data = draft_data(plan)
    draft = render.RerankDraft(
        query=str(data["query"]),
        terms=[str(item) for item in data["terms"]],
        candidates=[str(item) for item in data["candidates"]],
    )
    zh_state = render.to_state(draft, plan, locales.load("zh"))
    en_state = render.to_state(draft, plan, locales.load("en"))
    assert render.detect_locale(zh_state) == "zh"
    assert render.detect_locale(en_state) == "en"
    assert render.detect_locale("随便一段没有抬头的文本") is None
    assert render.detect_locale(123) is None
    empty = render.parse_state("随便一段没有抬头的文本")
    assert empty.query is None and empty.candidates == ()


def test_unknown_locale_reports_available() -> None:
    """输入：不存在的语言。输出：无。预期行为：抛 `FileNotFoundError` 并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="可用"):
        locales.load("nope")


def test_backend_fake_produces_parseable_drafts() -> None:
    """输入：假后端。输出：无。预期行为：它造的草稿真的能过校验（假后端自己得先站得住）。"""
    plan = _plan(terms=2, relevant=2, best="last")
    backend = FakeBackend()
    from workflows.rerank.spec import RerankSpec

    spec = RerankSpec(case=features.RerankCase(2, 2, "last"), phrasing=0, split="train", seq=0, plan=plan)
    task = render.build_task(spec, locales.load("zh"))
    draft = backend.generate(task)
    parsed = render.RerankDraft(
        query=str(draft.data["query"]),
        terms=[str(item) for item in draft.data["terms"]],
        candidates=[str(item) for item in draft.data["candidates"]],
    )
    render.check_draft(parsed, plan, locales.load("zh"))
