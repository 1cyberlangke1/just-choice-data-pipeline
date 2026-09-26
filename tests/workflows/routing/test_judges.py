"""判别器：文本反解 → 独立重算 → 逐题比对，以及各种坏输入的报错。"""

from __future__ import annotations

import random

import pytest

from jc import assemble
from tests.workflows.routing.conftest import _make_spec
from workflows.routing import judges, locales, parse, questions, render, spec as sp

LANGS = ("zh", "en")


def _sample(lang: str, **overrides: object) -> tuple[dict, sp.TicketSpec, dict]:
    """输入：语言与 case 覆盖项。输出：(样本, 真值规格, 语言字典)。

    预期行为：走**生产同一条路**——`questions.build` 出题、`assemble` 组装，
    这样判别器面对的就是真实产物形态，而不是手搓的近似品。
    """
    loc = locales.load(lang)
    spec = _make_spec(**overrides)
    state = render.build(spec, loc)
    labeled = questions.build(spec, loc, 0, random.Random(1))
    sample = assemble.assemble_sample("routing-train-0000001", state, labeled, random.Random(2))
    return sample, spec, loc


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"issue": "feature", "compliance": "gdpr"},
        {"tier": "free", "wait": "short", "impact": "single_user", "urgency": "can_wait"},
        {"issue": "data_loss", "compliance": "security", "wait": "medium"},
    ],
)
def test_recompute_matches_the_spec(lang: str, overrides: dict) -> None:
    """输入：几组 case × 两种语言。输出：无。

    预期行为：从**文本**重算出来的八族答案与规格逐项一致——问题类型只在正文里，
    所以这条同时验了「正文能读回问题类型」。
    """
    sample, spec, loc = _sample(lang, **overrides)
    ticket = parse.parse_ticket(sample["state"], parse.label_index(loc))
    got = judges.recompute(ticket, loc)
    assert got == {
        "severity": spec.severity,
        "queue": spec.queue,
        "escalation": spec.escalation,
        "critical_area": spec.critical_area,
        "needs_human": spec.needs_human,
        "sla_breach": spec.sla_breach,
        "data_risk": spec.data_risk,
        "priority": spec.priority,
    }


def test_recompute_covers_exactly_the_declared_families() -> None:
    """输入：无。输出：无。

    预期行为：重算结果的键集合等于 `questions.FAMILIES`——少一个键，那一族就会被
    判别器判成「未知题族」，整批数据全灭。
    """
    sample, _spec, loc = _sample("zh")
    ticket = parse.parse_ticket(sample["state"], parse.label_index(loc))
    assert set(judges.recompute(ticket, loc)) == set(questions.FAMILIES)


def test_issue_of_needs_exactly_one_hit() -> None:
    """输入：认不出问题类型的正文 / 两句都出现的正文。输出：无。

    预期行为：两种情况都报错——「认不出」和「认出两句」都不是「数据难判」，
    而是渲染与语言文件对不上的硬伤。
    """
    loc = locales.load("zh")
    with pytest.raises(ValueError, match="问题类型"):
        judges.issue_of("这里什么都没有", loc["area"]["payments"], loc)
    both = (
        loc["body"]["issue"]["bug"].format(area=loc["area"]["payments"])
        + loc["body"]["issue"]["howto"].format(area=loc["area"]["payments"])
    )
    with pytest.raises(ValueError, match="问题类型"):
        judges.issue_of(both, loc["area"]["payments"], loc)


def test_elapsed_of_rejects_time_travel() -> None:
    """输入：当前时间早于创建时间。输出：无。预期行为：报错而不是给出负数。"""
    assert judges.elapsed_of("08:00", "11:20") == 200
    with pytest.raises(ValueError, match="早于"):
        judges.elapsed_of("11:20", "08:00")


def test_reverse_rejects_duplicate_text() -> None:
    """输入：两条文案相同的词表。输出：无。预期行为：报错（反解会有歧义）。"""
    with pytest.raises(ValueError, match="文案重复"):
        judges.reverse({"a": "同一个词", "b": "同一个词"})


@pytest.mark.parametrize("lang", LANGS)
def test_both_judges_pass_a_clean_sample(lang: str) -> None:
    """输入：一条干净样本。输出：无。预期行为：两个判别器都通过。"""
    sample, spec, _loc = _sample(lang, compliance="gdpr", tier="enterprise")
    for judge in (judges.RoundtripJudge(locale=lang), judges.LabelJudge(locale=lang)):
        verdict = judge.judge(sample, spec)
        assert verdict.ok, f"{judge.name}: {verdict.reason}"


def test_roundtrip_judge_skips_without_spec() -> None:
    """输入：不带真值规格的样本。输出：无。

    预期行为：跳过而不是判错——判别器在「拿不到真值」时该闭嘴，不能把不确定说成不合格。
    """
    sample, _spec, _loc = _sample("zh")
    verdict = judges.RoundtripJudge().judge(sample, None)
    assert verdict.ok and verdict.raw.get("skipped") is True


@pytest.mark.parametrize(
    ("old", "new"),
    [("影响面：单个用户", "影响面：全部用户"), ("紧急度：可以等", "紧急度：业务受阻")],
)
def test_roundtrip_judge_catches_tampered_state(old: str, new: str) -> None:
    """输入：被改过一格的工单文本。输出：无。

    预期行为：结构判别器判不通过并点出是哪一格——它是「渲染有没有走样」的看门人。
    """
    spec = _make_spec(impact="single_user", urgency="can_wait")
    loc = locales.load("zh")
    state = render.build(spec, loc).replace(old, new)
    sample = {"state": state, "questions": {}, "answers": {}}
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok
    assert verdict.raw["diff"]


def test_label_judge_catches_a_wrong_choice_index() -> None:
    """输入：把某道 choice 的答案编号挪一格。输出：无。

    预期行为：判不通过，且理由里出现题号——这条是「标签有没有错位」的看门人。
    """
    sample, _spec, _loc = _sample("zh")
    qid = "queue"
    options = list(sample["questions"][qid]["criteria"])
    sample["answers"][qid]["choice"] = (sample["answers"][qid]["choice"] + 1) % len(options)
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    assert any(problem.startswith(f"{qid}:") for problem in verdict.raw["problems"])


def test_label_judge_catches_a_flipped_noul_and_a_wrong_score() -> None:
    """输入：翻转一个 noul、改一个 score。输出：无。预期行为：两处都被抓到。"""
    sample, _spec, _loc = _sample("zh")
    sample["answers"]["data_risk"]["noul"] = not sample["answers"]["data_risk"]["noul"]
    sample["answers"]["priority"]["score"] = (sample["answers"]["priority"]["score"] + 1) % 5
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    problems = verdict.raw["problems"]
    assert any(problem.startswith("data_risk:") for problem in problems)
    assert any(problem.startswith("priority:") for problem in problems)


def test_label_judge_flags_an_unknown_family() -> None:
    """输入：塞一道题号不在题族清单里的题。输出：无。

    预期行为：判成「未知题族」而不是静默放过——静默放过会让整批数据里混进没人验过的题。
    """
    sample, _spec, _loc = _sample("zh")
    sample["questions"]["mystery"] = {"type": "noul", "instructions": "?"}
    sample["answers"]["mystery"] = {"type": "noul", "noul": True}
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    assert any("未知题族" in problem for problem in verdict.raw["problems"])


def test_label_judge_flags_an_out_of_range_index() -> None:
    """输入：把 choice 的编号改成越界值。输出：无。预期行为：报「编号越界」。"""
    sample, _spec, _loc = _sample("zh")
    sample["answers"]["severity"]["choice"] = 99
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    assert any("越界" in problem for problem in verdict.raw["problems"])

