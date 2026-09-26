"""真值规格：规则引擎、组合枚举、采样不变量、值留出。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from tests.workflows.citation.conftest import tiny_config
from workflows.citation import spec as sp


def _spec(case: sp.CitationCase, seed: int = 3) -> sp.CitationSpec:
    """输入：case、种子。输出：真值规格。预期行为：同 case 必得同一份规格。"""
    return sp.sample_spec(random.Random(seed), case, "train", 0, 0)


def _case(**overrides: Any) -> sp.CitationCase:
    """输入：维度覆盖项。输出：一个 case（默认「鉴权 / 通过 / 照抄 / 三节 / 中间」）。"""
    fields: dict[str, Any] = {
        "topic": "auth",
        "status": "verified",
        "quote_form": "verbatim",
        "sections": 3,
        "noise": "plain",
        "claim_style": 1,
        "where": "middle",
        "doc_style": "bracketed",
    }
    fields.update(overrides)
    return sp.CitationCase(**fields)


# ---------- 规则引擎：四档结论、出处、处置、强度 ----------


def test_verdict_table_matches_the_question_text() -> None:
    """输入：三条事实的每一种组合。输出：无。

    预期行为：四档结论就是题面里写的那张表——引文不在文档里 = 编造；在文档里但不在声称的
    小节里 = 证据不足；在声称的小节里按该小节的态度分三档（支持 = 通过、反驳 = 反驳、
    没提 = 证据不足）。
    """
    assert sp.verdict_of(False, False, "supports") == "fabricated"
    assert sp.verdict_of(False, False, "says_nothing") == "fabricated"
    assert sp.verdict_of(True, False, "supports") == "unsupported"
    assert sp.verdict_of(True, False, "contradicts") == "unsupported"
    assert sp.verdict_of(True, True, "supports") == "verified"
    assert sp.verdict_of(True, True, "contradicts") == "contradicted"
    assert sp.verdict_of(True, True, "says_nothing") == "unsupported"


def test_source_of_splits_the_three_places() -> None:
    """输入：前两条事实。输出：无。预期行为：不在文档 / 声称的小节 / 别的小节。"""
    assert sp.source_of(False, False) == "not_in_doc"
    assert sp.source_of(True, True) == "cited_section"
    assert sp.source_of(True, False) == "other_section"


def test_action_sends_only_unsupported_to_a_human() -> None:
    """输入：四档结论。输出：无。

    预期行为：只有「证据不足」要人看；「编造」由字符串匹配直接定案，也不需要人。
    """
    assert sp.action_of("unsupported") == "human_review"
    for verdict in ("verified", "contradicted", "fabricated"):
        assert sp.action_of(verdict) == "auto_accept"


def test_strength_scale_is_monotone_in_evidence() -> None:
    """输入：前三条事实。输出：无。

    预期行为：不在文档里 0；在但不在声称小节 1；在声称小节但没提 2；有明确态度 3。
    """
    assert sp.strength_of(False, False, "says_nothing") == 0
    assert sp.strength_of(True, False, "supports") == 1
    assert sp.strength_of(True, True, "says_nothing") == 2
    assert sp.strength_of(True, True, "supports") == 3
    assert sp.strength_of(True, True, "contradicts") == 3


def test_answers_of_covers_the_seven_families_and_agrees_with_the_tables() -> None:
    """输入：三条事实的每一种组合。输出：无。

    预期行为：七个题族一次算全，且每一项都等于对应那张表的结果——任何一侧另起一套判断
    都会在这里露出来。
    """
    families = {
        "found",
        "located",
        "relation",
        "quote_source",
        "verdict",
        "action",
        "evidence_strength",
    }
    seen: set[str] = set()
    for found in (False, True):
        for located in (False, True):
            for relation in ("supports", "contradicts", "says_nothing"):
                answers = sp.answers_of(found, located, relation)
                assert set(answers) == families
                assert answers["found"] is found and answers["located"] is located
                assert answers["relation"] == relation
                assert answers["quote_source"] == sp.source_of(found, located)
                assert answers["verdict"] == sp.verdict_of(found, located, relation)
                assert answers["action"] == sp.action_of(answers["verdict"])
                assert answers["evidence_strength"] == sp.strength_of(found, located, relation)
                seen.add(answers["verdict"])
    assert seen == {"verified", "contradicted", "unsupported", "fabricated"}


# ---------- 规则引擎：一句话与一个小节的关系 ----------


def test_fact_opposite_and_key() -> None:
    """输入：一句话。输出：无。预期行为：反面只翻极性；键含主题、句号与极性。"""
    fact = sp.Fact("auth", "mfa_admin")
    assert fact.opposite == sp.Fact("auth", "mfa_admin", True)
    assert fact.opposite.opposite == fact
    assert fact.key() == ("auth", "mfa_admin", False)
    assert fact.opposite.key() != fact.key()


def test_relation_of_reads_only_the_given_section() -> None:
    """输入：主张与一个小节的句子。输出：无。

    预期行为：同句同极性 = 支持；同句反极性 = 反驳；其余 = 没提——只看这个小节，
    别的小节说了什么不算。
    """
    claim = sp.Fact("auth", "token_rotation")
    other = sp.Fact("auth", "session_scope")
    assert sp.relation_of(claim, (other, claim)) == "supports"
    assert sp.relation_of(claim, (other, claim.opposite)) == "contradicts"
    assert sp.relation_of(claim, (other,)) == "says_nothing"
    assert sp.relation_of(claim, (sp.Fact("billing", "token_rotation"),)) == "says_nothing"


def test_section_of_fact_finds_the_unique_home() -> None:
    """输入：文档、一句话。输出：无。预期行为：命中的小节编号；不在文档里给 None。"""
    spec = _spec(_case())
    home = spec.quote_home
    assert home is not None
    assert sp.section_of_fact(spec.sections, spec.quote_source) == home
    assert sp.section_of_fact(spec.sections, None) is None
    stranger = sp.Fact(spec.claim.topic, spec.claim.sentence, not spec.claim.negated)
    if not any(item.key() == stranger.key() for section in spec.sections for item in section.facts):
        assert sp.section_of_fact(spec.sections, stranger) is None


# ---------- 组合枚举 ----------


def test_where_fits_rejects_middle_on_two_sections() -> None:
    """输入：位置档与小节数。输出：无。预期行为：两节没有「中间」，三节才有。"""
    assert sp.where_fits("first", 2) and sp.where_fits("last", 2)
    assert not sp.where_fits("middle", 2)
    assert sp.where_fits("middle", 3)


def test_quote_form_fits_keeps_only_fabricated_non_verbatim() -> None:
    """输入：结论档与引文写法。输出：无。

    预期行为：只有「编造」允许引文不是原文；其余档引文必须照抄，否则结论与字符串匹配
    自相矛盾。
    """
    assert sp.quote_form_fits("fabricated", "truncated")
    assert sp.quote_form_fits("fabricated", "reworded")
    assert not sp.quote_form_fits("fabricated", "verbatim")
    for status in ("verified", "contradicted", "unsupported", "mislocated"):
        assert sp.quote_form_fits(status, "verbatim")
        assert not sp.quote_form_fits(status, "truncated")


def test_build_cases_is_deterministic_and_unique() -> None:
    """输入：同一份采样参数。输出：两次枚举的 case 列表。预期行为：逐项一致且互不重复。"""
    config = tiny_config()
    first = sp.build_cases(config)
    second = sp.build_cases(config)
    assert first == second
    assert len({case.as_dict().items().__str__() for case in first}) == len(first)
    assert len(first) > 100


def test_build_cases_drops_the_impossible_cells() -> None:
    """输入：全开的采样参数。输出：case 列表。预期行为：两类无解的组合一格都不出现。"""
    cases = sp.build_cases(sp.CitationConfig())
    for case in cases:
        if case.sections == 2:
            assert case.where != "middle", case.describe()
            if case.noise in ("decoy", "off_topic"):
                assert case.status != "mislocated", case.describe()


def test_ood_only_keeps_cases_with_an_unseen_value() -> None:
    """输入：`ood_only` 的采样参数。输出：case 列表。预期行为：每个 case 至少含一个留出值。"""
    cases = sp.build_cases(sp.CitationConfig(split="test_ood", ood_only=True))
    assert cases
    assert all(case.unseen_dims() for case in cases)
    dims = {name for case in cases for name in case.unseen_dims()}
    assert dims == {"topic", "status", "quote_form", "noise", "claim_style"}


def test_unseen_dims_names_the_left_out_values() -> None:
    """输入：含留出取值的 case。输出：无。预期行为：点名是哪个维度留出了。"""
    assert sp.CitationCase("rollout", "verified", "verbatim", 3, "plain", 1, "first", "bracketed").unseen_dims() == ("topic",)
    case = sp.CitationCase("auth", "mislocated", "verbatim", 3, "plain", 1, "first", "bracketed")
    assert case.unseen_dims() == ("status",)
    seen = sp.CitationCase("auth", "verified", "verbatim", 3, "plain", 1, "first", "bracketed")
    assert seen.unseen_dims() == ()


# ---------- 采样不变量 ----------


def test_case_rng_is_derived_from_the_case_only() -> None:
    """输入：同一个 case 采两次。输出：两份规格。预期行为：逐字段一致（不掷骰子）。"""
    case = _case(status="contradicted", noise="decoy")
    first = _spec(case)
    second = _spec(case)
    assert first.sections == second.sections
    assert first.claim == second.claim and first.claimed == second.claimed
    other = _spec(_case(status="contradicted", noise="decoy", topic="billing"))
    assert other.claim.topic == "billing"


@pytest.mark.parametrize("status", sp.STATUSES)
def test_status_matches_the_three_facts(status: str) -> None:
    """输入：每个结论档。输出：无。

    预期行为：档名与「引文在不在文档里 / 在不在声称的小节里 / 该小节的态度」对得上——
    这一条是判别器能独立重算的前提。
    """
    form = "verbatim" if status != "fabricated" else "reworded"
    spec = _spec(_case(status=status, quote_form=form))
    found, located, relation = spec.truth()
    assert found is (status != "fabricated")
    if status == "mislocated":
        assert found and not located and spec.quote_home != spec.claimed
    elif status == "fabricated":
        assert not located and spec.quote_home is None
    else:
        assert located and spec.quote_home == spec.claimed
    assert relation in ("supports", "contradicts", "says_nothing")
    assert sp.verdict_of(found, located, relation) == (
        "unsupported" if status == "mislocated" else status
    )


def test_every_section_has_two_distinct_sentences() -> None:
    """输入：全部 case。输出：无。

    预期行为：每节恰好两句、全文任何一句只出现一次——同一句话出现两次会让「引文在哪个
    小节」失去唯一解。
    """
    for case in sp.build_cases(tiny_config()):
        spec = _spec(case)
        keys = [item.key() for section in spec.sections for item in section.facts]
        assert len(keys) == 2 * case.sections, case.describe()
        assert len(set(keys)) == len(keys), case.describe()
        assert [section.index for section in spec.sections] == list(range(1, case.sections + 1))


def test_claim_sentence_is_never_a_document_sentence_of_the_claimed_section() -> None:
    """输入：全部 case。输出：无。

    预期行为：声称的小节里放的那句决定了结论档——「通过」放主张本身、「反驳」放它的反面、
    「证据不足 / 编造」放的是一句与主张无关的句子。
    """
    for case in sp.build_cases(tiny_config()):
        spec = _spec(case)
        keys = {item.key() for item in spec.claimed_section.facts}
        if case.status == "verified":
            assert spec.claim.key() in keys
        elif case.status == "contradicted":
            assert spec.claim.opposite.key() in keys
        else:
            assert spec.claim.key() not in keys
            assert spec.claim.opposite.key() not in keys


def test_truncated_quotes_come_from_a_sentence_that_really_is_in_the_document() -> None:
    """输入：全部「编造 × 截断」的 case。输出：无。

    预期行为：来源句**真的在文档里**（渲染层要靠它写引文），但引文本身不在文档里
    （前半句不是整句），所以 `quote_home` 仍为 None。
    """
    checked = 0
    for case in sp.build_cases(tiny_config()):
        if case.status != "fabricated" or case.quote_form != "truncated":
            continue
        spec = _spec(case)
        assert spec.quote_source is not None, case.describe()
        assert spec.quote_home is None, case.describe()
        assert sp.section_of_fact(spec.sections, spec.quote_source) is not None
        checked += 1
    assert checked > 0


def test_reworded_quotes_have_no_source_sentence() -> None:
    """输入：全部「编造 × 转述」的 case。输出：无。预期行为：引文写的是主张的复述，没有来源句。"""
    for case in sp.build_cases(tiny_config()):
        if case.status != "fabricated" or case.quote_form != "reworded":
            continue
        spec = _spec(case)
        assert spec.quote_source is None and spec.quote_home is None, case.describe()


def test_off_topic_section_comes_from_another_topic() -> None:
    """输入：全部 `off_topic` 的 case。输出：无。预期行为：有一整节的标题与句子都来自别的主题。"""
    checked = 0
    config = sp.CitationConfig(
        topics=sp.TOPIC_NAMES,
        statuses=("verified",),
        section_counts=(3,),
        noises=("off_topic",),
        claim_styles=(1,),
        wheres=("middle",),
        doc_styles=("bracketed",),
    )
    for case in sp.build_cases(config):
        spec = _spec(case)
        strangers = [section for section in spec.sections if section.heading_topic != case.topic]
        assert len(strangers) == 1, case.describe()
        assert all(fact.topic != case.topic for fact in strangers[0].facts)
        checked += 1
    assert checked > 0


def test_decoy_section_holds_the_opposite_of_the_claim() -> None:
    """输入：全部 `decoy` 的 case。输出：无。

    预期行为：主张的反面被放在**既不是声称出处、也不是引文老家**的那一节里，
    用来钓「只看全文有没有反例」的偷懒解法。
    """
    config = sp.CitationConfig(
        topics=("auth",),
        statuses=("verified",),
        section_counts=(3,),
        noises=("decoy",),
        claim_styles=(1,),
        wheres=("middle",),
        doc_styles=("bracketed",),
    )
    for case in sp.build_cases(config):
        spec = _spec(case)
        homes = [
            section.index
            for section in spec.sections
            if any(item.key() == spec.claim.opposite.key() for item in section.facts)
        ]
        assert len(homes) == 1, case.describe()
        assert homes[0] not in {spec.claimed, spec.quote_home}, case.describe()


def test_position_of_maps_the_three_places() -> None:
    """输入：位置档与小节数。输出：小节编号。预期行为：首 / 尾 / 中间（三节时是 2）。"""
    assert sp._position_of("first", 3) == 1
    assert sp._position_of("last", 3) == 3
    assert sp._position_of("middle", 3) == 2
    assert sp._position_of("middle", 4) == 2
    with pytest.raises(ValueError, match="中间"):
        sp._position_of("middle", 2)
    with pytest.raises(ValueError, match="未知位置档"):
        sp._position_of("sideways", 3)


def test_fillers_skip_the_claim_and_refuse_to_pad_with_duplicates() -> None:
    """输入：句子池、主张、已用键、要几句。输出：若干句。

    预期行为：不碰主张那一句（否则「没提」会变成「支持 / 反驳」）；凑不够就报错，
    不静默返回短列表。
    """
    claim = sp.Fact("auth", "token_rotation")
    used: set[tuple[str, str, bool]] = set()
    got = sp._fillers(sp.SENTENCE_IDS["auth"], claim, used, 3, random.Random(1))
    assert len(got) == 3
    assert all(fact.sentence != claim.sentence for fact in got)
    assert all(fact.key() in used for fact in got)
    with pytest.raises(ValueError, match="句子池不够"):
        sp._fillers(sp.SENTENCE_IDS["auth"], claim, set(), 20, random.Random(1))


def test_spec_section_lookup_reports_missing_indices() -> None:
    """输入：越界的小节编号。输出：无。预期行为：报人话，不返回空小节。"""
    spec = _spec(_case())
    assert spec.section(1).index == 1
    assert spec.claimed_section.index == spec.claimed
    with pytest.raises(KeyError, match="没有编号"):
        spec.section(99)


def test_case_helpers_are_stable() -> None:
    """输入：一个 case。输出：无。预期行为：`count` / `as_dict` / `describe` 与字段一致。"""
    case = _case(sections=2, where="last")
    assert case.count == 2
    assert list(case.as_dict()) == [
        "topic",
        "status",
        "quote_form",
        "sections",
        "noise",
        "claim_style",
        "where",
        "doc_style",
    ]
    assert "2节" in case.describe()


# ---------- 旋钮校验 ----------


def test_config_rejects_bad_knobs_at_construction_time() -> None:
    """输入：坏旋钮。输出：ValueError。预期行为：构造期就报人话，不拖到采样阶段。"""
    with pytest.raises(ValueError, match="topics"):
        sp.CitationConfig(topics=())
    with pytest.raises(ValueError, match="不存在的取值"):
        sp.CitationConfig(topics=("auth", "nope"))
    with pytest.raises(ValueError, match="section_counts"):
        sp.CitationConfig(section_counts=(5,))
    with pytest.raises(ValueError, match="statuses"):
        sp.CitationConfig(statuses=("verified", "made_up"))
