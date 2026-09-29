"""特征层测试：定位、两条标签路径的一致性、case 枚举。"""

from __future__ import annotations

import pytest

from workflows.citation import features
from tests.workflows.citation.conftest import SENTENCES

def sections() -> tuple[tuple[str, ...], ...]:
    """输入：无。输出：`SENTENCES` 按每节两句切好的文档。"""
    return tuple(
        tuple(SENTENCES[start : start + features.SENTENCES_PER_SECTION])
        for start in range(0, len(SENTENCES), features.SENTENCES_PER_SECTION)
    )


def test_locate_finds_the_section_and_reports_misses() -> None:
    """输入：一句原文 / 一段查不到的文字。输出：无。预期行为：命中给节号，查不到给 None。"""
    doc = sections()
    assert features.locate(doc, SENTENCES[0]) == 0
    assert features.locate(doc, SENTENCES[3]) == 1
    assert features.locate(doc, SENTENCES[5]) == 2
    assert features.locate(doc, "this sentence is nowhere in the document") is None


def test_observed_labels_reads_the_document_only() -> None:
    """输入：三种情形（查得到且出处对 / 查得到但出处错 / 查不到）。输出：无。"""
    doc = sections()
    assert features.observed_labels(doc, SENTENCES[0], 0).verdict == features.VERDICT_VERIFIED
    assert features.observed_labels(doc, SENTENCES[0], 1).verdict == features.VERDICT_MISLOCATED
    assert features.observed_labels(doc, "not in the doc", 0).verdict == features.VERDICT_FABRICATED


def test_at_claimed_needs_both_being_in_doc_and_the_right_section() -> None:
    """输入：三种标签。输出：无。预期行为：只有「在文档里且落在声称那一节」才算 True。"""
    doc = sections()
    assert features.observed_labels(doc, SENTENCES[0], 0).at_claimed is True
    assert features.observed_labels(doc, SENTENCES[0], 2).at_claimed is False
    assert features.observed_labels(doc, "not in the doc", 0).at_claimed is False


@pytest.mark.parametrize("form", features.FORMS)
@pytest.mark.parametrize("where", features.WHERES)
@pytest.mark.parametrize("place", features.PLACES)
@pytest.mark.parametrize("target", features.TARGETS)
def test_both_label_paths_agree(form: str, where: str, place: str, target: str) -> None:
    """输入：全部 36 组特征。输出：无。

    预期行为：**构造侧**（从计划推）与**审计侧**（从文本重推）得到同一份标签——两边对不上就
    说明渲染没兑现计划，这条把它钉在每一组取值上。
    """
    case = features.CitationCase(form=form, where=where, place=place, target=target)
    plan = features.plan_for(case, 0)
    index = plan.sentence_index
    quote = {
        features.FORM_EXACT: SENTENCES[index],
        features.FORM_PARTIAL: SENTENCES[index][: len(SENTENCES[index]) // 2],
        features.FORM_ALTERED: "a quote that is nowhere in the document",
    }[form]
    assert features.observed_labels(sections(), quote, plan.claimed) == features.label_for(plan)


def test_plan_is_reproducible() -> None:
    """输入：同一 case 的两次计算。输出：无。预期行为：同 (case, 问法) 必得同一份计划。"""
    case = features.CitationCase(form="exact", where="other", place="first", target="first")
    assert features.plan_for(case, 0) == features.plan_for(case, 0)
    assert features.plan_for(case, 1) == features.plan_for(case, 1)


def test_phrasing_moves_the_claimed_section_somewhere() -> None:
    """输入：同一 case 的两种问法。输出：无。

    预期行为：`where=other` 时声称出处由「指纹 + 问法」掷出，**至少有一个 case** 在两种问法下
    掷到不同的节——否则两种问法只会渲染出结构相同的样本，问法池就成了摆设。

    为什么不断言「每个 case 都不同」：两节里选一节，两种问法掷到同一节是完全正常的。
    """
    moved = [
        case
        for case in features.build_cases(512)
        if case.where == features.WHERE_OTHER
        and features.plan_for(case, 0).claimed != features.plan_for(case, 1).claimed
    ]
    assert moved
    for case in moved:
        plan = features.plan_for(case, 0)
        assert plan.claimed != plan.true_section  # 声称的只能是另外两节


def test_same_where_claims_the_true_section() -> None:
    """输入：`where=same` 的计划。输出：无。预期行为：声称出处就是真正出处。"""
    case = features.CitationCase(form="partial", where="same", place="last", target="last")
    plan = features.plan_for(case, 0)
    assert plan.claimed == plan.true_section == 2
    assert features.label_for(plan).verdict == features.VERDICT_VERIFIED


def test_altered_is_always_fabricated() -> None:
    """输入：三种造法各一条计划。输出：无。预期行为：只有 `altered` 落成查无此文。"""
    for form in features.FORMS:
        case = features.CitationCase(form=form, where="same", place="first", target="first")
        labels = features.label_for(features.plan_for(case, 0))
        if form == features.FORM_ALTERED:
            assert labels.verdict == features.VERDICT_FABRICATED
            assert labels.in_doc is False and labels.section is None
            assert labels.at_claimed is False
        else:
            assert labels.verdict == features.VERDICT_VERIFIED
            assert labels.in_doc is True and labels.section == 0


def test_sentence_index_maps_place_and_target() -> None:
    """输入：全部 (落点, 第几句) 组合。输出：无。预期行为：下标 = 节号 × 每节句数 + 句内序号。"""
    for place, base in features.PLACE_INDEX.items():
        for target, offset in features.TARGET_INDEX.items():
            case = features.CitationCase(form="exact", where="same", place=place, target=target)
            plan = features.plan_for(case, 0)
            assert plan.sentence_index == base * features.SENTENCES_PER_SECTION + offset
            assert plan.sentence_index < features.SENTENCE_COUNT


def test_build_cases_enumerates_the_whole_space() -> None:
    """输入：默认维度表。输出：无。预期行为：全枚举 3×2×3×2 = 36 组，顺序稳定、无重复。"""
    cases = features.build_cases(512)
    assert len(cases) == 36
    assert len({case.fingerprint() for case in cases}) == len(cases)
    assert cases == features.build_cases(512)
    assert cases[0].describe() == "exact-same-first-first"


def test_case_as_dict_has_stable_key_order() -> None:
    """输入：一个 case。输出：无。预期行为：键序固定（哈希分桶按它算，键序变了切分会漂）。"""
    case = features.CitationCase(form="altered", where="other", place="middle", target="last")
    assert list(case.as_dict()) == ["form", "where", "place", "target"]
    assert case.fingerprint() == "form=altered|where=other|place=middle|target=last"
