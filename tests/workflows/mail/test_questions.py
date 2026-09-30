"""题族构造测试：题号、题型、选项与答案形态。"""

from __future__ import annotations

import pytest

from workflows.mail import features, locales, questions

LOCALES = locales.available()


def _build(case: features.TagSet, lang: str = "zh") -> list[questions.LabeledQuestion]:
    """输入：一组特征、语言。输出：该特征下的题目列表。"""
    return questions.build(case, locales.load(lang), phrasing=0)


@pytest.mark.parametrize("lang", LOCALES)
def test_question_order_and_families(lang: str) -> None:
    """题号顺序必须按 `FAMILIES` 走，题号就是题族名。"""
    case = features.TagSet("known_vendor", "none", "none", "official_honest", "formal")
    built = _build(case, lang)
    assert [item.family for item in built] == list(questions.FAMILIES)


def test_linkless_mail_has_no_link_text_question() -> None:
    """没有链接就不问链接文字——否则那道题会退化成 11/44/44 的偏斜分布。"""
    case = features.TagSet("known_vendor", "none", "none", "none", "formal")
    families = [item.family for item in _build(case)]
    assert "link_text" not in families
    assert len(families) == len(questions.FAMILIES) - 1


def test_choice_answers_are_option_names() -> None:
    """choice 的答案必须是选项名本身（编号由组装器最后一步映射）。"""
    case = features.TagSet("unknown_personal", "curiosity", "click_link", "lookalike_mismatch", "casual")
    for item in _build(case):
        if item.question["type"] != "choice":
            continue
        assert item.answer in item.question["criteria"], item.family


def test_ordered_family_is_not_shuffled() -> None:
    """处置是阶梯，必须保持顺序；其余 choice 允许打乱以打掉位置捷径。"""
    case = features.TagSet("forged_brand", "fear", "credentials", "ip_mismatch", "curt")
    by_family = {item.family: item for item in _build(case)}
    assert by_family["action"].shuffle is False
    assert by_family["action"].question["criteria"] == {
        "deliver": locales.load("zh")["fam"]["action"]["criteria"]["deliver"],
        "review": locales.load("zh")["fam"]["action"]["criteria"]["review"],
        "quarantine": locales.load("zh")["fam"]["action"]["criteria"]["quarantine"],
        "block": locales.load("zh")["fam"]["action"]["criteria"]["block"],
    }
    for family in ("requested", "pressure", "link_text", "sender_identity"):
        assert by_family[family].shuffle is True


def test_choice_options_follow_the_contract_table() -> None:
    """选项表必须与 `features.CHOICE_OPTIONS` 逐项一致（含顺序）。"""
    case = features.TagSet("known_vendor", "none", "none", "official_mismatch", "formal")
    for item in _build(case):
        if item.question["type"] != "choice":
            continue
        assert list(item.question["criteria"]) == list(features.CHOICE_OPTIONS[item.family])


def test_score_answer_is_the_level_index() -> None:
    """严重度题的答案是等级下标，等级数与契约上限一致。"""
    case = features.TagSet("forged_brand", "fear", "credentials", "ip_mismatch", "curt")
    item = {q.family: q for q in _build(case)}["severity"]
    assert item.question["type"] == "score"
    assert item.answer == case.label().severity
    assert 0 <= item.answer < len(item.question["criteria"])


def test_noul_answer_is_a_bool() -> None:
    """noul 的答案只能是 JSON 布尔。"""
    case = features.TagSet("known_vendor", "none", "none", "none", "formal")
    for item in _build(case):
        if item.question["type"] == "noul":
            assert isinstance(item.answer, bool)


@pytest.mark.parametrize("lang", LOCALES)
def test_phrasing_index_wraps(lang: str) -> None:
    """问法编号越界要回绕，不许报错也不许给出空串。"""
    case = features.TagSet("known_vendor", "none", "none", "none", "formal")
    loc = locales.load(lang)
    count = len(loc["fam"]["requested"]["q"])
    assert questions.build(case, loc, phrasing=count + 1)[0].question["instructions"] == (
        questions.build(case, loc, phrasing=1)[0].question["instructions"]
    )
