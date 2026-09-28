"""语言文件与代码常量的契约测试。

这组测试是**锁**：`TraceDraft` 的 `Literal`、locale 里 `criteria` 的 key、
`features` 里的维度取值，三者只要有一处漂移就立刻红灯。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from typing import get_args

import pytest

from workflows.mail import features, judges, locales, questions

LOCALES = locales.available()


def test_locales_are_present() -> None:
    """至少要中英两份，且文件名就是语言代码。"""
    assert set(LOCALES) == {"zh", "en"}


def test_missing_language_file_lists_what_exists() -> None:
    """输入：不存在的语言代码。输出：FileNotFoundError。

    预期行为：报错里必须列出可用语言，否则调用方只能猜。
    """
    with pytest.raises(FileNotFoundError, match="可用"):
        locales.load("fr")


def test_pick_wraps_around() -> None:
    """输入：候选列表、越界编号。输出：回绕后的那一项。"""
    assert locales.pick(["a", "b"], 3) == "b"


@pytest.mark.parametrize("lang", LOCALES)
def test_feature_phrases_cover_every_dimension_value(lang: str) -> None:
    """每个维度取值都要有 brief 与 hint——缺一条就等于让模型自由发挥。"""
    loc = locales.load(lang)
    for dim, values in features.DIMS.items():
        for value in values:
            spec = loc["feature"][dim][value]
            assert spec["brief"].strip()
            assert spec["hint"].strip()


@pytest.mark.parametrize("lang", LOCALES)
def test_families_match_questions_module(lang: str) -> None:
    """locale 里的题族集合必须与 `questions.FAMILIES` 完全一致。"""
    loc = locales.load(lang)
    assert set(loc["fam"]) == set(questions.FAMILIES)


@pytest.mark.parametrize("lang", LOCALES)
def test_choice_criteria_keys_match_feature_values(lang: str) -> None:
    """choice 题的选项名就是标签取值，必须与特征层的取值集合一致。"""
    loc = locales.load(lang)
    assert set(loc["fam"]["requested"]["criteria"]) == set(features.ASK)
    assert set(loc["fam"]["pressure"]["criteria"]) == set(features.LURE)
    # link_text 不再需要「完整答案空间」：那一族由零 token 判别器数地址得出、不进反解提示词，
    # 所以 locale 里只留数据集真正会出的那两个选项
    assert set(loc["fam"]["link_text"]["criteria"]) == set(features.CHOICE_OPTIONS["link_text"])
    assert set(loc["fam"]["sender_identity"]["criteria"]) == {"matches", "no_claim", "forged"}
    assert set(loc["fam"]["action"]["criteria"]) == {"deliver", "review", "quarantine", "block"}
    for family, options in features.CHOICE_OPTIONS.items():
        assert options, family
        assert set(options) <= set(loc["fam"][family]["criteria"]), family


@pytest.mark.parametrize("lang", LOCALES)
def test_pressure_criteria_mirror_the_render_contract(lang: str) -> None:
    """`urgency` 要明确时限、`curiosity` 不许被催促语抢走——判据必须与渲染契约对齐。

    输入：语言。输出：无。预期行为：渲染侧要求 `urgency` 必须写时间窗
    （见 `feature.lure.urgency.hint`），判别侧就得把「光有催促语、没有时限」排除掉；
    `curiosity` 的渲染契约**不禁**催促语（「尽快回复」是任何邮件都会有的客套），
    判别侧就得说明催促语抢不走这一档。两处口径对不上就会拒掉好数据：
    2026-09-26 逐条读 13 条 `pressure` 拒绝 → 误拒 8、真阳性 5，其中 3 条误拒正是
    被催促语抢走的 `curiosity`。
    """
    loc = locales.load(lang)
    window = {"zh": "时限", "en": "deadline"}[lang]
    hurry = {"zh": "催促", "en": "hurry"}[lang]
    urgency = loc["fam"]["pressure"]["criteria"]["urgency"]
    curiosity = loc["fam"]["pressure"]["criteria"]["curiosity"]
    assert window in urgency, "urgency 判据没要求明确时限"
    assert hurry in urgency, "urgency 判据没排除「光催促、无时限」"
    assert hurry in curiosity, "curiosity 判据没说清催促语抢不走这一档"


@pytest.mark.parametrize("lang", LOCALES)
def test_noul_and_score_shapes(lang: str) -> None:
    """noul 只能有 true/false 两个 key；score 的等级数与严重度上限一致。"""
    loc = locales.load(lang)
    for family in questions.NOUL_FAMILIES:
        assert set(loc["fam"][family]["criteria"]) == {"true", "false"}
    assert len(loc["fam"]["severity"]["levels"]) == 5


def test_trace_draft_literals_match_the_contract() -> None:
    """`TraceDraft`（反解问到的原子族）的取值域必须与特征层常量、locale 的选项 key 三处一致。

    输入：无。输出：无。预期行为：任何一处漂移都红灯——反解用的选项和数据集里的选项
    必须是同一套，否则「反解通过」不代表「标签正确」。字段集合本身也钉死：
    换一族进 / 出反解提示词必须连着改这条测试，不能悄悄漂。
    """
    expected: dict[str, set[str]] = {
        "requested": set(features.ASK),
        "pressure": set(features.LURE),
    }
    for lang in LOCALES:
        loc: dict[str, JsonValue] = locales.load(lang)
        assert set(judges.TraceDraft.model_fields) == {*expected, "extra_requests"}
        for field, values in expected.items():
            assert set(get_args(judges.TraceDraft.model_fields[field].annotation)) == values
            assert set(loc["fam"][field]["criteria"]) == values, f"{lang}.{field} 与 TraceDraft 不一致"


@pytest.mark.parametrize("lang", LOCALES)
def test_verify_prompt_does_not_ask_link_text(lang: str) -> None:
    """`link_text` 不进反解提示词：它的真值由零 token 判别器数链接行里的地址得出。

    输入：语言。输出：无。预期行为：提示词里不出现那道题——留着它只会多一处误拒
    （2026-09-26 实测：这一族的失败占了全部失败的一半，而纯代码规则 129/130 判对）。
    """
    loc = locales.load(lang)
    prompt = judges.build_verify_prompt("From: a <b@c>\nSubject: s\n\nbody", loc)
    assert loc["fam"]["link_text"]["q"][0] not in prompt


@pytest.mark.parametrize("lang", LOCALES)
def test_verify_prompt_does_not_ask_sender_identity(lang: str) -> None:
    """`sender_identity` 不进反解提示词：它的真值由零 token 判别器读「官方域名」行得出。

    输入：语言。输出：无。预期行为：提示词里不出现那道题——装配时的
    `check_official_domain` 已经强制「声称机构 ⟺ 官方域名行非空」，所以这一族也是
    结构事实（2026-09-26 在 167/168 条真产物上与注入标签一致）。
    """
    loc = locales.load(lang)
    prompt = judges.build_verify_prompt("From: a <b@c>\nSubject: s\n\nbody", loc)
    assert loc["fam"]["sender_identity"]["q"][0] not in prompt


@pytest.mark.parametrize("lang", LOCALES)
def test_verify_prompt_mentions_every_option(lang: str) -> None:
    """反解提示词必须把选项名与描述都带上，模型才可能按同一套口径回答。"""
    loc = locales.load(lang)
    prompt = judges.build_verify_prompt("From: a <b@c>\nSubject: s\n\nbody", loc)
    for family in judges.TRACE_FAMILIES:
        for name in loc["fam"][family]["criteria"]:
            assert name in prompt


@pytest.mark.parametrize("lang", LOCALES)
def test_render_prompt_is_only_the_feature_sheet(lang: str) -> None:
    """渲染提示词必须**只**由特征清单与硬性要求组成——多一行都可能泄漏答案。

    输入：语言、一组特征。输出：无。
    预期行为：逐行比对，提示词的每一行都必须能在「开头 / 特征行 / 硬性要求行」里找到出处。
    这比「不许出现某个词」严格得多，也不会被英文单词的巧合误伤。
    """
    loc = locales.load(lang)
    case = features.TagSet("forged_brand", "urgency", "credentials", "ip_mismatch", "curt")
    from workflows.mail import render

    prompt = render.build_prompt(case, loc)
    values = case.as_dict()
    allowed = {loc["prompt"]["intro"], ""}
    allowed |= {locales.feature_line(loc, dim, values[dim]) for dim in render.DIM_ORDER}
    allowed |= {
        "- " + rule.format(language=loc["language_name"]) for rule in loc["prompt"]["rules"]
    }
    assert set(prompt.split("\n")) <= allowed


@pytest.mark.parametrize("lang", LOCALES)
def test_verify_prompt_asks_about_smuggled_requests(lang: str) -> None:
    """反解提示词要多问一道「额外索取」，且选项里不含「什么都不索取」那一档。

    输入：语言代码。输出：无。
    预期行为：夹带检测靠这一问，缺了它单选判别器永远看不出「文本有两个索取」；
    `none` 那一档是「什么都不做」的意思，不该混进「还索取了什么」的选项里。
    """
    loc = locales.load(lang)
    prompt = judges.build_verify_prompt("From: a <b@c>\nSubject: s\n\nbody", loc)
    assert loc["verify"]["extra_question"] in prompt
    extra_block = prompt.split(loc["verify"]["extra_question"], 1)[1]
    assert f"   - {features.ASK_NONE}" not in extra_block
    for name in features.ASK:
        if name != features.ASK_NONE:
            assert f"   - {name}" in extra_block
