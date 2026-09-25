"""语言文件与代码常量的契约测试。

这组测试是**锁**：`TraceDraft` 的 `Literal`、locale 里 `criteria` 的 key、
`features` 里的维度取值，三者只要有一处漂移就立刻红灯。
"""

from __future__ import annotations

import re
from typing import Any, Final, get_args

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
    # locale 里保留完整的答案空间（反解提示词要用），选项表另由 CHOICE_OPTIONS 收窄
    assert set(loc["fam"]["link_text"]["criteria"]) == {"no_link", "matches", "mismatched"}
    assert set(loc["fam"]["sender_identity"]["criteria"]) == {"matches", "no_claim", "forged"}
    assert set(loc["fam"]["action"]["criteria"]) == {"deliver", "review", "quarantine", "block"}
    for family, options in features.CHOICE_OPTIONS.items():
        assert options, family
        assert set(options) <= set(loc["fam"][family]["criteria"]), family


@pytest.mark.parametrize("lang", LOCALES)
def test_noul_and_score_shapes(lang: str) -> None:
    """noul 只能有 true/false 两个 key；score 的等级数与严重度上限一致。"""
    loc = locales.load(lang)
    for family in questions.NOUL_FAMILIES:
        assert set(loc["fam"][family]["criteria"]) == {"true", "false"}
    assert len(loc["fam"]["severity"]["levels"]) == 5


def test_trace_draft_literals_match_the_contract() -> None:
    """`TraceDraft` 的取值域必须与特征层常量、以及 locale 的选项 key 三处一致。

    输入：无。输出：无。预期行为：任何一处漂移都红灯——反解用的选项和数据集里的选项
    必须是同一套，否则「反解通过」不代表「标签正确」。
    """
    expected: dict[str, set[str]] = {
        "requested": set(features.ASK),
        "pressure": set(features.LURE),
        # 反解要覆盖「本来就没链接」的样本，所以这里是**完整答案空间**，不是选项表
        "link_text": {"no_link", "matches", "mismatched"},
        "sender_identity": {"matches", "no_claim", "forged"},
    }
    for lang in LOCALES:
        loc: dict[str, Any] = locales.load(lang)
        for field, values in expected.items():
            assert set(get_args(judges.TraceDraft.model_fields[field].annotation)) == values
            assert set(loc["fam"][field]["criteria"]) == values, f"{lang}.{field} 与 TraceDraft 不一致"


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


#: 渲染侧那条「软性指引」规则的解析口径：语言 → (正则, 分隔符)。
#: 规则里用引号把示例措辞列了一遍，判别侧必须把同一批措辞收进 `click_link` 判据——
#: 两边各写一份清单，就是灰区的来源。
SOFT_POINTER_RULE: Final[dict[str, tuple[str, str]]] = {
    "zh": (r"「([^」]+)」这类\*\*软性指引\*\*", " / "),
    "en": (r"\(([^)]*)\), it just states the matter", ", "),
}


@pytest.mark.parametrize("lang", LOCALES)
def test_click_link_criteria_covers_every_soft_pointer_the_prompt_bans(lang: str) -> None:
    """渲染侧点名禁止的软性指引，判别侧必须能判成 `click_link`。

    输入：语言。输出：无。预期行为：从渲染提示词那条规则里**解析**出措辞清单，
    逐条要求出现在 `fam.requested.criteria.click_link` 里。

    实跑踩过这个灰区（`mail-kimi-ood-b` 34 条里 3 条：「可访问」「了解详情」「请查阅」）：
    规则写「索取不是点击链接时，连软性指引都不许写」，判据却写「**明确**请对方点击」——
    同一句正文渲染侧禁止、判别侧放行，成了两边都不认的灰区。措辞清单只写一处就会再漂回去，
    所以这里从规则里解析、不许人记。
    """
    loc = locales.load(lang)
    pattern, sep = SOFT_POINTER_RULE[lang]
    rule = next((r for r in loc["prompt"]["rules"] if "软性指引" in r or "nudge" in r), "")
    assert rule, f"{lang} 的渲染规则里找不到「软性指引」那条"
    match = re.search(pattern, rule)
    assert match, f"{lang} 的规则里解析不出软性指引措辞清单：{rule}"
    phrases = [part.strip().strip('"') for part in match.group(1).split(sep)]
    assert phrases, f"{lang} 解析出来的措辞清单是空的"
    criteria = loc["fam"]["requested"]["criteria"]["click_link"]
    missing = [phrase for phrase in phrases if phrase not in criteria]
    assert not missing, f"{lang} 的 click_link 判据漏了渲染侧禁止的软性指引：{missing}"
