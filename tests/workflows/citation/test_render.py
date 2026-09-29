"""文本外壳测试：引文剪法、草稿校验、state 往返、语言识别。"""

from __future__ import annotations

import pytest

from workflows.citation import features, locales, render
from workflows.citation.spec import CitationSpec
from tests.workflows.citation.conftest import (
    SENTENCES,
    TITLE,
    FakeBackend,
    draft_data,
    spec_for,
)


def _draft(**switches: bool) -> render.CitationDraft:
    """输入：草稿开关。输出：草稿模型。预期行为：默认是一份完全兑现计划的草稿。"""
    data = draft_data(**switches)
    return render.CitationDraft(
        title=str(data["title"]), sentences=[str(item) for item in data["sentences"]]
    )


def _spec(
    form: str = "exact", where: str = "same", place: str = "first", target: str = "first"
) -> CitationSpec:
    """输入：四个特征取值。输出：真值规格。预期行为：测试里随手造规格的便捷入口。"""
    return spec_for(features.CitationCase(form, where, place, target))


def test_sections_of_chunks_by_two() -> None:
    """输入：6 句话。输出：无。预期行为：每 2 句一节，顺序不变。"""
    groups = render.sections_of(list(SENTENCES))
    assert len(groups) == features.SECTIONS
    assert all(len(group) == features.SENTENCES_PER_SECTION for group in groups)
    assert groups[0] == SENTENCES[:2]
    assert groups[2] == SENTENCES[4:]


def test_sec_ref_is_one_based() -> None:
    """输入：0 起的节下标。输出：无。预期行为：0 → 第 1 节（差一格是最容易犯的错）。"""
    loc = locales.load("zh")
    assert render.sec_ref(loc, 0) == "第 1 节"
    assert render.sec_ref(loc, 2) == "第 3 节"


def test_make_quote_exact_is_verbatim() -> None:
    """输入：一句原文。输出：无。预期行为：整句照抄，连句末标点都在。"""
    assert render.make_quote(SENTENCES[0], features.FORM_EXACT, SENTENCES) == SENTENCES[0]


def test_make_quote_partial_is_a_proper_substring() -> None:
    """输入：一句原文。输出：无。预期行为：截前半句，是原句的真子串、且不短于下限。"""
    quote = render.make_quote(SENTENCES[1], features.FORM_PARTIAL, SENTENCES)
    assert quote in SENTENCES[1]
    assert quote != SENTENCES[1]
    assert len(quote) >= features.MIN_PARTIAL


def test_partial_quote_backs_off_to_a_word_boundary() -> None:
    """输入：一句以空格分词的英文。输出：无。

    预期行为：半句引文不砍在单词中间。真机第二批有一条引文是
    `The final release includes 8 downloadable CSV fi`——字符确实在文档里，可它砍在 `files`
    中间，读者模型当场判成「文档里没有这句」。退到词边界之后一眼就能对上。
    """
    sentence = (
        "The final release includes 8 downloadable CSV files, "
        "each timestamped with the current year 2026."
    )
    quote = render.make_quote(sentence, features.FORM_PARTIAL, (sentence,))
    assert quote in sentence and quote != sentence
    assert quote.split()[-1] in sentence.split()  # 最后一个词是完整的词，不是半截
    assert sentence[len(quote)] == " "


def test_partial_quote_keeps_a_character_cut_for_text_without_spaces() -> None:
    """输入：一句没有空格的中文。输出：无。

    预期行为：找不到词边界时就按字数截——中文按字数截断本来就是常态，不该退成空片段。
    """
    sentence = "本次巡检记录显示泵体压力偏高，随后回落到正常区间。"
    quote = render.make_quote(sentence, features.FORM_PARTIAL, (sentence,))
    assert quote in sentence and quote != sentence
    assert len(quote) == len(sentence) // 2


def test_make_quote_altered_replaces_a_digit_and_stays_out_of_the_doc() -> None:
    """输入：一句原文。输出：无。预期行为：换掉一个数字，换完整篇文档一处都不出现。"""
    quote = render.make_quote(SENTENCES[2], features.FORM_ALTERED, SENTENCES)
    assert quote != SENTENCES[2]
    assert quote.split()[5] == "8"  # 原来的 7 被换掉了
    assert all(quote not in item for item in SENTENCES)


def test_make_quote_altered_skips_a_colliding_replacement() -> None:
    """输入：两句只在数字上不同的文档。输出：无。

    预期行为：直接把 `41` 换成 `42` 会**正好等于另一句**，于是往后试到不撞为止——这是「查无
    此文」这条判据唯一会被绕过的地方，必须机械挡住。
    """
    twin = (
        "Line one reported 41 units at the gate.",
        "Line one reported 42 units at the gate.",
    )
    quote = render.make_quote(twin[0], features.FORM_ALTERED, twin)
    assert quote == "Line one reported 43 units at the gate."


@pytest.mark.parametrize(
    ("sentence", "form", "match"),
    [
        ("23 H", features.FORM_PARTIAL, "太短"),
        ("no digits in this sentence at all", features.FORM_ALTERED, "没有具体数字"),
        (SENTENCES[0], "nope", "未知造法"),
    ],
)
def test_make_quote_reports_bad_inputs(sentence: str, form: str, match: str) -> None:
    """输入：太短的句子 / 没有数字的句子 / 未知造法。输出：无。预期行为：都当场报人话。"""
    with pytest.raises(ValueError, match=match):
        render.make_quote(sentence, form, SENTENCES)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_prompt_spells_out_the_document_requirements(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：节数、每节句数、数字、长度四条硬要求都在。"""
    loc = locales.load(lang)
    prompt = render.build_prompt(loc)
    assert "{" not in prompt and "}" not in prompt
    assert str(features.SENTENCE_COUNT) in prompt
    assert str(render.MIN_SENTENCE_CHARS) in prompt
    assert str(loc["prompt"]["digits_rule"]) in prompt


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_prompt_carries_a_sentence_by_sentence_self_check(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：提示词末尾把「每句都得有阿拉伯数字」再自查一遍——真机第一批三条拒稿全是同一个
    理由（模型写了「一年」「四大发明」这种汉字数字），这条自查就是照那个加的。
    """
    loc = locales.load(lang)
    prompt = render.build_prompt(loc)
    assert str(loc["prompt"]["self_check"]) in prompt


def test_check_draft_accepts_a_faithful_draft() -> None:
    """输入：一份完全兑现计划的草稿。输出：无。预期行为：不抛异常（三种造法都试一遍）。"""
    for form in features.FORMS:
        render.check_draft(_draft(), _spec(form=form).plan, locales.load("zh"))


@pytest.mark.parametrize(
    ("switch", "match"),
    [
        ("duplicate", "句子重复"),
        ("drop_sentence", "at least 6 items"),
        ("no_digit", "没有具体数字"),
        ("short_sentence", "句子太短"),
        ("fence_line", "带围栏"),
        ("title_fence", "带围栏"),
    ],
)
def test_check_draft_rejects_broken_drafts(switch: str, match: str) -> None:
    """输入：每种「故意写歪」。输出：无。预期行为：当场抛 `ValueError`，理由对得上。"""
    with pytest.raises(ValueError, match=match):
        render.check_draft(_draft(**{switch: True}), _spec().plan, locales.load("zh"))


def test_check_draft_rejects_a_quote_that_leaks_into_another_section() -> None:
    """输入：半句引文在别的小节里也出现的文档。输出：无。

    预期行为：不通过——引文一旦跨节出现，「它出自第几节」就没有唯一解，那一条的答案会错。
    """
    shared = "Shared opening words here and a reading of {n} units."
    sentences = [
        shared.format(n=41),
        "A second probe logged the drift as 12 points.",
        "Operators swapped the gasket after 7 failed checks.",
        shared.format(n=77),
        "Night shift recorded a peak of 23 hertz.",
        "Final inspection found 64 loose fasteners.",
    ]
    draft = render.CitationDraft(title=TITLE, sentences=sentences)
    with pytest.raises(ValueError, match="该只出在第 1 节"):
        render.check_draft(draft, _spec(form="partial", place="first", target="first").plan, locales.load("zh"))


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_state_round_trips_through_parse(lang: str) -> None:
    """输入：两种语言各渲染一次。输出：无。

    预期行为：标题、三节句子、引文、声称出处逐项原样反解回来。
    """
    spec = _spec(form="partial", where="other", place="middle", target="last")
    loc = locales.load(lang)
    draft = _draft()
    state = render.to_state(draft, spec.plan, loc)
    parsed = render.parse_with(state, loc)
    assert parsed.title == TITLE
    assert parsed.sentences == tuple(SENTENCES)
    assert parsed.sections[1] == SENTENCES[2:4]
    assert parsed.quote == render.make_quote(
        SENTENCES[spec.plan.sentence_index], spec.plan.form, SENTENCES
    )
    assert parsed.claimed == spec.plan.claimed
    assert render.detect_locale(state) == lang


def test_detect_locale_does_not_let_one_language_steal_the_other() -> None:
    """输入：中英各一段 state。输出：无。

    预期行为：各认各的——节抬头只有「第 1 节」这几个字，靠它认语言会让先试的那一种抢走全部
    state；认的是文档抬头（这条坑在 lint 上踩过）。
    """
    spec = _spec()
    draft = _draft()
    zh_state = render.to_state(draft, spec.plan, locales.load("zh"))
    en_state = render.to_state(draft, spec.plan, locales.load("en"))
    assert render.detect_locale(zh_state) == "zh"
    assert render.detect_locale(en_state) == "en"
    assert render.detect_locale("随便一段没有抬头的文本") is None
    assert render.detect_locale(123) is None
    empty = render.parse_state("随便一段没有抬头的文本")
    assert empty.title is None and empty.sections == () and empty.quote is None


def test_unknown_locale_reports_available() -> None:
    """输入：不存在的语言。输出：无。预期行为：抛 `FileNotFoundError` 并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="可用"):
        locales.load("nope")


def test_build_task_carries_the_plan() -> None:
    """输入：规格。输出：无。预期行为：任务 id 与 spec 里的造法 / 出处都对得上。"""
    spec = _spec(form="altered", where="other", place="last", target="last")
    task = render.build_task(spec, locales.load("zh"))
    assert task.task_id == spec.render_task_id
    assert task.spec["form"] == "altered"
    assert task.spec["true_section"] == 2
    assert task.spec["claimed"] == spec.plan.claimed
    assert task.response_model is render.CitationDraft


def test_backend_fake_produces_parseable_drafts() -> None:
    """输入：假后端。输出：无。预期行为：它造的草稿真的能过校验（假后端自己得先站得住）。"""
    spec = _spec(form="partial", where="same", place="last", target="last")
    task = render.build_task(spec, locales.load("zh"))
    draft = FakeBackend().generate(task)
    parsed = render.CitationDraft(
        title=str(draft.data["title"]),
        sentences=[str(item) for item in draft.data["sentences"]],
    )
    render.check_draft(parsed, spec.plan, locales.load("zh"))


def test_state_has_every_heading() -> None:
    """输入：一段渲染好的 state。输出：无。预期行为：五种抬头各出现一次。"""
    loc = locales.load("zh")
    state = render.to_state(_draft(), _spec().plan, loc)
    marks: list[str] = [
        str(loc["state"]["doc_mark"]),
        *[str(loc["state"]["sec_mark"]).format(n=index + 1) for index in range(features.SECTIONS)],
        str(loc["state"]["cite_mark"]),
        str(loc["state"]["claim_mark"]),
    ]
    for mark in marks:
        assert state.count(mark) == 1, mark
