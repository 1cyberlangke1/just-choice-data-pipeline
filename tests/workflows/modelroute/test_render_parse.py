"""渲染与反解：版式、逐格往返、标签认法、缺格与非字符串的报错。"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from workflows.modelroute import locales, parse, render
from workflows.modelroute.spec import RouteCase, sample_spec

ZH = locales.load("zh")
EN = locales.load("en")


def spec_of(task: str = "extract", **overrides: Any) -> Any:
    """输入：任务名与维度覆盖。输出：真值规格。预期行为：未覆盖的维度取中档默认值。"""
    fields: dict[str, Any] = {
        "task": task,
        "complexity": "mid",
        "latency": "normal",
        "cost": "normal",
        "retry": "first",
        "data": "public",
        "phrasing": 0,
    }
    fields.update(overrides)
    return sample_spec(RouteCase(**fields), "train", 0, 0)


def test_render_layout_is_title_plus_six_fields_plus_body() -> None:
    """输入：一份规格。输出：无。

    预期行为：固定八行——标题 1 行、字段 6 行、正文 1 行；正文**在最后**，
    它带着任务类型（画像里没有「任务类型」那一栏）。
    """
    lines = render.build(spec_of(), ZH).split("\n")
    assert len(lines) == 8
    assert lines[0] == ZH["meta"]["title"]
    assert lines[-1].startswith(ZH["meta"]["body_label"])


def test_render_field_values_come_from_the_locale() -> None:
    """输入：一份规格。输出：无。预期行为：每一格写的都是语言文件里的原文。"""
    spec = spec_of(task="codefix", complexity="high", latency="strict", cost="tight", retry="retry", data="private")
    text = render.build(spec, ZH)
    assert ZH["complexity"]["high"] in text
    assert ZH["latency"]["strict"] in text
    assert ZH["cost"]["tight"] in text
    assert ZH["retry"]["retry"] in text
    assert ZH["data"]["private"] in text
    assert spec.request_id in text


def test_body_text_only_says_what_to_do() -> None:
    """输入：四个任务的规格。输出：无。

    预期行为：正文就是那一句模板，**不复述任何字段值**（复杂度 / 延迟 / 成本 / 数据分级
    都在字段行里），也不带分隔符——正文是判据的一部分，不是第二张表。
    """
    for task in ("classify", "extract", "codefix", "reason"):
        spec = spec_of(task=task)
        body = render.body_text(spec, ZH)
        assert body == ZH["body"][task]
        for table in ("complexity", "latency", "cost", "retry", "data"):
            assert ZH[table][getattr(spec.case, table)] not in body
        assert ZH["meta"]["colon"] not in body


def test_parse_round_trips_the_rendered_state() -> None:
    """输入：一份渲染出来的 state。输出：无。

    预期行为：反解出来的每一格就是渲染时写进去的原文——反解**不复用**渲染的计算，
    只按文本读，所以字段漏写、标签写错都能被抓住。
    """
    spec = spec_of(task="reason", complexity="low", latency="relaxed", data="private", phrasing=1)
    request = parse.parse_request(render.build(spec, ZH), parse.label_index(ZH))
    assert request.request_id == spec.request_id
    assert request.complexity == ZH["complexity"]["low"]
    assert request.latency == ZH["latency"]["relaxed"]
    assert request.cost == ZH["cost"]["normal"]
    assert request.retry == ZH["retry"]["first"]
    assert request.data == ZH["data"]["private"]
    assert request.body == ZH["body"]["reason"]


def test_parse_ignores_lines_it_cannot_read() -> None:
    """输入：混进空行与一行没有标签的文本。输出：无。预期行为：忽略它们，其余照读。"""
    state = render.build(spec_of(), ZH) + "\n\n这是一行没人认得的备注。"
    request = parse.parse_request(state, parse.label_index(ZH))
    assert request.complexity == ZH["complexity"]["mid"]


def test_match_label_takes_the_longest_match() -> None:
    """输入：一张有前缀关系的标签表。输出：无。

    预期行为：取最长的那个标签——短标签虽然也是前缀，但后面紧跟的不是分隔符，
    所以两个分支都会被走到。
    """
    labels = {"A": "short", "AB": "long"}
    assert parse._match_label("AB: v", labels) == ("long", "v")
    assert parse._match_label("A：v", labels) == ("short", "v")


def test_match_label_requires_a_separator() -> None:
    """输入：标签后面没有分隔符的行。输出：无。预期行为：不认（返回 None）。"""
    assert parse._match_label("请求号 12345", {"请求号": "request_label"}) is None


def test_parse_reports_the_missing_grids() -> None:
    """输入：少了一格的 state。输出：无。预期行为：报人话并列出缺哪几格。"""
    lines = render.build(spec_of(), ZH).split("\n")
    state = "\n".join(line for line in lines if not line.startswith(ZH["meta"]["cost_label"]))
    with pytest.raises(ValueError, match="画像缺这几格"):
        parse.parse_request(state, parse.label_index(ZH))


def test_parse_rejects_a_non_string_state() -> None:
    """输入：非字符串 state。输出：无。预期行为：报类型错（不是 AttributeError）。"""
    with pytest.raises(TypeError, match="state 必须是字符串"):
        parse.parse_request(123, parse.label_index(ZH))  # type: ignore[arg-type]


def test_label_index_rejects_duplicate_texts() -> None:
    """输入：两格标签文案一样的语言文件。输出：无。预期行为：报错（反解会认错格）。"""
    broken = copy.deepcopy(ZH)
    broken["meta"]["latency_label"] = broken["meta"]["complexity_label"]
    with pytest.raises(ValueError, match="标签文案重复"):
        parse.label_index(broken)


def test_merged_labels_carries_every_language() -> None:
    """输入：无。输出：无。

    预期行为：合并表把各语言的标签取并集——`reparse` 没有语言参数，只能靠它。
    """
    merged = parse.merged_labels()
    for lang in locales.available():
        loc = locales.load(lang)
        for key in parse.META_KEYS:
            assert merged[loc["meta"][key]] == key


def test_merged_labels_rejects_a_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两种语言共用同一格标签。输出：无。预期行为：报错（不静默取其一）。"""
    zh = locales.load("zh")
    en = copy.deepcopy(locales.load("en"))
    en["meta"]["request_label"] = zh["meta"]["request_label"]
    fake = {"zh": zh, "en": en}
    monkeypatch.setattr(parse.locales, "available", lambda: ["zh", "en"])
    monkeypatch.setattr(parse.locales, "load", lambda lang: fake[lang])
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged_labels()


def test_english_state_round_trips_too() -> None:
    """输入：英文渲染出来的 state。输出：无。预期行为：英文标签也认得出来。"""
    spec = spec_of(task="classify", data="private")
    request = parse.parse_request(render.build(spec, EN), parse.label_index(EN))
    assert request.body == EN["body"]["classify"]
    assert request.data == EN["data"]["private"]
