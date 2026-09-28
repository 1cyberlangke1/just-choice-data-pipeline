"""模型路由的文本外壳：提示词、state 往返、候选表与门槛解析、草稿形状校验。"""

from __future__ import annotations

import pytest

from workflows.modelroute import features, locales, render
from workflows.modelroute.features import CATALOG, TagSet

CASE = TagSet("code_gen", "small", "batch", "tight", "all_ok")
EN_CASE = TagSet("vision", "huge", "interactive", "normal", "borealis_down")


def test_prompt_lists_every_feature_line() -> None:
    """输入：一组特征。输出：无。

    预期行为：提示词里每个维度各一行（按 `DIM_ORDER`），且带上语言名；不含任何答案。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(CASE, loc, 0)
    for dim in render.DIM_ORDER:
        assert loc["dims"][dim] in prompt
    assert loc["language_name"] in prompt
    for family in ("route", "blocked", "cost", "slack"):
        assert family not in prompt


def test_prompt_rotates_rules_by_phrasing() -> None:
    """输入：不同问法编号。输出：无。预期行为：规则顺序不同但集合相同。"""
    loc = locales.load("zh")
    first = render.build_prompt(CASE, loc, 0)
    second = render.build_prompt(CASE, loc, 1)
    assert first != second
    assert sorted(first.splitlines()) == sorted(second.splitlines())


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_state_roundtrip(lang: str) -> None:
    """输入：语言。输出：无。

    预期行为：`to_state` 拼出的 state 能被 `request_of` 原样读回来，候选表与门槛行也
    逐项还原；`parse_state` 不认语言也能读（它把各语言的标记都试一遍）。
    """
    loc = locales.load(lang)
    request = "帮我把这段函数改成异步的"
    state = render.to_state({"request": request}, CASE, loc)
    assert render.request_of(state, loc) == request
    assert render.catalog_of(state, loc) == CATALOG
    assert render.needs_of(state, loc) == CASE.needs
    parsed = render.parse_state(state)
    assert parsed.request == request
    assert parsed.catalog == CATALOG
    assert parsed.needs == CASE.needs
    assert not parsed.is_empty


def test_state_carries_the_code_rendered_table() -> None:
    """输入：一组特征。输出：无。

    预期行为：候选表的四行与门槛行由**代码**渲染（不是模型写的）——每一行都长得像
    `- 名字 | 提供方 | 200k | 3 | 3 | 3`，且段落名带括号列名。
    """
    loc = locales.load("zh")
    state = render.to_state({"request": "随便问问"}, CASE, loc)
    meta = loc["meta"]
    assert state.count(str(meta["row_prefix"])) == 5  # 4 行候选 + 1 行门槛
    assert f"{locales.header(loc, meta['catalog_label'])}" in state
    assert f"{locales.header(loc, meta['needs_label'])}" in state
    assert f"- atlas-xl {render.CELL_SEP} atlas {render.CELL_SEP} 200k" in state


def test_request_helpers_return_empty_when_missing() -> None:
    """输入：没有请求行的 state。输出：无。预期行为：返回空串 / 空列表，不抛异常。"""
    loc = locales.load("zh")
    assert render.request_of("候选模型：\n", loc) == ""
    assert render.request_lines(None, loc) == []
    assert render.request_of(123, loc) == ""
    assert render.parse_state(None).is_empty
    assert render.parse_state(None).catalog == ()
    assert render.parse_state(None).needs is None


def test_request_lines_reports_every_line() -> None:
    """输入：请求被写成两行的 state。输出：两行。

    预期行为：卫生判别器靠「不止一行」认出多行请求，所以这里必须全都返回，
    不能像 `request_of` 那样只取第一行。
    """
    loc = locales.load("zh")
    state = render.to_state({"request": "第一行\n第二行"}, CASE, loc)
    assert render.request_lines(state, loc) == ["第一行", "第二行"]
    assert render.request_of(state, loc) == "第一行"


def test_catalog_of_skips_malformed_rows() -> None:
    """输入：混了坏行的候选表。输出：无。

    预期行为：列数不对或数字解不出来的行直接跳过——判别器与审计拿解析结果跟目录表逐行
    比对，缺行会当场暴露，不该在这里抛异常把整批拖垮。
    """
    loc = locales.load("zh")
    state = "\n".join(
        [
            f"{locales.header(loc, loc['meta']['catalog_label'])}",
            "- atlas-xl | atlas | 200k | 3 | 3 | 3",
            "- 半行 | atlas",
            "- 坏数 | atlas | 不是数字k | 3 | 3 | 3",
        ]
    )
    assert render.catalog_of(state, loc) == (CATALOG[0],)


def test_needs_of_returns_none_on_broken_numbers() -> None:
    """输入：门槛行里的数字解不出来。输出：None。预期行为：不抛异常，交给判别器判不通过。"""
    loc = locales.load("zh")
    state = "\n".join(
        [
            f"{locales.header(loc, loc['meta']['needs_label'])}",
            "- 八k | 2 | 3 | 3 | 无",
        ]
    )
    assert render.needs_of(state, loc) is None


def test_needs_line_uses_the_locale_none_mark() -> None:
    """输入：没有提供方挂掉的门槛。输出：无。

    预期行为：占位符随语言变（zh 是「无」、en 是 `none`），两种都能被自己的语言解回来。
    """
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        state = render.to_state({"request": "x"}, CASE, loc)
        assert str(loc["meta"]["none_mark"]) in state
        assert render.needs_of(state, loc) == CASE.needs


def test_needs_line_lists_down_providers() -> None:
    """输入：挂了提供方的特征。输出：无。预期行为：门槛行里列出提供方名，解析能还原。"""
    case = TagSet("chat", "small", "batch", "normal", "borealis_down")
    loc = locales.load("zh")
    state = render.to_state({"request": "x"}, case, loc)
    assert "borealis" in state
    assert render.needs_of(state, loc) == case.needs


def test_parse_state_is_language_agnostic() -> None:
    """输入：英文 state。输出：无。

    预期行为：`parse_state` 把各语言的段落名都试一遍，命中即返回——审计与反解都在
    不知道语言的情况下工作。
    """
    loc = locales.load("en")
    state = render.to_state({"request": "look at this chart"}, EN_CASE, loc)
    parsed = render.parse_state(state)
    assert parsed.request == "look at this chart"
    assert parsed.catalog == CATALOG
    assert parsed.needs == EN_CASE.needs


def test_check_draft_accepts_a_clean_draft() -> None:
    """输入：合规草稿。输出：无。预期行为：不抛异常。"""
    render.check_draft({"request": "帮我把这段函数改成异步的"}, locales.load("zh"))


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"request": "   "}, "空的"),
        ({"request": "第一行\n第二行"}, "多行"),
        ({"request": "```\n帮我把这段函数改成异步的\n```"}, "围栏"),
        ({"request": "请求：帮我把这段函数改成异步的"}, "段落名"),
        ({"request": "帮我把这段函数改成异步的，用 atlas-xl"}, "模型名"),
        ({}, "空的"),
    ],
)
def test_check_draft_rejects_broken_shapes(data: dict[str, str], match: str) -> None:
    """输入：坏草稿。输出：无。预期行为：报人话，点名是哪一类问题。"""
    with pytest.raises(ValueError, match=match):
        render.check_draft(data, locales.load("zh"))


def test_check_draft_rejects_any_catalog_name() -> None:
    """输入：请求里点名了别的候选模型。输出：无。

    预期行为：四行目录里的**每一个**名字都算泄漏——请求里出现模型名等于把答案写进题面。
    """
    loc = locales.load("zh")
    for cand in CATALOG:
        with pytest.raises(ValueError, match="模型名"):
            render.check_draft({"request": f"这个任务交给 {cand.name} 就行"}, loc)


def test_check_draft_reads_the_marker_from_the_locale() -> None:
    """输入：英文文本 + 英文语言字典。输出：无。

    预期行为：行首标记随语言变——英文里 `Request: ...` 也算泄漏，代码里没写死中文。
    """
    loc = locales.load("en")
    with pytest.raises(ValueError, match="段落名"):
        render.check_draft({"request": "Request: summarize the log"}, loc)
    render.check_draft({"request": "summarize the log"}, loc)


def test_label_leak_only_reads_the_line_start() -> None:
    """输入：正文里正常提到段落名。输出：无。预期行为：不算泄漏——只认「段落名 + 冒号」开头。"""
    loc = locales.load("zh")
    assert render.label_leak("这个请求：请把日志压成摘要", loc) == ""
    assert render.label_leak("请求：把日志压成摘要", loc) == "请求"


def test_build_task_carries_tags_and_response_model() -> None:
    """输入：特征。输出：无。预期行为：任务带 tags、用 `RouteDraft` 当响应模型。"""
    loc = locales.load("zh")
    task = render.build_task(CASE, loc, "modelroute-train-0000000-render")
    assert task.task_id == "modelroute-train-0000000-render"
    assert task.spec["tags"] == CASE.as_dict()
    assert task.response_model is render.RouteDraft
    assert task.domain == "model-routing"


def test_route_draft_has_exactly_one_field() -> None:
    """输入：无。输出：无。

    预期行为：草稿模型**只有** `request` 一个字段——候选表与门槛由代码渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪。
    """
    assert set(render.RouteDraft.model_fields) == {"request"}
    assert features.CATALOG  # 目录是代码里的常量，不是草稿字段
