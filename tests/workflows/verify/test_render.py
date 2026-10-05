"""完成声明核验的文本外壳：提示词、state 往返、事实行与门槛行解析、草稿形状校验。"""

from __future__ import annotations

import pytest

from workflows.verify import features, locales, render
from workflows.verify.features import TagSet

CASE = TagSet("bug_fixed", "hedged", "partial_pass", "covered", "uncommitted")
EN_CASE = TagSet("api_added", "absolute", "all_pass", "covered", "committed")
CLEAN = "把订单列表偶发的空指针修掉了"


def test_prompt_lists_only_the_two_text_dimensions() -> None:
    """输入：一个 case。输出：无。

    预期行为：提示词里只有「声称做完的事」与「口气」两行特征 + 规则；**证据那三维的
    取值一个都不许出现**——出现了模型就会把证据也写进声明，而证据那一行是代码写的，
    两边一冲突这道题该读哪一行就有歧义了。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(CASE, loc, 0)
    bullets = [line for line in prompt.splitlines() if line.startswith("- ")]
    assert len(bullets) == len(render.DIM_ORDER) + len(loc["prompt"]["rules"])
    assert loc["dims"]["claim_kind"] in bullets[0]
    assert loc["dims"]["tone"] in bullets[1]
    assert loc["language_name"] in prompt
    for token in (*features.CHECKS, *features.COVERAGES, *features.ARTIFACTS):
        assert token not in prompt, token


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_state_roundtrip(lang: str) -> None:
    """输入：语言。输出：无。

    预期行为：`to_state` 拼出的 state 能被 `claim_of` 原样读回来，事实行与门槛行也逐项
    还原；`parse_state` 不认语言也能读（它把各语言的标记都试一遍）。
    """
    loc = locales.load(lang)
    tags = CASE if lang == "zh" else EN_CASE
    state = render.to_state({"claim": CLEAN}, tags, loc)
    assert render.claim_of(state, loc) == CLEAN
    assert render.facts_of(state, loc) == tags.facts
    assert render.gates_of(state, loc) == features.gates_from(tags.facts)
    parsed = render.parse_state(state)
    assert parsed.claim == CLEAN
    assert parsed.facts == tags.facts
    assert not parsed.is_empty


def test_state_carries_the_code_rendered_rows() -> None:
    """输入：一个 case。输出：无。

    预期行为：事实行与门槛行由**代码**渲染（不是模型写的）——事实行是三格 ASCII 词元，
    门槛行是三条门槛的 yes / no；整段只有这两行数据行。
    """
    loc = locales.load("zh")
    state = render.to_state({"claim": "随便写了一句"}, CASE, loc)
    meta = loc["meta"]
    assert state.count(str(meta["row_prefix"])) == 2
    assert f"- partial_pass {render.CELL_SEP} covered {render.CELL_SEP} uncommitted" in state
    assert f"- {meta['no']} {render.CELL_SEP} {meta['yes']} {render.CELL_SEP} {meta['no']}" in state
    assert locales.header(loc, meta["facts_label"]) in state
    assert locales.header(loc, meta["gates_label"]) in state
    assert str(meta["facts_columns"]) in state
    assert str(meta["gates_columns"]) in state


def test_facts_of_reads_the_first_well_formed_row() -> None:
    """输入：混了坏行的事实段。输出：无。

    预期行为：列数不对的行直接跳过（判别器与审计拿解析结果跟注入特征逐格比对，
    缺行会当场暴露），不在这里抛异常把整批拖垮。
    """
    loc = locales.load("zh")
    header = locales.header(loc, loc["meta"]["facts_label"])
    state = f"{header}\n- 半行 {render.CELL_SEP} covered\n- partial_pass | covered | uncommitted"
    assert render.facts_of(state, loc) == CASE.facts


def test_facts_of_refuses_unknown_tokens() -> None:
    """输入：三格里有认不出来的词元。输出：None。预期行为：不猜，交给判别器判不通过。"""
    loc = locales.load("zh")
    header = locales.header(loc, loc["meta"]["facts_label"])
    assert render.facts_of(f"{header}\n- all_pass | covered | 不知道", loc) is None
    assert render.facts_of(f"{header}\n- all_pass | 差不多 | committed", loc) is None


def test_gates_of_reads_the_locale_marks() -> None:
    """输入：两种语言各一条 state。输出：无。

    预期行为：yes / no 记号随语言走（代码里没写死），两种都能被自己的语言解回来。
    """
    for lang, tags in (("zh", CASE), ("en", EN_CASE)):
        loc = locales.load(lang)
        state = render.to_state({"claim": CLEAN}, tags, loc)
        assert render.gates_of(state, loc) == features.gates_from(tags.facts)


def test_gates_of_refuses_broken_cells() -> None:
    """输入：格数不对 / 记号认不出来的门槛行。输出：None。预期行为：不猜。"""
    loc = locales.load("zh")
    header = locales.header(loc, loc["meta"]["gates_label"])
    assert render.gates_of(f"{header}\n- no | yes", loc) is None
    assert render.gates_of(f"{header}\n- no | maybe | yes", loc) is None


def test_helpers_return_empty_when_missing() -> None:
    """输入：没有声明行的 state / 非文本。输出：无。预期行为：返回空值，不抛异常。"""
    loc = locales.load("zh")
    assert render.claim_of(f"{locales.header(loc, '自报事实')}\n", loc) == ""
    assert render.claim_lines(None, loc) == []
    assert render.claim_of(123, loc) == ""
    assert render.parse_state(None).is_empty
    assert render.parse_state(None).facts is None
    assert render.parse_state(None).gates is None


def test_claim_lines_reports_every_line() -> None:
    """输入：声明被写成两行的 state。输出：两行。

    预期行为：卫生判别器靠「不止一行」认出多行声明，所以这里必须全都返回，
    不能像 `claim_of` 那样只取第一行。
    """
    loc = locales.load("zh")
    state = render.to_state({"claim": "第一行\n第二行"}, CASE, loc)
    assert render.claim_lines(state, loc) == ["第一行", "第二行"]
    assert render.claim_of(state, loc) == "第一行"


def test_parse_state_is_language_agnostic() -> None:
    """输入：英文 state。输出：无。

    预期行为：`parse_state` 把各语言的段落名都试一遍，命中即返回——审计与反解都在
    不知道语言的情况下工作。
    """
    loc = locales.load("en")
    state = render.to_state({"claim": "fixed the flaky retry"}, EN_CASE, loc)
    parsed = render.parse_state(state)
    assert parsed.claim == "fixed the flaky retry"
    assert parsed.facts == EN_CASE.facts
    assert parsed.gates == features.gates_from(EN_CASE.facts)


def test_check_draft_accepts_a_clean_draft() -> None:
    """输入：合规草稿。输出：无。预期行为：不抛异常。"""
    render.check_draft({"claim": CLEAN}, locales.load("zh"))


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"claim": "   "}, "空的"),
        ({"claim": "第一行\n第二行"}, "多行"),
        ({"claim": "```\n把订单列表的空指针修掉了\n```"}, "围栏"),
        ({"claim": "声明：把订单列表的空指针修掉了"}, "段落名"),
        ({"claim": "把订单列表的空指针修掉了，测试全过了"}, "证据"),
        ({}, "空的"),
    ],
)
def test_check_draft_rejects_broken_shapes(data: dict[str, str], match: str) -> None:
    """输入：坏草稿。输出：无。预期行为：报人话，点名是哪一类问题。"""
    with pytest.raises(ValueError, match=match):
        render.check_draft(data, locales.load("zh"))


def test_check_draft_rejects_every_banned_phrase() -> None:
    """输入：locale 里点名的每一条证据话术。输出：无。

    预期行为：一条都不许漏——声明里写了证据，就会跟代码渲染的事实行打架，
    这道题该读哪一行当场变成歧义。
    """
    loc = locales.load("zh")
    for phrase in loc["banned_claim_phrases"]:
        with pytest.raises(ValueError, match="证据"):
            render.check_draft({"claim": f"把空指针修掉了，{phrase}"}, loc)


def test_check_draft_reads_the_marker_from_the_locale() -> None:
    """输入：英文文本 + 英文语言字典。输出：无。

    预期行为：行首标记与证据话术都随语言变——英文里 `Claim: ...` 也算泄漏，
    代码里没写死中文。
    """
    loc = locales.load("en")
    with pytest.raises(ValueError, match="段落名"):
        render.check_draft({"claim": "Claim: fixed the retry"}, loc)
    with pytest.raises(ValueError, match="证据"):
        render.check_draft({"claim": "fixed the retry, all tests pass"}, loc)
    render.check_draft({"claim": "fixed the retry"}, loc)


def test_evidence_echo_ignores_ordinary_wording() -> None:
    """输入：正文里正常提到这些词。输出：无。

    预期行为：只认 locale 点名的高精度话术——「修掉了」「补齐了」这类正常措辞不算，
    否则会白拒一条本来能用的数据（重渲染要再花一次配额）。
    """
    loc = locales.load("zh")
    assert render.evidence_echo("把订单列表的空指针修掉了", loc) == []
    assert render.evidence_echo("把订单模块拆成了三个小文件", loc) == []
    assert render.evidence_echo("把订单列表的空指针修掉了，测试全过了", loc) == ["测试全过"]


def test_build_task_carries_tags_and_response_model() -> None:
    """输入：特征。输出：无。预期行为：任务带 tags、用 `VerifyDraft` 当响应模型。"""
    loc = locales.load("zh")
    task = render.build_task(CASE, loc, "verify-train-0000000-render")
    assert task.task_id == "verify-train-0000000-render"
    assert task.spec["tags"] == CASE.as_dict()
    assert task.response_model is render.VerifyDraft
    assert task.domain == "claim-verification"


def test_verify_draft_has_exactly_one_field() -> None:
    """输入：无。输出：无。

    预期行为：草稿模型**只有** `claim` 一个字段——事实与门槛由代码渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪。
    """
    assert set(render.VerifyDraft.model_fields) == {"claim"}
    assert render.DIM_ORDER == ("claim_kind", "tone")
