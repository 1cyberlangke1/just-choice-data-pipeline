"""文本外壳测试：提示词该说的话都说了、state 能来回转、写歪的日志在渲染期就被拦下。"""

from __future__ import annotations

import pytest

from jc.workflow import DraftError
from jc.jsontypes import JsonValue

from tests.workflows.causechain.conftest import FRAGMENTS, logs_of, spec_for, state_of
from workflows.causechain import features, locales, render
from workflows.causechain.features import SERVICES, CauseCase

#: 一个普通 case：链式图上根因在末端。
CASE = CauseCase(graph="G1", root="S6", probe="h2")

#: 根因是入口服务的 case：出错集合只有 S1，其余五个服务那行都是「正常」——
#: 贴判据那一条测试拿 S2（一个「正常」服务）当素材。
ENTRY_ROOT = CauseCase(graph="G1", root="S1", probe="h0")


def test_state_roundtrips_edges_and_logs() -> None:
    """输入：一段 state。输出：无。

    预期行为：依赖图那一段反解出来就是这张图的边表、现场日志那一段反解出来就是六行原文、
    素材抬头认得出语言。依赖图借「判据表段落」的解析机制落进 `tags`，所以审计不必再写一份
    边表解析器。
    """
    plan = features.plan_for(CASE, 0)
    logs = logs_of(plan)
    state = state_of(plan, logs)
    parsed = render.parse_state(state)
    assert parsed.found
    assert render.parse_edges(parsed.tags) == features.GRAPHS[plan.graph]
    assert features.graph_of(render.parse_edges(parsed.tags)) == plan.graph
    assert render.parse_logs(parsed.text, locales.load("zh")) == logs
    assert render.detect_locale(state) == "zh"
    assert render.detect_locale(42) is None
    assert render.detect_locale("这里什么抬头都没有") is None


def test_state_keeps_the_plan_order_and_rules() -> None:
    """输入：一段 state。输出：无。

    预期行为：现场日志按计划的时刻顺序排；判读规则表与三种事件形状表都印在 state 里——
    「这次是谁出错、错误从哪儿往上冒」必须自明，考生不该需要外部规则。
    """
    plan = features.plan_for(CASE, 0)
    loc = locales.load("zh")
    state = state_of(plan, logs_of(plan))
    rows = render.parse_rows(render.parse_state(state).text, loc)
    assert [service for _, service, _ in rows] == list(plan.order)
    assert [time for time, _, _ in rows] == [f"T{index + 1}" for index in range(6)]
    for rule in render.rules(loc):
        assert rule in state
    for event in features.EVENTS:
        assert render.event_line(loc, event) in state


def test_parse_edges_skips_lines_that_are_not_edges() -> None:
    """输入：判据表段落里的几种行。输出：无。

    预期行为：只认两端都是合法服务编号的 `上游 > 下游`——规则表里那行恰好含 `>` 的说明
    （「`上游 > 下游` 读作…」）两端不是服务编号，照样跳过。

    `parse_edges` 吃的是 `parse_sections` 已经去掉行前缀之后的字段，所以带前缀的整行
    同样跳过（那是拼错口径，不该悄悄认下来）。
    """
    loc = locales.load("zh")
    lines = (*render.rules(loc), *(render.event_line(loc, e) for e in features.EVENTS))
    assert render.parse_edges(lines) == ()
    assert render.parse_edges((*lines, "S1 > S2", "S2 > S9", "- S3 > S4")) == (("S1", "S2"),)


def test_parse_rows_requires_the_line_prefix_and_a_service() -> None:
    """输入：几行不像现场日志的文本。输出：无。

    预期行为：不以行前缀开头的行、抬头里凑不出「时刻 + 服务」的行都跳过——素材里出现一行
    像判据表的文本不会把现场日志数歪。
    """
    loc = locales.load("zh")
    text = "\n".join(
        [
            "T1 S1：这一行没有行前缀",
            "- T2 S2 少了个分隔符",
            "- T3 S1：正文里带：冒号也不影响",
            "- T4 S9：服务编号不认识",
        ]
    )
    assert render.parse_rows(text, loc) == [("T3", "S1", "正文里带：冒号也不影响")]
    assert render.parse_logs(None, loc) == {}


def test_check_draft_accepts_every_faithful_case() -> None:
    """输入：全空间 100 组 case 的忠实草稿。输出：无。预期行为：一条都不报错。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        render.check_draft(render.ContentDraft(logs=logs_of(plan)), plan, loc)


def test_check_draft_rejects_a_wrong_event_shape() -> None:
    """输入：把第一个服务那行换成另一种形状。输出：无。

    预期行为：抛 `DraftError`，理由里说清「按字面算出来是什么」与「计划要的是什么」——
    事件类型是金标签的一半，写歪了必须在渲染期就拦下，不能等判别器。
    """
    plan = features.plan_for(CASE, 0)
    logs = logs_of(plan)
    head = SERVICES[0]
    other = next(event for event in features.EVENTS if event != plan.events[head])
    logs[head] = f"{head} 节点记录：{FRAGMENTS[other]}"
    with pytest.raises(DraftError, match="按字面算出来"):
        render.check_draft(render.ContentDraft(logs=logs), plan, locales.load("zh"))


def test_check_draft_rejects_a_mixed_shape() -> None:
    """输入：一行里同时踩中两种形状。输出：无。预期行为：按「认不出的形状」拦下。"""
    plan = features.plan_for(CASE, 0)
    logs = logs_of(plan)
    head = SERVICES[0]
    logs[head] = f"{head} 节点：上游连接超时，随后写入失败"
    with pytest.raises(DraftError, match="认不出的形状"):
        render.check_draft(render.ContentDraft(logs=logs), plan, locales.load("zh"))


def test_check_draft_rejects_a_missing_or_extra_service() -> None:
    """输入：少一个服务、多一个服务的草稿。输出：无。预期行为：都抛 `DraftError`。"""
    plan = features.plan_for(CASE, 0)
    logs = logs_of(plan)
    short = dict(logs)
    short.pop(SERVICES[-1])
    with pytest.raises(DraftError, match="服务对不上"):
        render.check_draft(render.ContentDraft(logs=short), plan, locales.load("zh"))
    extra = dict(logs)
    extra["S9"] = "S9 节点：健康检查通过"
    with pytest.raises(DraftError, match="服务对不上"):
        render.check_draft(render.ContentDraft(logs=extra), plan, locales.load("zh"))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("   ", "空的"),
        ("短", "太短"),
        ("会" * (render.MAX_CONTENT_CHARS + 1), "太长"),
        ("```上游连接超时，正在重试第 2 次```", "围栏"),
    ],
)
def test_check_draft_rejects_empty_short_long_and_fenced_lines(text: str, match: str) -> None:
    """输入：空 / 太短 / 太长 / 带围栏的那一行。输出：无。预期行为：四种都当场抛错。"""
    plan = features.plan_for(CASE, 0)
    logs = logs_of(plan)
    logs[SERVICES[0]] = text
    with pytest.raises(DraftError, match=match):
        render.check_draft(render.ContentDraft(logs=logs), plan, locales.load("zh"))


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_check_draft_rejects_a_pasted_rule(lang: str) -> None:
    """输入：把事件判据原文**整段贴**进日志。输出：无。

    预期行为：抛 `DraftError`，理由是贴判据原文——提示词里唯一可抄的具体文字就是它，贴进去
    标签照样算命中（正则只看子串），但那一行读起来不是日志。照抄闸查在长度之前，所以这里
    不必挑「短到能过长度闸」的那一条判据。
    """
    plan = features.plan_for(ENTRY_ROOT, 0)
    loc = locales.load(lang)
    logs = logs_of(plan)
    head = SERVICES[1]
    assert plan.events[head] == features.EVENT_OK
    rule = str(render.event_copy(loc, plan.events[head])["rule"])
    logs[head] = rule
    with pytest.raises(DraftError, match="判据原文"):
        render.check_draft(render.ContentDraft(logs=logs), plan, loc)


def test_pasted_rules_only_flags_verbatim_text() -> None:
    """输入：整段贴 / 改了一个词的两种文本。输出：无。

    预期行为：只认**逐字相同**的整段判据原文——改一个词就不再算，否则正常产物会被误伤。
    """
    loc = locales.load("zh")
    rule = str(render.event_copy(loc, features.EVENT_OK)["rule"])
    assert render.pasted_rules(rule, loc) == [rule]
    assert render.pasted_rules(rule.replace("健康", "康健"), loc) == []
    assert render.pasted_rules(FRAGMENTS[features.EVENT_OK], loc) == []


def test_build_task_carries_the_injected_features() -> None:
    """输入：真值规格。输出：无。

    预期行为：任务 id 可重算、`spec` 里带上注入特征与现场计划（假后端与真后端都能照着它答）、
    响应模型是草稿模型。
    """
    spec = spec_for(CASE)
    task = render.build_task(spec, locales.load("zh"))
    assert task.task_id == "causechain-train-0000001-render"
    assert task.domain == "causechain"
    assert task.response_model is render.ContentDraft
    assert task.spec["case"] == {"graph": "G1", "root": "S6", "probe": "h2"}
    assert task.spec["events"] == dict(spec.plan.events)
    assert task.spec["order"] == list(spec.plan.order)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_build_prompt_names_every_service_and_forbids_the_other_shapes(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：提示词里有这张图的每一条边、有六个服务各自点名的那一行、有把语言钉死的约束，
    而且**不给成品日志样例**——给了就会被整批抄走（形状只以判据原文的形式出现）。
    """
    loc = locales.load(lang)
    plan = features.plan_for(CASE, 0)
    prompt = render.build_prompt(loc, plan)
    assert str(loc["language_name"]) in prompt
    for upstream, downstream in features.GRAPHS[plan.graph]:
        assert render.graph_line(upstream, downstream) in prompt
    for service in SERVICES:
        assert service in prompt
        assert str(render.event_copy(loc, plan.events[service])["rule"]) in prompt
    for event in features.EVENTS:
        assert FRAGMENTS[event] not in prompt


def test_to_state_needs_every_service() -> None:
    """输入：少一个服务的草稿。输出：无。预期行为：拼装时当场 KeyError，不静默少一行。"""
    plan = features.plan_for(CASE, 0)
    logs = logs_of(plan)
    logs.pop(SERVICES[0])
    with pytest.raises(KeyError):
        render.to_state(render.ContentDraft(logs=logs), plan, locales.load("zh"))


def test_event_copy_reports_missing_fields() -> None:
    """输入：缺字段的语言字典。输出：无。预期行为：抛 ValueError，点名缺了哪个字段。"""
    with pytest.raises(ValueError, match="缺事件 ok"):
        render.event_copy({}, features.EVENT_OK)
    broken: dict[str, JsonValue] = {"event": {features.EVENT_OK: {"name": "正常"}}}
    with pytest.raises(ValueError, match="缺字段 rule"):
        render.event_copy(broken, features.EVENT_OK)


def test_role_and_rules_report_missing_content() -> None:
    """输入：缺内容 / 缺规则的语言字典。输出：无。预期行为：都抛 ValueError。"""
    with pytest.raises(ValueError, match="缺角色"):
        render.role_copy({"role": {}}, features.ROLE_ENTRY)
    with pytest.raises(ValueError, match="rules"):
        render.rules({})
