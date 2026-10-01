"""hirecheck 判别器：兑现检查、骨架检查、反解检查各自拦得住什么。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from jc.backend.types import AllProvidersDown, Draft, ProviderUnavailable
from jc.jsontypes import JsonValue
from tests.workflows.hirecheck.conftest import FakeBackend, packet_of, quote_for
from workflows.hirecheck import features, judges, locales, render
from workflows.hirecheck.features import HireCheckCase
from workflows.hirecheck.spec import HireCheckSpec

LANGS: tuple[str, ...] = ("zh", "en")


def case_for(tags: tuple[str, ...]) -> HireCheckCase:
    """输入：该命中的判据。输出：一个挂在 `referral` 渠道上的 case。"""
    return HireCheckCase(flags="+".join(tags), channel=features.CHANNEL_REFERRAL)


def spec_for(tags: tuple[str, ...], phrasing: int = 0) -> HireCheckSpec:
    """输入：该命中的判据、问法编号。输出：真值规格。"""
    case = case_for(tags)
    return HireCheckSpec(
        case=case,
        phrasing=phrasing,
        role=0,
        years=5,
        name=0,
        split="train",
        seq=0,
        plan=features.plan_for(case, phrasing),
    )


def state_of(tags: tuple[str, ...], lang: str = "zh") -> str:
    """输入：该命中的判据、语言。输出：渲染好的 state。"""
    return render.to_state(
        render.ContentDraft(packet=packet_of(tags, lang)), locales.load(lang)
    )


def note_swap(text: str, value: str) -> str:
    """输入：一屏材料 / 一段 state、要换上的 `note:` 那一整行。输出：换完之后的多行文本。

    预期行为：只换 `note:` 那一行——「把备注清空」这类样本要真的清空，不能靠替换前几个字符。
    """
    return "\n".join(
        value if line.startswith("note:") else line for line in text.splitlines()
    )


@dataclass
class StubBackend:
    """只会照着给定数据回话的后端：验反解的**格式**分支用。"""

    data: dict[str, JsonValue] = field(default_factory=dict)
    error: Exception | None = None

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：草稿。预期行为：有 `error` 就抛，否则回给定的数据。"""
        if self.error is not None:
            raise self.error
        return Draft(
            data=self.data,
            raw_text="stub",
            backend="stub",
            provider="stub",
            model="stub",
            mode="stub",
        )


@pytest.mark.parametrize("lang", LANGS)
def test_hygiene_passes_a_faithful_sample(lang: str) -> None:
    """输入：一屏完全兑现计划的材料。输出：无。预期行为：通过，理由说清它兑现了什么。"""
    verdict = judges.HygieneJudge(lang).judge({"state": state_of(("H1", "H2"), lang)}, spec_for(("H1", "H2")))
    assert verdict.ok
    assert not verdict.skipped
    assert "兑现" in verdict.reason


def test_hygiene_skips_without_spec_or_state() -> None:
    """输入：没有规格 / 没有 state / 串了别的 workflow 的规格。输出：无。

    预期行为：三种都记「没判成」（`skipped=True`）——没判成不是不通过，端点与接线问题不该写成数据问题。
    """
    judge = judges.HygieneJudge("zh")
    assert judge.judge({"state": state_of(())}).skipped
    assert judge.judge({}, spec_for(())).skipped
    assert judge.judge({"state": state_of(())}, object()).skipped
    assert judge.judge({"state": 42}, spec_for(())).skipped


@pytest.mark.parametrize(
    ("packet", "needle"),
    [
        (packet_of(()).replace("permit: citizen\n", ""), "缺这几行"),
        (note_swap(packet_of(()), "note:"), "空的"),
        (note_swap(packet_of(()), "note: 这个人年限不足，先放着"), "初筛结论"),
        (note_swap(packet_of(()), "note: 他搭过 kubernetes 平台"), "硬性技能"),
    ],
)
def test_hygiene_catches_broken_material(packet: str, needle: str) -> None:
    """输入：四类写歪的材料。输出：无。预期行为：每一类都被拦下，理由点明是哪一类。"""
    verdict = judges.HygieneJudge("zh").judge(
        {"state": render.to_state(render.ContentDraft(packet=packet), locales.load("zh"))},
        spec_for(()),
    )
    assert not verdict.ok
    assert needle in verdict.reason


def test_hygiene_catches_a_mismatched_hit_set() -> None:
    """输入：材料只兑现了计划的一部分。输出：无。预期行为：拦下并把两个组合都写出来。"""
    verdict = judges.HygieneJudge("zh").judge({"state": state_of(())}, spec_for(("H1",)))
    assert not verdict.ok
    assert "命中组合" in verdict.reason


def test_hygiene_catches_pasted_rules_and_leftovers() -> None:
    """输入：把判据原文贴进材料 / 整段 state 里留了围栏。输出：无。预期行为：两种都拦下。"""
    rule = str(render.fact_copy(locales.load("zh"), "H1")["rule"])
    pasted = render.to_state(
        render.ContentDraft(packet=packet_of(()).replace("note: 现在", f"note: {rule}")),
        locales.load("zh"),
    )
    verdict = judges.HygieneJudge("zh").judge({"state": pasted}, spec_for(()))
    assert not verdict.ok
    assert "判据原文" in verdict.reason
    leftover = judges.HygieneJudge("zh").judge({"state": state_of(()) + "\n```"}, spec_for(()))
    assert not leftover.ok
    assert "格式残留" in leftover.reason


def test_hygiene_reports_an_unparseable_content_block() -> None:
    """输入：state 里没有材料抬头。输出：无。预期行为：拦下，理由是「那一块解析不出来」。"""
    verdict = judges.HygieneJudge("zh").judge({"state": "只有一行"}, spec_for(()))
    assert not verdict.ok
    assert "解析不出来" in verdict.reason


@pytest.mark.parametrize("lang", LANGS)
def test_frame_passes_a_canonical_state(lang: str) -> None:
    """输入：规范形态的 state。输出：无。预期行为：通过。"""
    verdict = judges.FrameJudge(lang).judge({"state": state_of(("H3",), lang)}, spec_for(("H3",)))
    assert verdict.ok


def test_frame_skips_without_spec_or_state() -> None:
    """输入：没有规格 / 没有 state。输出：无。预期行为：记「没判成」。"""
    judge = judges.FrameJudge("zh")
    assert judge.judge({"state": state_of(())}).skipped
    assert judge.judge({}, spec_for(())).skipped


def test_frame_catches_a_broken_skeleton() -> None:
    """输入：判据表缺一行 / 权重行被改 / 档位表被改。输出：无。

    预期行为：三种都被拦下——骨架对不上就说明渲染挂错了 case，或者语言文件缺了等级，标签会跟着一起错。
    """
    loc = locales.load("zh")
    good = state_of(())
    cases = [
        (good.replace(f"- {render.fact_line(loc, 'H3')}\n", ""), "判据表"),
        (good.replace(f"- {render.weight_line()}", "- H1=1 H2=2 H3=3 H4=4"), "定级规则"),
        (
            good.replace(render.action_line(loc, 5), render.action_line(loc, 4)),
            "处置档位表",
        ),
        (good.replace("candidate: 林素\n", ""), "缺这几行"),
    ]
    for state, needle in cases:
        verdict = judges.FrameJudge("zh").judge({"state": state}, spec_for(()))
        assert not verdict.ok, state
        assert needle in verdict.reason


def test_frame_catches_an_empty_note() -> None:
    """输入：`note:` 那一行是空的。输出：无。预期行为：拦下并点名那一行。"""
    loc = locales.load("zh")
    state = note_swap(state_of(()), "note:")
    verdict = judges.FrameJudge("zh").judge({"state": state}, spec_for(()))
    assert not verdict.ok
    assert "note" in verdict.reason
    assert loc["language_name"]


@pytest.mark.parametrize("lang", LANGS)
def test_trace_passes_when_the_reader_agrees(lang: str) -> None:
    """输入：反解读对了命中组合与等级。输出：无。预期行为：通过。"""
    spec = spec_for(("H1", "H2"))
    verdict = judges.TraceJudge(FakeBackend(locale=lang), lang).judge(
        {"state": state_of(("H1", "H2"), lang)}, spec
    )
    assert verdict.ok
    assert verdict.raw["traced_level"] == spec.plan.level


def test_trace_catches_a_missing_hit() -> None:
    """输入：反解漏报最后一条命中。输出：无。预期行为：拦下，并说清丢了几条误报。"""
    spec = spec_for(("H1", "H3"))
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(
        {"state": state_of(("H1", "H3"))}, spec
    )
    assert not verdict.ok
    assert "反解读成" in verdict.reason


def test_trace_catches_a_misread_level() -> None:
    """输入：命中都对、等级说错。输出：无。

    预期行为：拦下——`grade` 是这一族的中心题型，读者读不出等级，那一道题就等于没有答案。
    """
    spec = spec_for(("H1",))
    verdict = judges.TraceJudge(FakeBackend(trace_level_lies=True), "zh").judge(
        {"state": state_of(("H1",))}, spec
    )
    assert not verdict.ok
    assert "等级" in verdict.reason


def test_trace_drops_quotes_that_do_not_hold() -> None:
    """输入：反解报了一条抄不出原文的命中。输出：无。

    预期行为：那条按**误报**丢掉，理由里说明丢了什么——判据是机械可查的，抄不出证据就不算数。
    """
    spec = spec_for(("H1",))
    stub = StubBackend(data={"tags": [{"tag": "H1", "quote": "根本没出现过的一段"}], "level": 2})
    verdict = judges.TraceJudge(stub, "zh").judge({"state": state_of(("H1",))}, spec)
    assert not verdict.ok
    assert "抄不出原文" in verdict.reason


def test_trace_accepts_a_hit_whose_quote_is_real() -> None:
    """输入：反解报的命中带着真引文。输出：无。预期行为：通过，理由里附上引文核查的结论。"""
    spec = spec_for(("H2",))
    stub = StubBackend(
        data={"tags": [{"tag": "H2", "quote": quote_for("H2")}], "level": spec.plan.level}
    )
    verdict = judges.TraceJudge(stub, "zh").judge({"state": state_of(("H2",))}, spec)
    assert verdict.ok


@pytest.mark.parametrize(
    "data",
    [
        {"level": 2},
        {"tags": [], "level": "2"},
        {"tags": [], "level": True},
        {"tags": [], "level": 9},
        {"tags": [1], "level": 2},
        {"tags": [{"tag": "H9", "quote": "x"}], "level": 2},
    ],
)
def test_trace_treats_malformed_drafts_as_not_judged(data: dict[str, JsonValue]) -> None:
    """输入：缺字段 / 类型不对 / 越界 / 不认识编号的反解草稿。输出：无。

    预期行为：一律记「没判成」（`skipped=True`）——那是反解格式没对齐，不是数据本身错。
    """
    verdict = judges.TraceJudge(StubBackend(data=data), "zh").judge(
        {"state": state_of(())}, spec_for(())
    )
    assert verdict.skipped


def test_trace_skips_without_spec() -> None:
    """输入：没有规格。输出：无。预期行为：记「没判成」。"""
    assert judges.TraceJudge(FakeBackend(), "zh").judge({"state": state_of(())}).skipped


def test_trace_treats_an_endpoint_failure_as_not_judged() -> None:
    """输入：反解时端点故障。输出：无。预期行为：记「没判成」，不许写成数据质量问题。"""
    stub = StubBackend(error=ProviderUnavailable("端点挂了"))
    verdict = judges.TraceJudge(stub, "zh").judge({"state": state_of(())}, spec_for(()))
    assert verdict.skipped
    assert "端点故障" in verdict.reason


def test_trace_reraises_a_fatal_backend_failure() -> None:
    """输入：反解时全部 provider 都挂了。输出：无。

    预期行为：异常往上抛——整批该停下（PLAN 3.1.4），而不是把「全军覆没」记成「这几条数据不行」。
    """
    stub = StubBackend(error=AllProvidersDown("全都挂了"))
    with pytest.raises(AllProvidersDown):
        judges.TraceJudge(stub, "zh").judge({"state": state_of(())}, spec_for(()))


def test_quote_holds_needs_three_things_at_once() -> None:
    """输入：几种引文。输出：无。

    预期行为：引文**原样**在材料里、引文**自己**满足判据、**整屏材料**也满足判据——三样缺一不可。
    第三条是范式 §9.47 的落地：只把形状塞进引文而整段不算命中，是误报。
    """
    body = packet_of(("H1",))
    assert judges.quote_holds("H1", quote_for("H1"), body)
    assert judges.quote_holds("H1", json.dumps([quote_for("H1")]), body)  # 引文被装进 JSON 数组字符串（范式 §9.88）
    assert not judges.quote_holds("H1", "years: 1", body.replace("years: 1", "years: 4"))
    assert not judges.quote_holds("H1", "candidate: 林素", body)
    assert not judges.quote_holds("H9", quote_for("H1"), body)
    assert not judges.quote_holds("H1", "years: 1", "years: 4")


def test_quote_holds_ignores_whitespace_differences() -> None:
    """输入：引文与原文只差空白。输出：无。预期行为：算抄对（空白差异不算抄错）。"""
    assert judges.quote_holds("H3", "permit:   none", packet_of(("H3",)))
