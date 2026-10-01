"""判别器测试：两个零 token 判别器能抓到什么、反解怎么记端点故障。"""

from __future__ import annotations

from jc.backend.types import AllProvidersDown, BackendError, Draft
from jc.jsontypes import JsonValue

from tests.workflows.causechain.conftest import (
    FRAGMENTS,
    FakeBackend,
    faithful,
    logs_of,
    spec_for,
    state_of,
)
from workflows.causechain import features, judges, locales, render
from workflows.causechain.features import SERVICES, CauseCase

#: 一个普通 case：链式图上根因在末端。
CASE = CauseCase(graph="G1", root="S6", probe="h2")

#: 根因是入口服务的 case：出错集合只有 S1。
ENTRY_ROOT = CauseCase(graph="G1", root="S1", probe="h0")


def _draft(data: dict[str, JsonValue]) -> Draft:
    """输入：解析后的数据。输出：`Draft`。预期行为：记账字段填假值，不进产物。"""
    return Draft(
        data=data,
        raw_text="fake",
        backend="fake",
        provider="fake",
        model="fake",
        mode="fake",
    )


class _JunkBackend:
    """只会答一个非字符串根因的假后端（验「草稿格式漂移」这条路）。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：`root` 是整数的草稿。预期行为：不联网。"""
        del task
        return _draft({"root": 3})


def test_hygiene_accepts_a_faithful_state() -> None:
    """输入：完全兑现计划的样本。输出：无。预期行为：通过，且不是「没判成」。"""
    spec, sample = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert "兑现" in verdict.reason


def test_hygiene_rejects_a_flipped_event() -> None:
    """输入：把一个服务那行换成另一种形状。输出：无。预期行为：不通过，理由里点出那个服务。"""
    spec, sample = faithful(CASE)
    plan = spec.plan
    logs = logs_of(plan)
    head = SERVICES[0]
    other = next(event for event in features.EVENTS if event != plan.events[head])
    logs[head] = f"{head} 节点记录：{FRAGMENTS[other]}"
    verdict = judges.HygieneJudge("zh").judge({"state": state_of(plan, logs)}, spec)
    assert not verdict.ok and not verdict.skipped
    assert head in verdict.reason


def test_hygiene_rejects_a_missing_service_and_a_pasted_rule() -> None:
    """输入：少一个服务、以及把判据原文贴进日志的样本。输出：无。预期行为：两种都不通过。"""
    plan = features.plan_for(CASE, 0)
    logs = logs_of(plan)
    loc = locales.load("zh")
    sep = str(loc["state"]["tag_sep"])
    # 少一个服务只能从**拼好的 state 里删掉那一行**——`to_state` 自己要求六个服务齐全。
    drop = f" {SERVICES[-1]}{sep}"
    short = "\n".join(
        line
        for line in state_of(plan, logs).splitlines()
        if not (line.startswith(render.LINE_PREFIX) and drop in line)
    )
    verdict = judges.HygieneJudge("zh").judge({"state": short}, spec_for(CASE))
    assert not verdict.ok
    assert "缺" in verdict.reason
    pasted = dict(logs)
    pasted[SERVICES[1]] = str(render.event_copy(loc, plan.events[SERVICES[1]])["rule"])
    verdict = judges.HygieneJudge("zh").judge(
        {"state": state_of(plan, pasted)}, spec_for(CASE)
    )
    assert not verdict.ok
    assert "判据原文" in verdict.reason


def test_hygiene_rejects_a_state_without_a_content_block() -> None:
    """输入：一段没有现场日志抬头的 state。输出：无。预期行为：不通过，理由是认不出来。"""
    verdict = judges.HygieneJudge("zh").judge({"state": "这里什么抬头都没有"}, spec_for(CASE))
    assert not verdict.ok and not verdict.skipped
    assert "现场日志" in verdict.reason


def test_hygiene_rejects_format_leftovers() -> None:
    """输入：state 里多出一行围栏 / 混进 HTML 换行标签。输出：无。预期行为：两种都不通过。"""
    spec, sample = faithful(CASE)
    judge = judges.HygieneJudge("zh")
    fenced = judge.judge({"state": f"{sample['state']}\n```"}, spec)
    assert not fenced.ok and "残留" in fenced.reason
    tagged = judge.judge({"state": f"{sample['state']}<br>"}, spec)
    assert not tagged.ok and "HTML" in tagged.reason


def test_hygiene_skips_when_there_is_no_spec_or_state() -> None:
    """输入：没有规格 / 没有 state。输出：无。

    预期行为：都记「没判成」（`ok=True, skipped=True`）——那是接线问题，不该写成数据质量问题。
    """
    spec, sample = faithful(CASE)
    judge = judges.HygieneJudge("zh")
    assert judge.judge(sample, None).skipped
    assert judge.judge({"state": 42}, spec).skipped


def test_frame_accepts_faithful_and_rejects_a_broken_graph() -> None:
    """输入：忠实 state 与缺了一条边的 state。输出：无。

    预期行为：边表对不上就报出来——那是构造 bug，标签会跟着一起错。
    """
    judge = judges.FrameJudge("zh")
    spec, sample = faithful(CASE)
    assert judge.judge(sample, spec).ok
    upstream, downstream = features.GRAPHS[spec.plan.graph][0]
    want = f"{render.LINE_PREFIX}{render.graph_line(upstream, downstream)}"
    broken = "\n".join(
        line for line in str(sample["state"]).splitlines() if line.strip() != want
    )
    verdict = judge.judge({"state": broken}, spec)
    assert not verdict.ok
    assert "依赖图" in verdict.reason


def test_frame_rejects_a_wrong_log_order() -> None:
    """输入：把现场日志那一段的头两行对调。输出：无。

    预期行为：不通过，理由是服务顺序与计划不一致——时刻是现场先后，顺序本身也是结构事实。
    """
    spec, sample = faithful(CASE)
    lines = str(sample["state"]).splitlines()
    mark = lines.index(str(locales.load("zh")["state"]["content_mark"]))
    first, second = lines[mark + 1], lines[mark + 2]
    lines[mark + 1], lines[mark + 2] = second, first
    verdict = judges.FrameJudge("zh").judge({"state": "\n".join(lines)}, spec)
    assert not verdict.ok
    assert "顺序" in verdict.reason


def test_frame_skips_without_a_spec() -> None:
    """输入：没有规格。输出：无。预期行为：记「没判成」，不硬判。"""
    _, sample = faithful(CASE)
    assert judges.FrameJudge("zh").judge(sample, None).skipped


def test_trace_agrees_with_the_plan() -> None:
    """输入：诚实的反解。输出：无。预期行为：通过，raw 里留下反解出来的根因。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw == {"traced_root": CASE.root}


def test_trace_catches_a_lie() -> None:
    """输入：反解换了一个服务。输出：无。预期行为：不通过，理由里计划的与读出来的都列出来。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert CASE.root in verdict.reason
    assert verdict.raw["traced_root"] != CASE.root


def test_trace_records_endpoint_faults_and_junk_ids_as_skipped() -> None:
    """输入：端点抛错、以及反解给一个不存在的编号。输出：无。

    预期行为：两种都记「没判成」——端点抖动或格式漂移不是数据问题，写成「不通过」会把噪声
    记成结论。
    """
    spec, sample = faithful(CASE)
    failed = judges.TraceJudge(FakeBackend(trace_raises=BackendError("端点炸了")), "zh").judge(
        sample, spec
    )
    assert failed.ok and failed.skipped
    assert failed.raw == {"error": "BackendError"}
    junk = judges.TraceJudge(FakeBackend(trace_junk=True), "zh").judge(sample, spec)
    assert junk.ok and junk.skipped
    malformed = judges.TraceJudge(_JunkBackend(), "zh").judge(sample, spec)
    assert malformed.ok and malformed.skipped


def test_trace_reraises_when_every_provider_is_down() -> None:
    """输入：反解时所有 provider 都挂了。输出：无。

    预期行为：当场往上抛——这是整批停下的唯一条件，吞掉它就等于拿一个「全被拒」的假数据集。
    """
    spec, sample = faithful(CASE)
    judge = judges.TraceJudge(FakeBackend(trace_raises=AllProvidersDown("全挂了")), "zh")
    with __import__("pytest").raises(AllProvidersDown):
        judge.judge(sample, spec)


def test_trace_skips_without_a_spec() -> None:
    """输入：没有规格。输出：无。预期行为：记「没判成」。"""
    _, sample = faithful(CASE)
    assert judges.TraceJudge(FakeBackend(), "zh").judge(sample, None).skipped


def test_entry_root_case_is_judged_the_same_way() -> None:
    """输入：根因是入口服务的样本。输出：无。预期行为：三个判别器照样给通过。"""
    spec, sample = faithful(ENTRY_ROOT)
    assert judges.HygieneJudge("zh").judge(sample, spec).ok
    assert judges.FrameJudge("zh").judge(sample, spec).ok
    trace = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert trace.ok and trace.raw == {"traced_root": ENTRY_ROOT.root}
