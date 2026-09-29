"""通用层测试：判别器拒掉就重渲染（PLAN 6.1 的第二道闸）。

这里用「假 workflow + 假判别器」只验接线本身：**判别器拒掉**重渲染；生成期里
**自己声明了 `retryable_for_case` 的异常**（`DraftError`：后端没照契约写）也重渲染；
其余生成期失败（`KeyError` 之类）不重渲染——那才是生成器 bug，重渲染只会把 bug 藏起来；
渲染次数要进进度日志与统计，续跑时能原样读回来。**不发任何网络请求**。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import json
import random
from pathlib import Path


import pytest

from jc import pipeline
from jc.backend.types import GenerationBackend
from jc.judge import Verdict
from jc.workflow import DraftError, LabeledQuestion, SplitSpec


class FlakyWorkflow:
    """假 workflow：判别器按渲染次数决定放不放行。

    输入：`reject_until` = 渲染次数没超过它就一律拒（0 = 永不拒）；`questions` = 出几道题
    （0 用来制造生成期失败）；`draft_failures` = 前几次渲染直接抛异常（用来验生成期异常的
    重渲染判据），`error` = 抛哪一种（默认 `DraftError`，可换成 `KeyError` 当生成器 bug）。
    输出：走 `jc.workflow.Workflow` 协议的最小实现。
    预期行为：`render` 每次计数并把它写进 state，判别器读计数——「到底重渲染了没有」
    是数出来的，不靠猜。
    """

    name = "fake_flaky"
    version = "0.0.1"
    kinds = frozenset({"noul"})
    needs = frozenset({"text"})
    judges: tuple[str, ...] = ("flaky",)

    def __init__(
        self,
        reject_until: int = 1,
        questions: int = 1,
        draft_failures: int = 0,
        error: type[Exception] = DraftError,
    ) -> None:
        self.reject_until = reject_until
        self.questions = questions
        self.draft_failures = draft_failures
        self.error = error
        self.renders = 0
        self.bound = 0
        self.backend: GenerationBackend | None = None

    def bind_backend(self, backend: GenerationBackend) -> None:
        """输入：生成后端。输出：无。预期行为：存下并计数。"""
        self.backend = backend
        self.bound += 1

    def split_names(self) -> tuple[str, ...]:
        return ("train",)

    def config_for(self, split: str) -> object:
        return {}

    def build_cases(self, config: object) -> list[JsonValue]:
        return [{"i": 0}]

    def case_label(self, case: object) -> str:
        return f"case-{case['i']}"

    def stats_dimensions(self, case: object) -> dict[str, JsonValue]:
        return {"i": case["i"]}

    def dimension_titles(self) -> dict[str, str]:
        return {"i": "序号"}

    def plan_sample(self, rng: random.Random, config: object, case: object, seq: int) -> SplitSpec:
        return SplitSpec(name="train", params={"i": case["i"]})

    def sample(self, rng: random.Random, split: SplitSpec) -> object:
        return split.params

    def render(self, spec: object, rng: random.Random, locale: str) -> str:
        """输入：spec、RNG、语言。输出：带渲染次数的 state。

        预期行为：每次调用都计数；前 `draft_failures` 次直接抛 `error`（模拟「后端没照
        契约写」这类生成期异常），计数照旧加上去，所以「重渲染了几次」看得出来。
        """
        self.renders += 1
        if self.renders <= self.draft_failures:
            raise self.error(f"第 {self.renders} 次渲染没照契约写")
        return f"state-{spec['i']}-{self.renders}"

    def ask(self, spec: object, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：spec、RNG、语言。输出：题目列表。预期行为：`questions=0` 时返回空表，
        让管线走「没有出到任何题」这条生成期失败路径。"""
        if not self.questions:
            return []
        return [LabeledQuestion("hit", {"type": "noul", "instructions": "命中吗"}, True, shuffle=False)]

    def reparse(self, state: object, *, case: object = None) -> object:
        return None

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[object, ...]:
        return (FlakyJudge(self),) if "flaky" in names else ()

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        return {}, []

    def available_locales(self) -> list[str]:
        return ["zh"]


class FlakyJudge:
    """假判别器：渲染次数没超过 `reject_until` 就拒。

    输入：workflow（读它的渲染计数）。输出：`Verdict`。
    预期行为：理由里写明是第几次渲染不合格，断言时不用猜是哪一次。
    """

    name = "flaky"

    def __init__(self, workflow: FlakyWorkflow) -> None:
        self.workflow = workflow

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        if self.workflow.renders <= self.workflow.reject_until:
            return Verdict(False, reason=f"第 {self.workflow.renders} 次渲染不合格")
        return Verdict(True, reason="合格")


def _run(
    monkeypatch: object, tmp_path: Path, workflow: FlakyWorkflow, attempts: int, run_id: str = "r1"
) -> pipeline.RunResult:
    """输入：monkeypatch、临时目录、假 workflow、渲染上限、run_id。输出：RunResult。

    预期行为：把 `pipeline.load` 换成假 workflow 再跑一次，不碰网络、不碰真 workflow。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: workflow)
    return pipeline.run(
        pipeline.RunConfig(
            workflow="fake_flaky",
            out_root=tmp_path,
            run_id=run_id,
            render_attempts=attempts,
            backend=object(),
        )
    )


def test_judge_reject_triggers_a_rerender(tmp_path: Path, monkeypatch: object) -> None:
    """输入：第一次渲染会被判别器拒掉。输出：RunResult。

    预期行为：渲染两次、第二次过、`render_retries` 记 1 次，且**没有**拒绝记录
    （这条样本最终是合格的，拒掉的那一版不进产物）。
    """
    workflow = FlakyWorkflow(reject_until=1)
    result = _run(monkeypatch, tmp_path, workflow, attempts=2)
    assert workflow.renders == 2
    assert result.accepted == 1
    assert result.rejected == []
    assert result.stats["render_retries"] == 1


def test_retries_stop_at_the_cap_and_keep_the_last_reason(tmp_path: Path, monkeypatch: object) -> None:
    """输入：怎么渲染都被拒。输出：RunResult。

    预期行为：渲染次数正好等于上限（不无限重试）；只留**最后一次**的拒绝理由
    （前面几次的理由没有诊断价值，留一条就够看清为什么最终没成）。
    """
    workflow = FlakyWorkflow(reject_until=99)
    result = _run(monkeypatch, tmp_path, workflow, attempts=3)
    assert workflow.renders == 3
    assert result.accepted == 0
    assert len(result.rejected) == 1
    assert result.rejected[0]["reason"] == "第 3 次渲染不合格"
    assert result.stats["render_retries"] == 2
    assert result.stats["reject_codes"]["E_JUDGE_FLAKY"] == 1


def test_one_attempt_means_no_rerender(tmp_path: Path, monkeypatch: object) -> None:
    """输入：上限设成 1（关掉重渲染）。输出：RunResult。预期行为：只渲染一次。"""
    workflow = FlakyWorkflow(reject_until=1)
    result = _run(monkeypatch, tmp_path, workflow, attempts=1)
    assert workflow.renders == 1
    assert result.accepted == 0
    assert result.stats["render_retries"] == 0


def test_generate_failure_is_not_rerendered(tmp_path: Path, monkeypatch: object) -> None:
    """输入：一条出不出题来的样本（生成期失败）。输出：RunResult。

    预期行为：只渲染一次、记 `E_ASSEMBLE`——生成期失败是**生成器 bug**，
    重渲染只会把它藏起来（PLAN 6.1 第一道闸：拒绝率 >0 视为生成器 bug，要修）。
    """
    workflow = FlakyWorkflow(questions=0)
    result = _run(monkeypatch, tmp_path, workflow, attempts=3)
    assert workflow.renders == 1
    assert result.accepted == 0
    assert [row["code"] for row in result.rejected] == [pipeline.E_ASSEMBLE]
    assert result.stats["render_retries"] == 0


def test_progress_records_attempts_and_resume_replays_them(tmp_path: Path, monkeypatch: object) -> None:
    """输入：跑一遍再按同一 run_id 续跑。输出：两次的 RunResult。

    预期行为：进度日志里记下这条样本渲染了几次；续跑**一次都不重渲染**，
    统计里的重渲染次数照旧从进度日志回放出来（与一次跑完完全一致）。
    """
    workflow = FlakyWorkflow(reject_until=1)
    first = _run(monkeypatch, tmp_path, workflow, attempts=2)
    record = json.loads(
        (first.run_dir / "progress-train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["attempts"] == 2
    assert record["status"] == "ok"
    again = _run(monkeypatch, tmp_path, workflow, attempts=2)
    assert workflow.renders == 2  # 续跑没再渲染
    assert again.accepted == 1
    assert again.stats["render_retries"] == 1


def test_manifest_records_the_attempt_limit(tmp_path: Path, monkeypatch: object) -> None:
    """输入：上限设成 3。输出：manifest。预期行为：把上限记进 manifest（溯源用）。"""
    workflow = FlakyWorkflow(reject_until=0)
    result = _run(monkeypatch, tmp_path, workflow, attempts=3)
    manifest = json.loads(
        (result.run_dir / "manifest-train.json").read_text(encoding="utf-8")
    )
    assert manifest["render_attempts"] == 3


def test_render_attempts_must_be_positive(tmp_path: Path, monkeypatch: object) -> None:
    """输入：上限 0。输出：异常。预期行为：报人话，且一个样本都不渲染。"""
    workflow = FlakyWorkflow()
    with pytest.raises(ValueError, match="render_attempts"):
        _run(monkeypatch, tmp_path, workflow, attempts=0)
    assert workflow.renders == 0


def test_draft_error_is_rerendered(tmp_path: Path, monkeypatch: object) -> None:
    """输入：第一次渲染抛 `DraftError`（后端没照契约写），第二次正常。

    输出：RunResult。预期行为：重渲染一次就过，`render_retries` 记 1 次、零拒绝记录——
    「模型没照区间写」是采样抖动，重渲染正是它该走的路（契约级失败**单发**会把
    四成的好样本白白丢掉，实测见 PLAN 10.83）。
    """
    workflow = FlakyWorkflow(reject_until=0, draft_failures=1)
    result = _run(monkeypatch, tmp_path, workflow, attempts=2)
    assert workflow.renders == 2
    assert result.accepted == 1
    assert result.rejected == []
    assert result.stats["render_retries"] == 1


def test_draft_error_stops_at_the_cap(tmp_path: Path, monkeypatch: object) -> None:
    """输入：怎么渲染都抛 `DraftError`。输出：RunResult。

    预期行为：渲染次数正好等于上限（不无限重试），最后一条进拒绝、码是 `E_ASSEMBLE`。
    """
    workflow = FlakyWorkflow(reject_until=0, draft_failures=99)
    result = _run(monkeypatch, tmp_path, workflow, attempts=3)
    assert workflow.renders == 3
    assert result.accepted == 0
    assert [row["code"] for row in result.rejected] == [pipeline.E_ASSEMBLE]
    assert result.stats["render_retries"] == 2


def test_plain_exception_is_still_not_rerendered(tmp_path: Path, monkeypatch: object) -> None:
    """输入：生成期抛 `KeyError`（没有 `retryable_for_case` 标记）。

    输出：RunResult。预期行为：只渲染一次——判据看的是**异常自己声明的标记**，
    不是「异常是不是 ValueError」，所以生成器 bug 照旧藏不住。
    """
    workflow = FlakyWorkflow(reject_until=0, draft_failures=99, error=KeyError)
    result = _run(monkeypatch, tmp_path, workflow, attempts=3)
    assert workflow.renders == 1
    assert result.stats["render_retries"] == 0
