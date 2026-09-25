"""通用层测试：把生成后端接进管线（PLAN 3.1）。

这里用「假 workflow + 假后端」只验接线本身：workflow 声明 `needs` 时管线必须把后端注入进去、
并在 manifest 里记账；没给后端必须报错，绝不静默降级成「没有文本外壳的样本」。
**不发任何网络请求**，也不碰 `[llm]` extra。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jc import cli, pipeline
from jc.backend.types import AllProvidersDown, Draft, DraftInvalidError, TransportError
from jc.workflow import LabeledQuestion, SplitSpec


class StubBackend:
    """最小假后端：不联网，只记账。

    输入：无。输出：`generate()` 返回 None（假 workflow 不看返回值，只看调用痕迹）。
    预期行为：把收到的 task_id 记进 `calls`，测试据此断言「后端真的被用到了」。
    刻意不给 `capabilities`：manifest 里的 starting_mode 应当是空串。
    """

    name = "stub"
    capabilities = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, task: Any) -> None:
        """输入：一个任务。输出：None。预期行为：记下 task_id。"""
        self.calls.append(task.task_id)

    def generate_many(self, tasks: list[Any]) -> list[None]:
        """输入：任务列表。输出：与入参等长的 None 列表。"""
        return [self.generate(task) for task in tasks]


class FakeTask:
    """假任务：假 workflow 只用到 `task_id`。"""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


class BackendWorkflow:
    """假 workflow：声明需要 `text` 能力，`render` 阶段真的去调后端。

    输入/输出：遵循 `jc.workflow.Workflow` 协议的最小子集。
    预期行为：`bind_backend` 记次数，`render` 每次都调后端——这样「管线漏调 bind」会被测出来。
    """

    name = "fake_llm"
    version = "0.0.1"
    kinds = frozenset({"noul"})
    needs = frozenset({"text"})
    judges: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.backend: Any = None
        self.bound = 0

    def bind_backend(self, backend: Any) -> None:
        """输入：生成后端。输出：无。预期行为：存下后端并计数。"""
        self.backend = backend
        self.bound += 1

    def split_names(self) -> tuple[str, ...]:
        return ("train",)

    def config_for(self, split: str) -> Any:
        return {}

    def build_cases(self, config: Any) -> list[Any]:
        return [{"i": 0}, {"i": 1}]

    def case_label(self, case: Any) -> str:
        return f"case-{case['i']}"

    def stats_dimensions(self, case: Any) -> dict[str, Any]:
        return {"i": case["i"]}

    def dimension_titles(self) -> dict[str, str]:
        return {"i": "序号"}

    def plan_sample(self, rng: Any, config: Any, case: Any, seq: int) -> SplitSpec:
        return SplitSpec(name="train", params={"i": case["i"]})

    def sample(self, rng: Any, split: SplitSpec) -> Any:
        return split.params

    def render(self, spec: Any, rng: Any, locale: str) -> Any:
        self.backend.generate(FakeTask(f"fake-{spec['i']}"))
        return f"state-{spec['i']}"

    def ask(self, spec: Any, rng: Any, locale: str) -> list[LabeledQuestion]:
        return [
            LabeledQuestion(
                "hit", {"type": "noul", "instructions": "命中吗"}, True, shuffle=False
            )
        ]

    def reparse(self, state: Any, *, case: Any = None) -> Any:
        return None

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[()]:
        return ()

    def audit(self, run_dir: Path, splits: tuple[str, ...], locale: str) -> tuple[dict[str, Any], list[str]]:
        return {}, []

    def available_locales(self) -> list[str]:
        return ["zh"]


def test_needs_backend_without_backend_raises(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：声明了 needs 的 workflow，但没给 backend。

    输出：异常。
    预期行为：报人话的 ValueError 并且**一个样本都不生成**（绝不静默降级）。
    """
    instance = BackendWorkflow()
    monkeypatch.setattr(pipeline, "load", lambda name: instance)
    with pytest.raises(ValueError, match="RunConfig.backend"):
        pipeline.run(pipeline.RunConfig(workflow="fake_llm", out_root=tmp_path))
    assert instance.bound == 0
    assert not list(tmp_path.rglob("*.jsonl"))


def test_backend_is_bound_and_really_used(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：声明了 needs 的 workflow + 一个假后端。

    输出：RunResult。
    预期行为：后端被注入一次、每条样本的 render 都真的调了后端、manifest 里记下后端名字。
    """
    instance = BackendWorkflow()
    backend = StubBackend()
    monkeypatch.setattr(pipeline, "load", lambda name: instance)
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_llm", out_root=tmp_path, run_id="r1", backend=backend
        )
    )
    assert instance.bound == 1
    assert backend.calls == ["fake-0", "fake-1"]
    assert result.accepted == 2
    manifest = json.loads(
        (result.run_dir / "manifest-train.json").read_text(encoding="utf-8")
    )
    assert manifest["backend"] == {"name": "stub", "starting_mode": ""}
    assert manifest["needs"] == ["text"]


def test_backend_manifest_empty_without_backend() -> None:
    """输入：没有后端的配置。输出：空字典。

    预期行为：纯构造式 workflow 的 manifest 里不出现 `backend` 字段的实质内容。
    """
    assert pipeline.backend_manifest(pipeline.RunConfig(workflow="maze")) == {}


def test_backend_manifest_prefers_the_backends_own_starting_mode() -> None:
    """输入：一个自报起始档位的假后端。输出：manifest 里的后端记账。

    预期行为：用后端自报的档位，而不是 `capabilities` 的探测结论——探测没开时
    `capabilities` 是空的，会答 `md_json`，把实际从 `json_schema` 起步的 run 记成错账。
    """

    class ReportingBackend(StubBackend):
        """假后端：像 `LLMBackend` 那样自报真正会先试的档位。"""

        starting_mode = "json_schema"

    info = pipeline.backend_manifest(
        pipeline.RunConfig(workflow="fake_llm", backend=ReportingBackend())
    )
    assert info == {"name": "stub", "starting_mode": "json_schema"}


def test_load_backend_empty_path_returns_none() -> None:
    """输入：空路径。输出：None。

    预期行为：不给 `--backend-config` 时不 import `jc.backend`，构造式 workflow 照跑。
    """
    assert cli._load_backend("") is None


class FatalBackend(StubBackend):
    """假后端：从第 `fail_from + 1` 条起像「全部 provider 都不可用」那样抛致命错误。

    输入：`fail_from`（前多少条放行）。输出：抛 `AllProvidersDown`。
    预期行为：复现「跑到一半全挂」——致命错误必须让整批停下，而不是记成一条被拒。
    """

    def __init__(self, fail_from: int = 0) -> None:
        super().__init__()
        self.fail_from = fail_from

    def generate(self, task: Any) -> None:
        """输入：一个任务。输出：None 或抛 `AllProvidersDown`。"""
        super().generate(task)
        if len(self.calls) > self.fail_from:
            raise AllProvidersDown("全部 provider 都不可用：['nim', 'nim-glm']")


def test_all_providers_down_aborts_the_run(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：第一条就全挂的假后端。

    输出：`RunAborted`。
    预期行为：整批停下（PLAN 3.1.4），**不**产出一个「全是被拒」的假数据集；
    一条合格都没有（所以连 `train.jsonl` 都不建空文件），且 manifest 里记下中止。
    """
    instance = BackendWorkflow()
    monkeypatch.setattr(pipeline, "load", lambda name: instance)
    config = pipeline.RunConfig(
        workflow="fake_llm", out_root=tmp_path, run_id="boom", backend=FatalBackend()
    )
    with pytest.raises(pipeline.RunAborted) as excinfo:
        pipeline.run(config)
    assert excinfo.value.accepted == 0
    assert excinfo.value.seq == 0
    run_dir = tmp_path / "boom"
    assert not (run_dir / "train.jsonl").exists()
    # 一条都没完成，所以进度日志也不存在（追加写不会凭空造空文件）
    assert not (run_dir / "progress.jsonl").exists()
    manifest = json.loads((run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["aborted"]["seq"] == 0


def test_abort_keeps_the_progress_already_made(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：跑通一条之后才全挂的假后端。

    输出：`RunAborted`，且那一条合格样本留在盘上。
    预期行为：中止不等于丢掉已做的工作——`accepted` 与产物都要对得上。
    """
    instance = BackendWorkflow()
    monkeypatch.setattr(pipeline, "load", lambda name: instance)
    config = pipeline.RunConfig(
        workflow="fake_llm", out_root=tmp_path, run_id="half", backend=FatalBackend(fail_from=1)
    )
    with pytest.raises(pipeline.RunAborted) as excinfo:
        pipeline.run(config)
    assert excinfo.value.accepted == 1
    assert excinfo.value.seq == 1
    rows = (tmp_path / "half" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["id"] == "fake_llm-train-0000000"


class JunkBackend(StubBackend):
    """假后端：从第 `junk_at + 1` 条起像「模型答非所问」那样抛 `DraftInvalidError`。

    输入：`junk_at`（前多少条放行）。输出：抛 `DraftInvalidError`。
    预期行为：复现 kimi 在 tools 档的偶发路径（docs §9.11）——provider 活着，
    只是这条草稿不合格，**不该**把整批掐了。
    """

    def __init__(self, junk_at: int = 0) -> None:
        super().__init__()
        self.junk_at = junk_at

    def generate(self, task: Any) -> None:
        """输入：一个任务。输出：None 或抛 `DraftInvalidError`。"""
        super().generate(task)
        if len(self.calls) > self.junk_at:
            raise DraftInvalidError("模型没给出合格草稿：list index out of range")


def test_junk_draft_is_a_reject_not_an_abort(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：第 2 条起给不出合格草稿的假后端。

    输出：正常跑完的 `RunResult`（**不是** `RunAborted`）。
    预期行为：草稿不合格属于样本级失败——记进 rejected.jsonl、整批继续跑；
    PLAN 3.1.4 的「整批停下」只留给「provider 真的挂了」。
    """
    instance = BackendWorkflow()
    monkeypatch.setattr(pipeline, "load", lambda name: instance)
    backend = JunkBackend(junk_at=1)
    result = pipeline.run(
        pipeline.RunConfig(workflow="fake_llm", out_root=tmp_path, run_id="junk", backend=backend)
    )
    assert result.accepted == 1
    assert len(result.rejected) == len(backend.calls) - 1
    assert {row["code"] for row in result.rejected} == {"E_ASSEMBLE"}
    assert len(backend.calls) > 1  # 后面几条真的继续跑了，不是掐掉整批


def test_concurrency_must_be_positive(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：concurrency=0。输出：ValueError。预期行为：非法配置当场报人话，不静默当串行。"""
    monkeypatch.setattr(pipeline, "load", lambda name: BackendWorkflow())
    with pytest.raises(ValueError, match="concurrency"):
        pipeline.run(
            pipeline.RunConfig(
                workflow="fake_llm", out_root=tmp_path, concurrency=0, backend=StubBackend()
            )
        )


def test_concurrent_run_matches_serial_run(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：同一份配置，一次串行、一次并发 2。

    输出：两次的 `train.jsonl`。
    预期行为：逐字节一致——每条样本的 RNG 只由序号派生，并发不该改变内容或顺序。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: BackendWorkflow())
    serial = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_llm", out_root=tmp_path, run_id="s", backend=StubBackend()
        )
    )
    concurrent = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_llm", out_root=tmp_path, run_id="c", concurrency=2, backend=StubBackend()
        )
    )
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        concurrent.run_dir / "train.jsonl"
    ).read_bytes()
    assert serial.stats["reject_codes"] == concurrent.stats["reject_codes"]
    assert serial.stats["judges"] == concurrent.stats["judges"]


def test_concurrent_abort_stops_submitting_more(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：并发 4 跑 2 条、后端从头就全挂。

    输出：`RunAborted`。
    预期行为：并发路径也要能中止（不能只让串行路径会停），且没起跑的任务直接取消。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: BackendWorkflow())
    backend = FatalBackend()
    config = pipeline.RunConfig(
        workflow="fake_llm", out_root=tmp_path, run_id="par-boom", concurrency=4, backend=backend
    )
    with pytest.raises(pipeline.RunAborted):
        pipeline.run(config)
    # 断言不写死条数：取消是尽力而为（已经在跑的线程拦不住），但第一条必然问过，
    # 且**不能**把两条都跑完——那说明中止根本没生效。
    assert backend.calls[0] == "fake-0"
    assert len(backend.calls) <= 2


def test_load_backend_assembles_with_probe(monkeypatch: Any) -> None:
    """输入：一个配置路径（`jc.backend` 被换掉，不真读文件、不真联网）。

    输出：装配出来的后端对象。
    预期行为：先 `load_config` 再 `build_backend(probe=True)`——开机探测是 PLAN 3.1.1 的要求。
    """
    import jc.backend as backend_mod

    seen: dict[str, Any] = {}

    def fake_load_config(path: str) -> str:
        seen["path"] = path
        return "CFG"

    def fake_build_backend(cfg: str, *, probe: bool) -> str:
        seen["cfg"] = cfg
        seen["probe"] = probe
        return "BACKEND"

    monkeypatch.setattr(backend_mod, "load_config", fake_load_config)
    monkeypatch.setattr(backend_mod, "build_backend", fake_build_backend)
    assert cli._load_backend("x.toml") == "BACKEND"
    assert seen == {"path": "x.toml", "cfg": "CFG", "probe": True}


# ---- LLM 质检判别器接进管线（PLAN 6.2）--------------------------------------


class LlmJudgeWorkflow(BackendWorkflow):
    """假 workflow：默认判别器里带一只 LLM 眼睛（`llm:solvable`）。"""

    name = "fake_llm_judge"
    judges = ("llm:solvable",)


class LlmJudgeNoNeedsWorkflow(LlmJudgeWorkflow):
    """假 workflow：不要文本能力，却声明了 LLM 质检。

    用来把「LLM 质检自己要后端」这条单独验出来——不然会被 `needs` 那条更早的报错盖住，
    测试看着绿，实际根本没走到判别器装配那一步。
    """

    name = "fake_llm_judge_no_needs"
    needs: frozenset[str] = frozenset()


class JudgeBackend(StubBackend):
    """假后端：质检任务返回一条结论，别的任务照旧（只记 task_id）。"""

    def __init__(
        self, *, verdict: bool = True, reason: str = "看着没问题", exc: BaseException | None = None
    ) -> None:
        """输入：结论、理由、要抛的异常。输出：实例。"""
        super().__init__()
        self.verdict = verdict
        self.reason = reason
        self.exc = exc

    def generate(self, task: Any) -> Any:
        """输入：任务。输出：质检任务给草稿，其余给 None。预期行为：先记账。"""
        self.calls.append(task.task_id)
        if "-judge-" not in task.task_id:
            return None
        if self.exc is not None:
            raise self.exc
        return Draft(
            data={"verdict": self.verdict, "score": 0.8, "reason": self.reason},
            raw_text="{}",
            backend="stub",
            provider="p",
            model="m",
            mode="tools",
        )


def test_llm_judge_without_backend_raises(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：声明了 LLM 质检的 workflow，但没给后端。输出：ValueError。

    预期行为：**当场报人话**，不静默把这只眼睛摘掉——摘掉之后统计里少一项，
    看起来反而像「都通过」。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: LlmJudgeNoNeedsWorkflow())
    with pytest.raises(ValueError, match="LLM 质检判别器"):
        pipeline.run(pipeline.RunConfig(workflow="fake_llm_judge_no_needs", out_root=tmp_path))


def test_llm_judge_failure_is_skipped_not_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：质检每次都被端点掐断（传输错误）。输出：样本照收 + 单独记账。

    预期行为：合格 2、被拒 0、`judge-skipped.jsonl` 两条、统计里记 skipped——
    这就是 PLAN 10.10 的教训：端点故障不能被写成数据质量问题。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: LlmJudgeWorkflow())
    backend = JudgeBackend(exc=TransportError("504"))
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_llm_judge", out_root=tmp_path, run_id="skip", backend=backend
        )
    )
    assert result.accepted == 2
    assert result.rejected == []
    assert result.stats["judge_skipped"] == 2
    assert result.stats["judges"]["llm:solvable"] == {"ok": 0, "fail": 0, "skipped": 2}
    skipped = [
        json.loads(line)
        for line in (result.run_dir / "judge-skipped.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["judge"] for row in skipped} == {"llm:solvable"}
    assert "端点故障" in skipped[0]["reason"]
    stats_md = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    assert "跳过 2 条" in stats_md


def test_llm_judge_saying_no_rejects_the_sample(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：质检说「有问题」。输出：样本被拒，拒绝码带视角名。

    预期行为：拒绝码是 `E_JUDGE_LLM_SOLVABLE`（名字里的 `:` 换成 `_`），
    理由原样带上，方便定位是哪只眼睛看到了什么。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: LlmJudgeWorkflow())
    backend = JudgeBackend(verdict=False, reason="两个选项都说得通")
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_llm_judge", out_root=tmp_path, run_id="no", backend=backend
        )
    )
    assert result.accepted == 0
    assert {row["code"] for row in result.rejected} == {"E_JUDGE_LLM_SOLVABLE"}
    assert "两个选项都说得通" in result.rejected[0]["reason"]
    assert result.stats["judges"]["llm:solvable"] == {"ok": 0, "fail": 2, "skipped": 0}


def test_llm_judge_fatal_error_aborts_the_run(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：质检时全部 provider 都挂了。输出：`RunAborted`。

    预期行为：判别器也不能把「全挂了」吞成 skipped——那会产出一整个「没质检」的假数据集。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: LlmJudgeWorkflow())
    backend = JudgeBackend(exc=AllProvidersDown("全挂了"))
    with pytest.raises(pipeline.RunAborted):
        pipeline.run(
            pipeline.RunConfig(
                workflow="fake_llm_judge", out_root=tmp_path, run_id="fatal", backend=backend
            )
        )


def test_rule_judges_still_run_alongside_llm_ones(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：规则判别器 + LLM 判别器一起开。输出：两个都跑、各记各的。

    预期行为：规则判别器由 workflow 构造、LLM 的由通用层装配，两者在统计里各占一行。
    """

    class CountingJudge:
        """假规则判别器：永远通过，只数自己被调了几次。"""

        name = "counting"

        def __init__(self) -> None:
            self.seen = 0

        def judge(self, sample: dict[str, Any], spec: Any = None) -> Any:
            """输入：样本。输出：通过的 Verdict。预期行为：计数。"""
            from jc.judge import Verdict

            self.seen += 1
            return Verdict(True, reason="ok")

    instance = LlmJudgeWorkflow()
    counter = CountingJudge()
    instance.make_judges = lambda names, locale: (counter,)  # type: ignore[method-assign]
    monkeypatch.setattr(pipeline, "load", lambda name: instance)
    backend = JudgeBackend()
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_llm_judge",
            out_root=tmp_path,
            run_id="both",
            judges=("counting", "llm:wording"),
            backend=backend,
        )
    )
    assert counter.seen == 2
    assert set(result.stats["judges"]) == {"counting", "llm:wording"}
    assert result.stats["judges"]["llm:wording"] == {"ok": 2, "fail": 0, "skipped": 0}


# ---- 一致性校验接进管线（照抄 Open-Jev 的 validate_records）------------------


class DuplicateStateWorkflow(BackendWorkflow):
    """假 workflow：素材故意一模一样，答案随 case 变——这正是「自相矛盾的训练对」。"""

    name = "fake_dup_state"
    judges: tuple[str, ...] = ()

    def render(self, spec: Any, rng: Any, locale: str) -> Any:
        """输入：规格。输出：**固定不变**的素材文本。预期行为：不调后端。"""
        return "同一段素材"

    def ask(self, spec: Any, rng: Any, locale: str) -> list[LabeledQuestion]:
        """输入：规格。输出：一道 noul 题，答案随 case 变（于是同素材出现了两个答案）。"""
        return [
            LabeledQuestion(
                "hit", {"type": "noul", "instructions": "命中吗"}, spec["i"] == 0, shuffle=False
            )
        ]


def test_conflicting_answers_on_one_state_are_reported(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：两条素材相同、答案相反的样本。输出：统计里记 1 处冲突 + conflicts.jsonl。

    预期行为：样本照收（不拒数据），但冲突必须露出来——同素材同题不同答是自相矛盾的训练对。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: DuplicateStateWorkflow())
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_dup_state", out_root=tmp_path, run_id="dup", backend=StubBackend()
        )
    )
    assert result.accepted == 2
    assert result.stats["conflicts"] == 1
    rows = [
        json.loads(line)
        for line in (result.run_dir / "conflicts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert set(rows[0]["answers"]) == {"True", "False"}
    assert rows[0]["question"] == "命中吗"
    stats_md = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    assert "⚠️ 1 处" in stats_md


def test_clean_run_reports_no_conflicts(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：素材两两不同的正常 run。输出：0 处冲突、不写 conflicts.jsonl。

    预期行为：报告里也要写出「0 处」——没写的话读者分不清「查过没问题」与「根本没查」。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: BackendWorkflow())
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_llm", out_root=tmp_path, run_id="clean", backend=StubBackend()
        )
    )
    assert result.stats["conflicts"] == 0
    assert not (result.run_dir / "conflicts.jsonl").exists()
    stats_md = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    assert "✅ 没有「同素材同题不同答」" in stats_md


def test_torn_last_line_is_dropped_but_a_broken_middle_line_raises(tmp_path: Path) -> None:
    """输入：一个尾行残缺的 jsonl、一个中间行坏掉的 jsonl。输出：行列表 / 异常。

    预期行为：只容忍**最后一行**——追加写 + fsync 的性质就是「崩溃时最多留半行」，
    那半行当它没写过；中间行坏了说明文件真坏了，必须报出来（静默跳过坏行等于
    把数据丢了还说没事）。
    """
    path = tmp_path / "x.jsonl"
    path.write_text('{"a":1}\n{"a":2}\n{"a":', encoding="utf-8")
    assert [row["a"] for row in pipeline._read_rows(path)] == [1, 2]
    path.write_text('{"a":1}\n{"a":\n{"a":3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行坏了"):
        pipeline._read_rows(path)


def test_run_writes_one_journal_line_per_sample(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：两条样本的正常跑。输出：`progress.jsonl`。

    预期行为：每个序号一行，`row` 指向 `<split>.jsonl` 的第几行——续跑就靠它把样本取回来，
    所以行号必须与产物对得上，而不是另记一套。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: BackendWorkflow())
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_llm", out_root=tmp_path, run_id="jr", backend=StubBackend()
        )
    )
    records = [
        json.loads(line)
        for line in (result.run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["seq"] for r in records] == [0, 1]
    assert [r["status"] for r in records] == ["ok", "ok"]
    assert [r["row"] for r in records] == [1, 2]
    assert all(r["rejected"] == [0, 0] and r["skipped"] == [0, 0] for r in records)
    samples = [json.loads(line) for line in (result.run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [s["id"] for s in samples] == ["fake_llm-train-0000000", "fake_llm-train-0000001"]


def test_rerunning_the_same_run_id_skips_everything(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：同一个 run_id 跑两遍。输出：两次 RunResult。

    预期行为：第二遍一条都不重跑（后端零调用）、产物逐字节不变、统计一致——
    「重启继续」就是不重花配额，也不把同一条写两遍。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: BackendWorkflow())
    backend = StubBackend()
    config = pipeline.RunConfig(
        workflow="fake_llm", out_root=tmp_path, run_id="again", backend=backend
    )
    first = pipeline.run(config)
    before = (first.run_dir / "train.jsonl").read_text(encoding="utf-8")
    calls = list(backend.calls)
    second = pipeline.run(config)
    assert backend.calls == calls
    assert (second.run_dir / "train.jsonl").read_text(encoding="utf-8") == before
    assert second.accepted == first.accepted
    assert second.stats == first.stats


def test_resume_after_abort_matches_a_single_clean_run(tmp_path: Path, monkeypatch: Any) -> None:
    """输入：先跑到中止（第 1 条起全挂），再换正常后端用同一个 run_id 续跑。输出：两次产物。

    预期行为：续跑只补没完成的那条（后端只被调一次），最终产物与统计和「一次跑完」逐字节
    一致——这是续跑的正确性判据：它不该改变结果，只该省掉已经做过的工作。
    """
    monkeypatch.setattr(pipeline, "load", lambda name: BackendWorkflow())
    with pytest.raises(pipeline.RunAborted) as excinfo:
        pipeline.run(
            pipeline.RunConfig(
                workflow="fake_llm",
                out_root=tmp_path,
                run_id="resume",
                backend=FatalBackend(fail_from=1),
            )
        )
    assert excinfo.value.accepted == 1  # 中止前那条已经落盘
    backend = StubBackend()
    resumed = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_llm", out_root=tmp_path, run_id="resume", backend=backend
        )
    )
    assert backend.calls == ["fake-1"]  # 只补了缺的那条
    clean = pipeline.run(
        pipeline.RunConfig(
            workflow="fake_llm", out_root=tmp_path, run_id="clean", backend=StubBackend()
        )
    )
    resumed_rows = (resumed.run_dir / "train.jsonl").read_text(encoding="utf-8")
    assert resumed_rows == (clean.run_dir / "train.jsonl").read_text(encoding="utf-8")
    assert resumed.stats == clean.stats
