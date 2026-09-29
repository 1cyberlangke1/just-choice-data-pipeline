"""浏览器 / GUI 操作的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod

from workflows.browser import features, locales, render, workflow as browser_workflow
from workflows.browser.spec import BrowserSpec, spec_to_dict

from tests.workflows.browser.conftest import FakeBackend, request_for

CASE_LIMIT = 8

#: 只由代码算得出来的那六族（`target` 的答案空间随屏幕变，单独看）。
ANSWER_FAMILIES = ("intent", "action", "done", "blocked", "confirm", "risk")


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train", lang: str = "zh"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="browser",
        split=split,
        locales=(lang,),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=browser_workflow.BrowserConfig(split=split, case_limit=CASE_LIMIT),
        backend=backend,
    )


def levels_of(cases: list[features.TagSet]) -> set[tuple[str, object]]:
    """输入：一批 case。输出：这六族上出现过的 (题族, 答案) 组合。"""
    out: set[tuple[str, object]] = set()
    for case in cases:
        labels = case.label()
        for name in ANSWER_FAMILIES:
            out.add((name, getattr(labels, name)))
    return out


def test_audit_accepts_the_other_locales_tables(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：一份 **en** 产物，审计时故意传 zh。输出：(报告, 缺失项)。

    预期行为：零缺失——门控表的「说明」那一列是随语言渲染的文案，审计必须按
    「等于任一语言的规范表」判定，而不是拿调用方给的那一种语言去比。反面实测过：
    混语言的产物按 zh 口径审计，8 条里 4 条被误判成「门控表与规范表对不上」；
    这份全 en 的产物更是 8 条全中。
    """
    result = pipeline.run(make_config(tmp_path, "loc", fake_backend(), lang="en"))
    assert result.accepted == CASE_LIMIT
    report, missing = workflow_mod.load("browser").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["gates_bad"] == 0


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，
    而且七个题族与三张表全都自洽。
    """
    result = pipeline.run(make_config(tmp_path, "c1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("browser").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    counts = report["train"]
    assert counts["samples"] == CASE_LIMIT
    assert counts["frame_bad"] == 0
    assert counts["gates_bad"] == 0
    assert counts["criteria_bad"] == 0
    assert counts["target_bad"] == 0
    assert counts["gate_bad"] == 0
    assert counts["risk_bad"] == 0


def test_two_runs_are_byte_identical(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：同种子跑两次。输出：两份产物。预期行为：逐字节一致。"""
    first = pipeline.run(make_config(tmp_path, "c2a", fake_backend()))
    second = pipeline.run(make_config(tmp_path, "c2b", fake_backend()))
    assert (first.run_dir / "train.jsonl").read_bytes() == (
        second.run_dir / "train.jsonl"
    ).read_bytes()


def test_end_to_end_in_english(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：英文假后端。输出：RunResult。

    预期行为：换语言照样零拒绝、零缺失项——门控表的说明文字按语言写，判据词元不变。
    """
    result = pipeline.run(make_config(tmp_path, "c2c", fake_backend(lang="en"), lang="en"))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    report, missing = workflow_mod.load("browser").audit(result.run_dir, ("train",), "en")
    assert missing == []
    assert report["train"]["gates_bad"] == 0


def test_trace_judge_rejects_a_misread_intent(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解把意图说反的假后端。输出：RunResult。

    预期行为：这些样本被拒，拒绝码是 `E_JUDGE_TRACE`——判别器抓的是模型真会犯的错。
    """
    result = pipeline.run(make_config(tmp_path, "c3", fake_backend(trace_lies_intent=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_trace_judge_rejects_a_misread_row(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解把行号说错。输出：RunResult。预期行为：同样被拒。"""
    result = pipeline.run(make_config(tmp_path, "c4", fake_backend(trace_lies_row=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_trace_judge_rejects_a_misread_label(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解把元素名说成另一个。输出：RunResult。预期行为：同样被拒。"""
    result = pipeline.run(make_config(tmp_path, "c5", fake_backend(trace_lies_label=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_render_rejects_a_state_word_in_the_sentence(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：把界面状态写进那句话的假后端。输出：RunResult。

    预期行为：渲染期就拒掉，拒绝码是 `E_ASSEMBLE`（生成阶段的统一拒绝码）——
    那句话拼进 state 就洗不掉了，而且会与元素表打架。
    """
    result = pipeline.run(make_config(tmp_path, "c6", fake_backend(state_request=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_render_rejects_a_missing_element_name(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：那句话里没写目标元素名。输出：RunResult。

    预期行为：渲染期就拒掉——反解判别器要拿那个名字比对，少了它这道题读不出「指的是谁」。
    """
    result = pipeline.run(make_config(tmp_path, "c7", fake_backend(missing_label=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_render_rejects_a_multi_line_sentence(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：那句话写成两行的假后端。输出：RunResult。预期行为：重试耗尽后仍被拒，不静默放行。"""
    result = pipeline.run(make_config(tmp_path, "c8", fake_backend(multi_line_request=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：同分布三个切分互不相交、`test_ood` 与它们不相交，四者并起来等于全空间——
    一个组合都不丢、也不重复。
    """
    wf = browser_workflow.BrowserWorkflow()
    parts: dict[str, set[tuple[tuple[str, str], ...]]] = {}
    for split in wf.split_names():
        cases = wf.build_cases(wf.config_for(split))
        parts[split] = {tuple(case.as_dict().items()) for case in cases}
        assert cases, split
    id_union = parts["train"] | parts["val"] | parts["test_iid"]
    assert parts["train"] & parts["val"] == set()
    assert parts["train"] & parts["test_iid"] == set()
    assert parts["val"] & parts["test_iid"] == set()
    assert id_union & parts["test_ood"] == set()
    assert id_union == {
        tuple(case.as_dict().items())
        for case in features.build_cases(1024, dims=browser_workflow.id_dims())
    }


def test_train_side_covers_every_answer_level() -> None:
    """输入：train 切分的 case 列表。输出：无。

    预期行为：训练侧每一族每一档都到得了——只靠「同分布侧整体」到得了还不够，
    训练集本身要能教到每一个答案。
    """
    cases = browser_workflow.cases_for(browser_workflow.BrowserConfig(split="train"))
    got = levels_of(cases)
    assert {value for name, value in got if name == "intent"} == set(features.INTENTS)
    assert {value for name, value in got if name == "action"} == set(features.ACTIONS)
    assert {value for name, value in got if name in {"done", "blocked", "confirm"}} == {False, True}
    assert {value for name, value in got if name == "risk"} == set(range(features.RISK_LEVELS))


def test_ood_side_only_carries_the_held_out_value() -> None:
    """输入：`test_ood` 的 case 列表。输出：无。预期行为：每一条都带留出的那个取值。"""
    cases = browser_workflow.cases_for(browser_workflow.BrowserConfig(split="test_ood"))
    assert cases
    for case in cases:
        assert case.screen in browser_workflow.OOD_SCREENS


def test_cover_limit_forces_full_enumeration() -> None:
    """输入：默认 `cover_limit`。输出：无。

    预期行为：同分布空间（128 组）与全空间（192 组）都在 256 之下，所以走的是**全枚举**
    ——成对覆盖在这里只会白扔组合。
    """
    config = browser_workflow.BrowserConfig()
    assert config.cover_limit >= 192
    assert len(features.build_cases(config.cover_limit)) == 192
    assert browser_workflow.cases_for(browser_workflow.BrowserConfig(split="train"))


def test_holding_out_the_ood_value_keeps_every_answer_level() -> None:
    """输入：剔除 `screen=settings` 的维度表。输出：无。

    预期行为：六族的每一档都还在——`screen` 只决定填充行的角色顺序，不出现在任何题族的
    选项与判据条件里。这是「为什么 OOD 留 `screen`」的实证。
    """
    full = levels_of(features.build_cases(1024))
    held = levels_of(
        features.build_cases(
            1024,
            dims={
                **features.DIMS,
                "screen": tuple(v for v in features.SCREENS if v not in browser_workflow.OOD_SCREENS),
            },
        )
    )
    assert held == full


@pytest.mark.parametrize(
    ("dim", "value"),
    [
        ("intent", "fill"),
        ("intent", "submit"),
        ("state", "ready"),
        ("state", "current"),
        ("sensitive", "yes"),
    ],
)
def test_holdouts_that_would_kill_an_answer_level(dim: str, value: str) -> None:
    """输入：逐个留出另外几维的一档。输出：无。

    预期行为：都会杀掉某一族的答案档——所以这几维不能拿来当 OOD 的留出，
    能留的只有 `screen`（与 `decoy`）。
    """
    full = levels_of(features.build_cases(1024))
    held = levels_of(
        features.build_cases(
            1024, dims={**features.DIMS, dim: tuple(v for v in features.DIMS[dim] if v != value)}
        )
    )
    assert held < full, f"留出 {dim}={value} 竟然没杀掉任何答案档"


def test_bad_knobs_are_rejected() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话。"""
    with pytest.raises(ValueError):
        browser_workflow.BrowserConfig(split="train", case_limit=-1)
    with pytest.raises(ValueError):
        browser_workflow.BrowserConfig(split="train", phrasing_pool=())
    with pytest.raises(KeyError):
        browser_workflow.BrowserWorkflow().config_for("nope")


def test_render_without_backend_raises() -> None:
    """输入：没注入后端的 workflow。输出：无。

    预期行为：直接报错，绝不悄悄降级成模板——那会让「那句话由模型写」这条前提失效。
    """
    wf = browser_workflow.BrowserWorkflow()
    spec = BrowserSpec(tags=features.build_cases(1024)[0], phrasing=0, split="train", seq=0)
    with pytest.raises(RuntimeError):
        wf.render(spec, random.Random(0), "zh")


def test_unknown_judge_name_raises() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：直接报错。"""
    with pytest.raises(KeyError):
        browser_workflow.BrowserWorkflow().make_judges(("nope",), "zh")


def test_workflow_declares_what_it_needs() -> None:
    """输入：workflow 的类属性。输出：无。

    预期行为：`needs` 非空（那句话由后端写）、三种题型都声明、判别器名单与默认一致。
    """
    wf = browser_workflow.BrowserWorkflow()
    assert wf.name == "browser"
    assert wf.needs == frozenset({"text"})
    assert wf.kinds == frozenset({"choice", "noul", "score"})
    assert wf.judges == ("hygiene", "frame", "trace")
    assert wf.available_locales() == ["en", "zh"]
    assert wf.split_names() == ("train", "val", "test_iid", "test_ood")


def test_config_for_and_reparse_wire_up() -> None:
    """输入：切分名与一段 state。输出：无。

    预期行为：`config_for` 给出该切分的采样参数、`reparse` 走纯代码反解——
    这两条是通用层与 workflow 的接线点，接错了整条管线就哑了。
    """
    wf = browser_workflow.BrowserWorkflow()
    assert wf.config_for("train").split == "train"
    assert wf.config_for("test_ood").split == "test_ood"
    tags = features.build_cases(1024)[0]
    state = render.to_state({"request": request_for(tags.as_dict())}, tags, locales.load("zh"))
    parsed = wf.reparse(state)
    assert parsed.request == request_for(tags.as_dict())
    assert parsed.elements == render.canonical_elements(tags)


def test_case_label_and_stats_dimensions() -> None:
    """输入：一个 case。输出：无。

    预期行为：`case_label` 一维一格、`stats_dimensions` 五维齐全——统计报告与
    泛化留出检查都读这两条。
    """
    wf = browser_workflow.BrowserWorkflow()
    case = features.build_cases(1024)[0]
    label = wf.case_label(case)
    for name, value in case.as_dict().items():
        assert f"{name}={value}" in label
    assert wf.stats_dimensions(case) == case.as_dict()
    assert set(wf.dimension_titles()) == set(features.DIMS)


def test_spec_to_dict_and_task_ids() -> None:
    """输入：一条真值规格。输出：无。

    预期行为：`spec_to_dict` 可 JSON 序列化且字段齐全；渲染 / 反解任务 id 按同一条
    样本可重算（断点续跑按它查进度）。
    """
    tags = features.build_cases(1024)[0]
    spec = BrowserSpec(tags=tags, phrasing=2, split="train", seq=5)
    assert spec_to_dict(spec) == {
        "tags": tags.as_dict(),
        "phrasing": 2,
        "split": "train",
        "seq": 5,
    }
    assert spec.labels == tags.label()
    assert spec.render_task_id == "browser-train-0000005-render"
    assert spec.trace_task_id == "browser-train-0000005-trace"
