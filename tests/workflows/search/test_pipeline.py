"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from jc import pipeline
from jc import workflow as workflow_mod
from tests.workflows.search.conftest import tiny_config
from workflows.search import judges as judges_mod
from workflows.search import locales, parse, render, spec as sp
from workflows.search import workflow as search_workflow


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="search",
        split=split,
        locales=("zh",),
        run_seed=17,
        per_case=1,
        concurrency=concurrency,
        out_root=tmp_path,
        run_id=run_id,
        config_override=tiny_config(split),
    )


def _case_count(config: pipeline.RunConfig) -> int:
    """输入：运行配置。输出：该切分的 case 数。预期行为：与产物条数对得上。"""
    return len(search_workflow.SearchWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：极小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "s1")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("search").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul", "score"}


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """输入：同种子、不同并发度跑两次。输出：两份产物。

    预期行为：逐字节一致——RNG 由 (seed, workflow, split, seq) 派生，并发不改变内容。
    """
    serial = pipeline.run(make_config(tmp_path, "s2a", concurrency=1))
    parallel = pipeline.run(make_config(tmp_path, "s2b", concurrency=4))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        parallel.run_dir / "train.jsonl"
    ).read_bytes()


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分 / case 列表 / 「不需要后端」；
    stats 报告里出现 8 个维度的中文标题与两个判别器的通过率。
    """
    result = pipeline.run(make_config(tmp_path, "s3"))
    manifest = json.loads((result.run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "search" and manifest["needs"] == []
    assert manifest["split"] == "train" and manifest["cases"]
    stats = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    for title in search_workflow.DIMENSION_TITLES.values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_needs_no_backend_and_rejects_being_bound() -> None:
    """输入：无后端 / 硬塞一个后端。输出：断言 / AssertionError。

    预期行为：纯构造式的 workflow 不初始化任何后端；真被绑定说明接线错了，直接报错。
    """
    workflow = search_workflow.SearchWorkflow()
    assert workflow.needs == frozenset()
    with pytest.raises(AssertionError, match="纯构造式"):
        workflow.bind_backend(object())


def test_config_for_rejects_an_unknown_split() -> None:
    """输入：不存在的切分名。输出：KeyError。预期行为：不猜、不回落。"""
    with pytest.raises(KeyError, match="未知切分"):
        search_workflow.SearchWorkflow().config_for("test_oodd")


def test_split_names_and_slicing_are_disjoint() -> None:
    """输入：同分布三档。输出：三份 case 列表。

    预期行为：三档的规模按 60/20/20 切，且**互不重叠**——`ID_CONFIG` 是单例，
    忘了把 `split` 换掉会让三档切出同一份数据（shell 上实测踩过）。
    """
    workflow = search_workflow.SearchWorkflow()
    slices = {
        split: [case.as_dict() for case in workflow.build_cases(workflow.config_for(split))]
        for split in ("train", "val", "test_iid")
    }
    train, val, iid = (slices["train"], slices["val"], slices["test_iid"])
    assert len(train) > len(val) > 0 and len(iid) > 0
    keyed = lambda rows: {tuple(sorted(item.items())) for item in rows}  # noqa: E731
    assert not (keyed(train) & keyed(val))
    assert not (keyed(val) & keyed(iid))


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _without_dates(text: str) -> str:
    """输入：渲染出来的页面。输出：把更新日期抹成 `DATE` 的文本。

    预期行为：日期抖动由**整个** case 派生（含 `probe`），所以同一素材的不同题面日期不同；
    比「页面」时要先把这一层抹掉，才看得出「素材决定页面」。
    """
    return _DATE_RE.sub("DATE", text)


def test_a_page_never_straddles_two_splits() -> None:
    """输入：同分布三档。输出：无。

    预期行为：**同一份素材（页面）整组落在同一档**——`probe` 只改题面点名的编号、不改页面，
    按整个组合分桶会让同一页同时出现在两档里（实测 1363/1764 的素材横跨两档，见 PLAN 10.48）。
    """
    workflow = search_workflow.SearchWorkflow()
    where: dict[str, set[str]] = {}
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            where.setdefault(sp.material_of(case), set()).add(split)
    assert len(where) > 1000  # 素材确实被拆成了多组，不是「一档一组」
    assert not {key: splits for key, splits in where.items() if len(splits) > 1}


def test_the_material_key_captures_the_whole_page() -> None:
    """输入：同分布三档的全部 case。输出：无。

    预期行为：抹掉日期后，页面文本必须是**素材的函数**——同一素材的不同 `probe` 渲染出同一页，
    不同素材渲染出不同的页。这条锁住 `material_of` 的完整性：以后加一个「只改题面」的维度
    却忘了排除它，分桶就会重新漏页，这里当场炸。
    """
    workflow = search_workflow.SearchWorkflow()
    loc = locales.load("zh")
    by_page: dict[str, set[str]] = {}
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            spec = sp.sample_spec(random.Random(0), case, split, 0, 0)
            page = _without_dates(render.build(spec, loc))
            by_page.setdefault(page, set()).add(sp.material_of(case))
    assert len(by_page) > 1000
    assert not {page: mats for page, mats in by_page.items() if len(mats) > 1}


def test_audit_reports_a_missing_artifact(tmp_path: Path) -> None:
    """输入：空目录。输出：(报告, 缺失项)。

    预期行为：缺失的切分要明说「产物不存在」，而不是当成「0 条样本、审计通过」。
    """
    report, missing = workflow_mod.load("search").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


def test_audit_flags_an_answer_that_the_page_contradicts(tmp_path: Path) -> None:
    """输入：一条被改错的答案。输出：(报告, 缺失项)。

    预期行为：审计把答案搬到产物级重算，所以「老产物」里被改错的答案也会被点名。
    """
    config = make_config(tmp_path, "s5")
    result = pipeline.run(config)
    path = result.run_dir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
    qid = next(qid for qid in rows[0]["questions"] if qid.startswith("answerable"))
    rows[0]["answers"][qid]["noul"] = not rows[0]["answers"][qid]["noul"]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    report, missing = workflow_mod.load("search").audit(result.run_dir, ("train",), "zh")
    assert report["train"]["label_mismatch"] >= 1
    assert any("重算不一致" in item for item in missing)


def test_audit_flags_a_page_that_cannot_be_parsed(tmp_path: Path) -> None:
    """输入：一条被砍坏的 state。输出：(报告, 缺失项)。预期行为：点名反解失败。"""
    config = make_config(tmp_path, "s6")
    result = pipeline.run(config)
    path = result.run_dir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
    rows[0]["state"] = "完全不是一页结果"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    _report, missing = workflow_mod.load("search").audit(result.run_dir, ("train",), "zh")
    assert any("重算不一致" in item or "反解不出来" in item for item in missing)


def test_judges_and_reparse_wire_up_on_a_real_sample() -> None:
    """输入：一条按生产配置生成的样本。输出：无。

    预期行为：两个判别器都通过；`reparse` 能把两种语言都读出来——这是「接线对了」的冒烟。
    """
    case = sp.build_cases(tiny_config())[0]
    spec = sp.sample_spec(__import__("random").Random(0), case, "train", 0, 0)
    workflow = search_workflow.SearchWorkflow()
    for locale in ("zh", "en"):
        loc = locales.load(locale)
        state = render.build(spec, loc)
        page = workflow.reparse(state)
        assert len(page.results) == case.count
        probe, pair = case.probe, case.pair
        expected = judges_mod.recompute(
            parse.parse_page(state, parse.label_index(loc), (loc["meta"]["field_sep"],)),
            loc,
            probe,
            pair,
        )
        assert expected["best_result"] == f"result_{case.best_at}"
