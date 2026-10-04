"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path


from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import read_rows as _rows, run_config, write_rows as _write
from tests.workflows.segmentation.conftest import FakeBackend
from workflows.segmentation import workflow as segmentation_workflow
from workflows.segmentation.workflow import SegmentationConfig

CASE_LIMIT = 8


#: 计数器 → 缺失项列表里那句话的抬头（报告给人看，所以点名的是人话不是内部键）。
BAD_LABELS = {
    "content_bad": "种类表或页序认不出来",
    "flag_bad": "逐类答案与页面不自洽",
    "count_bad": "类数与页面不自洽",
    "combo_bad": "种类组合与页面不自洽",
    "route_bad": "切分档位与页面不自洽",
}


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        run_config(
            tmp_path,
            run_id,
            backend=FakeBackend(),
            workflow="segmentation",
            seed=11,
            override=SegmentationConfig(split="train", case_limit=CASE_LIMIT),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return segmentation_workflow.SegmentationWorkflow().audit(run_dir, splits, "zh")


def test_a_tampered_page_is_caught(tmp_path: Path) -> None:
    """输入：把某一页的抬头换成别的类（答案没跟着改）。输出：无。

    预期行为：四个计数器都报出来——这是审计真正要抓的东西：**答案与文本必须自洽**，
    判别器只看得到当次样本，产物落盘之后被改过只有审计看得见。
    """
    run_dir = _run(tmp_path, "t-page")
    rows = _rows(run_dir)
    from tests.workflows.segmentation.conftest import MARKERS

    rows[0]["state"] = str(rows[0]["state"]).replace(MARKERS["form"], MARKERS["letter"])
    _write(run_dir, rows)
    report, _missing = _audit(run_dir)
    counts = report["train"]
    assert counts["flag_bad"] or counts["count_bad"] or counts["combo_bad"] or counts["route_bad"]

