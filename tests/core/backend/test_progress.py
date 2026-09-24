"""断点续跑测试：追加写、残缺尾行容忍、先产物后进度。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc.backend.progress import (
    STATUS_FAILED,
    STATUS_OK,
    ProgressEntry,
    ProgressLog,
    RunSink,
)


def test_append_then_read_round_trip(tmp_path: Path) -> None:
    log = ProgressLog(tmp_path / "progress.jsonl")
    log.append(ProgressEntry(task_id="t-1", provider="nim", row=1))
    log.append(ProgressEntry(task_id="t-2", provider="nim", row=2))
    assert [e.task_id for e in log.entries()] == ["t-1", "t-2"]


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert ProgressLog(tmp_path / "none.jsonl").entries() == []


def test_truncated_last_line_is_dropped(tmp_path: Path) -> None:
    """追加写 + fsync 的性质：只有尾行可能残缺，残缺就当它没写过。"""
    path = tmp_path / "progress.jsonl"
    log = ProgressLog(path)
    log.append(ProgressEntry(task_id="t-1"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"task_id": "t-2", "stat')
    assert [e.task_id for e in log.entries()] == ["t-1"]


def test_corruption_in_the_middle_raises(tmp_path: Path) -> None:
    """中间行坏了说明文件真的损坏，必须报人话，不许静默跳过。"""
    path = tmp_path / "progress.jsonl"
    path.write_text('{broken\n{"task_id": "t-2"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 1 行坏了"):
        ProgressLog(path).entries()


def test_completed_keeps_last_and_skips_failed(tmp_path: Path) -> None:
    log = ProgressLog(tmp_path / "progress.jsonl")
    log.append(ProgressEntry(task_id="t-1", status=STATUS_FAILED))
    log.append(ProgressEntry(task_id="t-1", status=STATUS_OK, row=1))
    log.append(ProgressEntry(task_id="t-2", status=STATUS_FAILED))
    done = log.completed()
    assert set(done) == {"t-1"}
    assert done["t-1"].row == 1


def test_sink_numbers_rows_and_orders_product_before_progress(tmp_path: Path) -> None:
    sink = RunSink(tmp_path / "data.jsonl", tmp_path / "progress.jsonl")
    first = sink.write_sample({"id": "a"}, ProgressEntry(task_id="t-1"))
    second = sink.write_sample({"id": "b"}, ProgressEntry(task_id="t-2"))
    assert (first.row, second.row) == (1, 2)
    assert sink.rows == 2
    assert sink.dataset_path.read_text(encoding="utf-8").count("\n") == 2
    assert sink.progress.completed()["t-2"].row == 2


def test_sink_does_not_write_progress_when_product_write_fails(tmp_path: Path) -> None:
    """顺序铁律的反证：产物写不进去时，进度里绝不能出现这一条。"""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    sink = RunSink(blocked, tmp_path / "progress.jsonl")
    with pytest.raises(OSError):
        sink.write_sample({"id": "a"}, ProgressEntry(task_id="t-1"))
    assert sink.progress.entries() == []


def test_sink_resumes_row_numbering_from_existing_file(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text('{"id":"old"}\n', encoding="utf-8")
    sink = RunSink(data, tmp_path / "progress.jsonl")
    assert sink.rows == 1
    assert sink.write_sample({"id": "new"}, ProgressEntry(task_id="t-9")).row == 2


def test_progress_path_is_where_it_was_told_to_write(tmp_path: Path) -> None:
    """进度文件路径要能读出来，出问题时用户才知道去哪儿看。"""
    target = tmp_path / "nested" / "progress.jsonl"
    log = ProgressLog(target)
    assert log.path == target
    assert target.parent.is_dir()  # 父目录自动建


def test_entries_skip_blank_lines(tmp_path: Path) -> None:
    """手改进度文件时留下的空行不该被当成坏行。"""
    target = tmp_path / "progress.jsonl"
    log = ProgressLog(target)
    log.append(ProgressEntry(task_id="t-1"))
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n")
    log.append(ProgressEntry(task_id="t-2"))
    assert [e.task_id for e in log.entries()] == ["t-1", "t-2"]
