"""断点续跑：追加写的进度日志 + 产物落盘（PLAN 3.1.5）。

**这一块没有现成的库**：要的是「产物先落盘、进度后落盘」这个顺序保证，加上
「重启后按 task_id 跳过已完成」，通用缓存库不提供这种语义。

两条铁律：
1. **先写产物（数据集 jsonl），再写进度**。反过来会在崩溃时留下「进度说完成、产物没落盘」，
   而正着来最坏只是重做一条；
2. 进度文件**只追加不改**，每行写完 flush + fsync。这样中途 Ctrl-C 最多留下半行，
   解析失败的那行直接丢弃，当它没写过。

输入/输出：`ProgressLog.completed()` 读出已完成的 task_id；`RunSink.write_sample()`
写一条产物并回填行号。预期行为：并发调用由内部锁串行化，绝不交错写坏行。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import os
import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


from .. import jsonio

STATUS_OK: str = "ok"
STATUS_INVALID: str = "invalid"
STATUS_FAILED: str = "failed"

# 只有这些状态算「已完成、重启可跳过」；failed 必须重做。
DONE_STATUSES: frozenset[str] = frozenset({STATUS_OK, STATUS_INVALID})


@dataclass(frozen=True)
class ProgressEntry:
    """进度日志的一行（PLAN 3.1.5 的字段表）。

    字段：`row` 产物文件里的行号（1 起；没写产物时为 0）；`key_fingerprint` 是 key 的
    sha256 前 8 位，**不存明文**。
    """

    task_id: str
    status: str = STATUS_OK
    row: int = 0
    provider: str = ""
    model: str = ""
    mode: str = ""
    key_fingerprint: str = ""
    attempts: int = 0
    elapsed_s: float = 0.0
    usage: dict[str, JsonValue] = field(default_factory=dict)


def _flush_sync(handle: object) -> None:
    """输入：一个已打开的文件句柄。输出：无。预期行为：flush + fsync，断电也不丢这一行。"""
    handle.flush()
    os.fsync(handle.fileno())


class ProgressLog:
    """只追加的进度日志（线程安全）。"""

    def __init__(self, path: str | Path) -> None:
        """输入：进度文件路径。输出：ProgressLog 实例。预期行为：父目录自动建。"""
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """进度文件路径。"""
        return self._path

    def entries(self) -> list[ProgressEntry]:
        """输入：无。输出：已写下的全部进度行（按文件顺序）。

        预期行为：文件不存在返回空列表；**只有最后一行允许残缺**（追加写 + fsync 的性质），
        残缺则丢弃；中间某行解析失败说明文件真的坏了，报人话而不是静默跳过。
        """
        if not self._path.is_file():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        out: list[ProgressEntry] = []
        known = set(ProgressEntry.__dataclass_fields__)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                raw = jsonio.loads(line)
            except jsonio.StrictJsonError as exc:
                if index == len(lines) - 1:
                    break  # 崩在写一半的尾行，当它没写过
                raise ValueError(f"进度文件第 {index + 1} 行坏了：{exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"进度文件第 {index + 1} 行不是 JSON 对象：{raw!r}")
            out.append(ProgressEntry(**{k: v for k, v in raw.items() if k in known}))
        return out

    def completed(self) -> dict[str, ProgressEntry]:
        """输入：无。输出：{task_id: 最后一次进度}，只含 `DONE_STATUSES` 里的状态。

        预期行为：同一 task_id 出现多次时以最后一次为准（重跑过就该以新结果为准）。
        """
        latest: dict[str, ProgressEntry] = {}
        for entry in self.entries():
            latest[entry.task_id] = entry
        return {k: v for k, v in latest.items() if v.status in DONE_STATUSES}

    def append(self, entry: ProgressEntry) -> None:
        """输入：一条进度。输出：无。预期行为：追加一行紧凑 JSON 并 flush + fsync。"""
        line = jsonio.dumps_line(asdict(entry))
        with self._lock:
            with self._path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.write("\n")
                _flush_sync(handle)


class RunSink:
    """产物 + 进度的唯一写入口。

    参数：`dataset_path` 数据集 jsonl；`progress_path` 进度日志。
    预期行为：内部锁把并发调用串行化——多个线程直接写同一个 jsonl 会交错写坏行。
    """

    def __init__(self, dataset_path: str | Path, progress_path: str | Path) -> None:
        """输入：产物路径与进度路径。输出：RunSink 实例。预期行为：按已有产物行数续编号。"""
        self._dataset = Path(dataset_path)
        self._dataset.parent.mkdir(parents=True, exist_ok=True)
        self.progress = ProgressLog(progress_path)
        self._rows = self._count_rows()
        self._lock = threading.Lock()

    def _count_rows(self) -> int:
        """输入：无。输出：产物文件现有的非空行数。预期行为：文件不存在算 0。"""
        if not self._dataset.is_file():
            return 0
        text = self._dataset.read_text(encoding="utf-8")
        return sum(1 for line in text.splitlines() if line.strip())

    @property
    def dataset_path(self) -> Path:
        """产物文件路径。"""
        return self._dataset

    @property
    def rows(self) -> int:
        """当前已写下的产物行数。"""
        with self._lock:
            return self._rows

    def write_sample(self, sample: dict[str, JsonValue], entry: ProgressEntry) -> ProgressEntry:
        """输入：一条契约样本、一条进度（`row` 由本方法回填）。输出：回填了行号的进度。

        预期行为：**先写产物并 fsync，再写进度**；两步都在同一把锁里，顺序不会被并发打乱。
        """
        with self._lock:
            with self._dataset.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(jsonio.dumps_line(sample))
                handle.write("\n")
                _flush_sync(handle)
            self._rows += 1
            stamped = replace(entry, row=self._rows)
            self.progress.append(stamped)
            return stamped
