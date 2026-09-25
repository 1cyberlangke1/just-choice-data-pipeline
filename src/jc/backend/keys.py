"""多 key 负载均衡（PLAN 3.1.3）。

**这一块没有现成的库**：`pyrate-limiter` 管速率、`breaker` 管熔断，都不管
「同一个端点的 N 个 key 里挑哪个」。所以这里手写，而且刻意写小。

调度策略：**最少在途优先**（并发下比轮询均衡），平票时按轮转游标取，避免总压第一个 key。
失效的 key 永久剔除；池子空了抛 `AuthError`，由上层决定顺延下一家 provider。

输入/输出：`KeyPool.slot()` 是上下文管理器，产出本次调用要用的 key，退出时归还名额。
预期行为：日志里只允许出现 key 的指纹，**绝不出现明文**。
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from .types import AuthError


def fingerprint(key: str) -> str:
    """输入：明文 key。输出：sha256 前 8 位。预期行为：可安全写进日志与产物元数据。"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


class KeyPool:
    """一个 provider 的 key 池（线程安全）。

    参数：`keys` 全部 key（保序）；`max_in_flight_per_key` 单 key 在途上限（PLAN 3.1.6）。
    """

    def __init__(self, keys: Sequence[str], max_in_flight_per_key: int = 1) -> None:
        """输入：key 序列与单 key 在途上限。输出：KeyPool 实例。

        预期行为：空池或非法上限直接报错，不让管线带着坏配置往下跑。
        """
        if not keys:
            raise ValueError("key 池不能为空")
        if max_in_flight_per_key < 1:
            raise ValueError(f"max_in_flight_per_key 必须 >= 1，得到 {max_in_flight_per_key}")
        self._keys: list[str] = list(keys)
        self._limit = max_in_flight_per_key
        self._in_flight: dict[str, int] = {k: 0 for k in self._keys}
        self._invalid: set[str] = set()
        self._cursor = 0
        self._cond = threading.Condition()

    @property
    def size(self) -> int:
        """key 总数（含已失效的）。"""
        return len(self._keys)

    def available(self) -> int:
        """当前可用（未失效）的 key 数。"""
        with self._cond:
            return len(self._keys) - len(self._invalid)

    def _choose_locked(self) -> str | None:
        """在持锁状态下挑一个可用且未到在途上限的 key。

        输入：无（调用方必须已持有 `self._cond`）。
        输出：选中的 key，或 None（当前没有空位）。
        预期行为：最少在途优先；平票时从轮转游标开始扫，选中后游标前移。
        """
        candidates = [
            k for k in self._keys if k not in self._invalid and self._in_flight[k] < self._limit
        ]
        if not candidates:
            return None
        fewest = min(self._in_flight[k] for k in candidates)
        tied = {k for k in candidates if self._in_flight[k] == fewest}
        total = len(self._keys)
        for offset in range(total):
            key = self._keys[(self._cursor + offset) % total]
            if key in tied:
                self._cursor = (self._keys.index(key) + 1) % total
                self._in_flight[key] += 1
                return key
        return None  # pragma: no cover —— tied 非空时不可能走到

    @contextmanager
    def slot(self) -> Iterator[str]:
        """输入：无。输出：上下文管理器，产出本次调用独占名额的 key。

        预期行为：全部 key 失效时抛 `AuthError`；否则阻塞到有空位再产出。
        """
        with self._cond:
            while True:
                if len(self._keys) - len(self._invalid) == 0:
                    raise AuthError(f"key 池已空：{len(self._invalid)}/{len(self._keys)} 个 key 失效")
                chosen = self._choose_locked()
                if chosen is not None:
                    break
                self._cond.wait()
        try:
            yield chosen
        finally:
            with self._cond:
                self._in_flight[chosen] = max(0, self._in_flight[chosen] - 1)
                self._cond.notify_all()

    def mark_invalid(self, key: str) -> None:
        """输入：一个 key。输出：无。预期行为：永久剔除并唤醒等待者；重复标记无副作用。"""
        with self._cond:
            self._invalid.add(key)
            self._cond.notify_all()

    def snapshot(self) -> dict[str, int]:
        """返回 {key 指纹: 在途数}，供日志与调试（不含明文）。"""
        with self._cond:
            return {
                fingerprint(k): n for k, n in self._in_flight.items() if k not in self._invalid
            }
