"""最小熔断器：连续 N 次故障就打开，冷却 T 秒后只放**一个**试探。

为什么不直接用 `pybreaker`（PLAN 3.1.3 原本写的库）：
`pybreaker.CircuitBreaker.call()` 把被保护的调用整个放进它自己的 `RLock`
（`pybreaker/__init__.py:250` 的 `with self._lock: return self.state.call(...)`），
于是**同一个 provider 的请求被串行化**。实测（`py-spy dump` 5 路并发的邮件合成）：
5 个 worker 里 4 个卡在那把锁上，只剩 1 个真在发请求——8 条样本跑了 50 分钟，
「并发可配」形同虚设。

熔断真正需要的互斥是**状态判定与状态转换**，不是调用本身，所以这里自己实现：
锁只护住状态机，网络调用在锁外面跑。

状态机（对应 PLAN 的「连续 3 次 / 冷却 60 秒」）：

- `closed`：放行；连续故障攒到 `fail_max` 转 `open`；
- `open`：一律不放行；距打开超过 `reset_timeout_s` 后转 `half_open`；
- `half_open`：只放一个试探请求，成功回 `closed`，失败立刻回 `open`。

输入：`fail_max`、`reset_timeout_s`、`clock`（测试注入用）。
输出：`allow()` / `record_ok()` / `record_fault()` / `state`。
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Final

STATE_CLOSED: Final[str] = "closed"
STATE_OPEN: Final[str] = "open"
STATE_HALF_OPEN: Final[str] = "half_open"


class Breaker:
    """线程安全的熔断器（调用本身不占锁）。

    参数：`fail_max` 连续故障阈值；`reset_timeout_s` 冷却秒数；`clock` 单调时钟。
    """

    def __init__(
        self,
        fail_max: int,
        reset_timeout_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """输入：阈值、冷却时间、时钟。输出：Breaker 实例。预期行为：非法配置当场报错。"""
        if fail_max < 1:
            raise ValueError(f"fail_max 必须 >= 1，得到 {fail_max}")
        if reset_timeout_s < 0:
            raise ValueError(f"reset_timeout_s 不能为负，得到 {reset_timeout_s}")
        self._fail_max = fail_max
        self._reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._state = STATE_CLOSED
        self._faults = 0
        self._opened_at = 0.0
        self._trial_in_flight = False

    @property
    def state(self) -> str:
        """输入：无。输出：当前状态名（顺带推进冷却到点这件事）。"""
        with self._lock:
            self._advance_locked()
            return self._state

    def allow(self) -> bool:
        """输入：无。输出：这次请求放不放行。

        预期行为：`closed` 放行；`open` 到点转 `half_open` 并只放一个试探；
        试探还在飞的时候其余请求继续被拒（半开不许一拥而上）。
        """
        with self._lock:
            self._advance_locked()
            if self._state == STATE_CLOSED:
                return True
            if self._state == STATE_HALF_OPEN and not self._trial_in_flight:
                self._trial_in_flight = True
                return True
            return False

    def record_ok(self) -> None:
        """输入：无。输出：无。预期行为：一次成功就把状态与计数都归零（含试探成功）。"""
        with self._lock:
            self._state = STATE_CLOSED
            self._faults = 0
            self._trial_in_flight = False

    def record_fault(self) -> None:
        """输入：无。输出：无。预期行为：半开时一次失败立刻重开；闭合时攒够阈值才打开。"""
        with self._lock:
            self._trial_in_flight = False
            if self._state == STATE_HALF_OPEN:
                self._open_locked()
                return
            self._faults += 1
            if self._faults >= self._fail_max:
                self._open_locked()

    def _advance_locked(self) -> None:
        """冷却到点就把 `open` 推进成 `half_open`（调用方必须已持锁）。"""
        if self._state != STATE_OPEN:
            return
        if self._clock() - self._opened_at >= self._reset_timeout_s:
            self._state = STATE_HALF_OPEN
            self._trial_in_flight = False

    def _open_locked(self) -> None:
        """打开熔断并重新计时（调用方必须已持锁）。"""
        self._state = STATE_OPEN
        self._opened_at = self._clock()
        self._faults = self._fail_max
