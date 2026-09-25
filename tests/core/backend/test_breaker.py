"""熔断器测试（`jc.backend.breaker`）。

重点是最后那条并发测试：**它守的是「熔断器不许把调用串行化」这件事**。
换掉 pybreaker 的原因就是它的 `call()` 把被保护的调用锁在自己的 RLock 里，
5 路并发实测只剩 1 个请求在飞——那条测试在旧实现下会直接失败。
"""

from __future__ import annotations

import threading
import time

import pytest

from jc.backend.breaker import STATE_CLOSED, STATE_HALF_OPEN, STATE_OPEN, Breaker


class Clock:
    """可控时钟：测试里手动推进时间，不真等冷却。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        """输入：无。输出：当前秒数。"""
        return self.now


def test_rejects_bad_config() -> None:
    """输入：fail_max=0 / 负冷却。输出：ValueError。预期行为：非法配置当场报错。"""
    with pytest.raises(ValueError, match="fail_max"):
        Breaker(0, 1.0)
    with pytest.raises(ValueError, match="reset_timeout_s"):
        Breaker(1, -1.0)


def test_opens_after_fail_max_consecutive_faults() -> None:
    """连续攒够 fail_max 次故障才打开，之前一律放行。"""
    breaker = Breaker(3, 60.0, clock=Clock())
    assert [breaker.allow() for _ in range(2)] == [True, True]
    breaker.record_fault()
    breaker.record_fault()
    assert breaker.state == STATE_CLOSED
    breaker.record_fault()
    assert breaker.state == STATE_OPEN
    assert breaker.allow() is False


def test_one_success_resets_the_counter() -> None:
    """「连续」是字面意思：中间成功一次就把计数清零。"""
    breaker = Breaker(3, 60.0, clock=Clock())
    breaker.record_fault()
    breaker.record_fault()
    breaker.record_ok()
    breaker.record_fault()
    breaker.record_fault()
    assert breaker.state == STATE_CLOSED


def test_half_open_lets_exactly_one_trial_through() -> None:
    """冷却到点转半开，且只放一个试探——不许一拥而上。"""
    clock = Clock()
    breaker = Breaker(1, 60.0, clock=clock)
    breaker.record_fault()
    assert breaker.state == STATE_OPEN
    clock.now = 60.0
    assert breaker.state == STATE_HALF_OPEN
    assert breaker.allow() is True
    assert breaker.allow() is False  # 试探还在飞
    breaker.record_ok()
    assert breaker.state == STATE_CLOSED
    assert breaker.allow() is True


def test_half_open_failure_reopens_the_breaker() -> None:
    """试探失败立刻重新打开，并且重新计时（不会疯狂重试）。"""
    clock = Clock()
    breaker = Breaker(1, 60.0, clock=clock)
    breaker.record_fault()
    clock.now = 60.0
    assert breaker.allow() is True
    breaker.record_fault()
    assert breaker.state == STATE_OPEN
    clock.now = 119.0
    assert breaker.allow() is False
    clock.now = 120.0
    assert breaker.allow() is True


def test_calls_are_not_serialised() -> None:
    """6 个线程同时「允许 → 干活 → 记账」，必须真的并行。

    输入：6 个线程各占 0.15 秒。
    输出：峰值并发数 6、总耗时明显小于串行的 0.9 秒。
    预期行为：熔断器只在状态判定上互斥；一旦它把调用也锁住（pybreaker 就是这样），
    峰值会掉到 1、耗时翻成 6 倍——这条测试就是那个回归的守门员。
    """
    breaker = Breaker(3, 60.0)
    threads_n = 6
    barrier = threading.Barrier(threads_n)
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def worker() -> None:
        """输入：无。输出：无。预期行为：占着名额睡 0.15 秒，模拟一次网络调用。"""
        nonlocal in_flight, peak
        barrier.wait()
        assert breaker.allow()
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.15)
        with lock:
            in_flight -= 1
        breaker.record_ok()

    threads = [threading.Thread(target=worker) for _ in range(threads_n)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - started

    assert peak == threads_n, f"熔断器把调用串行化了：峰值并发只有 {peak}"
    assert elapsed < 0.6, f"耗时 {elapsed:.2f}s 接近串行（0.9s），说明调用被锁住了"
