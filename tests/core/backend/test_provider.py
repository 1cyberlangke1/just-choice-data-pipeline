"""provider 的运行态测试：熔断语义、限速自适应、key 剔除、并发闸门、重试退避。

全部用 `FakeEngine` 回放异常，**不发网络请求**；睡眠函数注入空实现，测试不真等。
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any

import pytest

from jc.backend.config import Limits
from jc.backend.keys import fingerprint
from jc.backend.provider import ProviderChain, _is_not_fault
from jc.backend.types import (
    AuthError,
    BackendCapabilities,
    DraftInvalidError,
    ProviderUnavailable,
    RateLimitError,
    SchemaUnsupportedError,
    TransportError,
)

#: 默认关掉限速与重试，只留被测的那一项，免得测试被限速拖慢。
_BASE = Limits(requests_per_minute=0, min_interval_ms=0, max_attempts=1)


def limits(**overrides: Any) -> Limits:
    """输入：限额字段覆盖项。输出：Limits。预期行为：在「全关」的基线上只改指定字段。"""
    return replace(_BASE, **overrides)


def test_only_counted_faults_are_faults() -> None:
    """熔断只该被「provider 真的坏了」触发；模型答错和我们自己的 bug 都不算。"""
    assert _is_not_fault(DraftInvalidError("答非所问")) is True
    assert _is_not_fault(SchemaUnsupportedError("400")) is True
    assert _is_not_fault(ValueError("认不出的异常")) is True
    assert _is_not_fault(TransportError("500")) is False
    assert _is_not_fault(RateLimitError("429")) is False


def test_transport_failure_is_retried_until_success(
    fake_engine: Any, provider_factory: Any, result_factory: Any, task_factory: Any
) -> None:
    """网络抖动要重试：两次失败后成功，尝试次数必须是 3。"""
    engine = fake_engine(script=[TransportError("连接超时"), TransportError("连接超时"), result_factory()])
    provider = provider_factory(engine, limits=limits(max_attempts=3))
    result, key_fp, attempts = provider.extract(task_factory(), "json_schema")
    assert attempts == 3
    assert key_fp == fingerprint("k1")
    assert result.data == {"answer": "x"}


def test_transport_failure_is_raised_after_attempts_run_out(
    fake_engine: Any, provider_factory: Any, task_factory: Any
) -> None:
    """重试用尽要原样抛出最后一个异常，不能吞掉。"""
    engine = fake_engine(script=[TransportError("500")] * 4)
    provider = provider_factory(engine, limits=limits(max_attempts=2))
    with pytest.raises(TransportError):
        provider.extract(task_factory(), "json_schema")
    assert len(engine.calls) == 2


def test_schema_unsupported_is_not_retried(
    fake_engine: Any, provider_factory: Any, task_factory: Any
) -> None:
    """「服务端不吃这一档」不是故障，重试同一档纯属浪费——交给上层降档。"""
    engine = fake_engine(script=[SchemaUnsupportedError("400")] * 3)
    provider = provider_factory(engine, limits=limits(max_attempts=3))
    with pytest.raises(SchemaUnsupportedError):
        provider.extract(task_factory(), "json_schema")
    assert len(engine.calls) == 1


def test_model_misbehaviour_never_opens_the_breaker(
    fake_engine: Any, provider_factory: Any, task_factory: Any
) -> None:
    """连续 4 次「模型答非所问」也不许熔断（fail_max=2），否则弱模型会被误杀。"""
    engine = fake_engine(script=[DraftInvalidError("答非所问")] * 4)
    provider = provider_factory(engine, limits=limits(fail_max=2))
    for _ in range(4):
        with pytest.raises(DraftInvalidError):
            provider.extract(task_factory(), "json_schema")
    assert provider.breaker_state == "closed"


def test_transport_failures_do_open_the_breaker(
    fake_engine: Any, provider_factory: Any, task_factory: Any
) -> None:
    """连续 fail_max 次真故障要打开熔断，之后直接报「不可用」而**不发请求**。

    注意 pybreaker 的 `throw_new_error_on_trip`：第 fail_max 次失败本身就抛
    `CircuitBreakerError`（被我们包成 `ProviderUnavailable`），而不是抛原始异常。
    """
    engine = fake_engine(script=[TransportError("500")] * 5)
    provider = provider_factory(engine, limits=limits(fail_max=3))
    for _ in range(2):
        with pytest.raises(TransportError):
            provider.extract(task_factory(), "json_schema")
    assert provider.breaker_state == "closed"
    with pytest.raises(ProviderUnavailable):
        provider.extract(task_factory(), "json_schema")  # 第三次触到阈值，当场熔断
    assert provider.breaker_state == "open"
    with pytest.raises(ProviderUnavailable):
        provider.extract(task_factory(), "json_schema")
    assert len(engine.calls) == 3  # 熔断打开后一次请求都没再发出去


def test_rate_limit_halves_effective_rpm(
    fake_engine: Any, provider_factory: Any, task_factory: Any
) -> None:
    """吃到 429 就把 RPM 减半——服务商不回限速头，只能自己收敛。"""
    engine = fake_engine(script=[RateLimitError("429")])
    provider = provider_factory(engine, limits=limits(requests_per_minute=40))
    with pytest.raises(RateLimitError):
        provider.extract(task_factory(), "json_schema")
    assert provider.effective_rpm == 20


def test_success_eases_rpm_back_towards_the_cap(
    fake_engine: Any, provider_factory: Any, result_factory: Any, task_factory: Any
) -> None:
    """降过速之后每次成功抬一点，但绝不超过配置值。"""
    engine = fake_engine(script=[RateLimitError("429"), result_factory()])
    provider = provider_factory(engine, limits=limits(requests_per_minute=40, max_attempts=2))
    provider.extract(task_factory(), "json_schema")
    assert 20 < provider.effective_rpm <= 40


def test_retry_after_wins_over_computed_backoff(
    fake_engine: Any, provider_factory: Any, result_factory: Any, task_factory: Any
) -> None:
    """429 带 Retry-After 时以服务端要求为准，不跟服务端讨价还价。"""
    sleeps: list[float] = []
    engine = fake_engine(script=[RateLimitError("429", retry_after_s=7.0), result_factory()])
    provider = provider_factory(
        engine,
        limits=limits(requests_per_minute=40, max_attempts=2),
        sleep=sleeps.append,
    )
    provider.extract(task_factory(), "json_schema")
    assert sleeps == [7.0]


def test_auth_error_retires_the_key(
    fake_engine: Any, provider_factory: Any, result_factory: Any, task_factory: Any
) -> None:
    """401 只说明这个 key 废了，不是 provider 挂了：换下一个 key 继续。"""
    engine = fake_engine(script=[AuthError("401"), result_factory()])
    provider = provider_factory(engine, keys=("k1", "k2"), limits=limits())
    with pytest.raises(AuthError):
        provider.extract(task_factory(), "json_schema")
    assert provider.available is True
    _, key_fp, _ = provider.extract(task_factory(), "json_schema")
    assert key_fp == fingerprint("k2")


def test_pool_empty_raises_auth_error(
    fake_engine: Any, provider_factory: Any, task_factory: Any
) -> None:
    """唯一的 key 废掉后，这家就没戏了，报错要说人话。"""
    engine = fake_engine(script=[AuthError("401")])
    provider = provider_factory(engine, keys=("k1",), limits=limits())
    with pytest.raises(AuthError):
        provider.extract(task_factory(), "json_schema")
    assert provider.available is False
    with pytest.raises(AuthError, match="key 池已空"):
        provider.extract(task_factory(), "json_schema")


def test_modes_follow_configuration(fake_engine: Any, provider_factory: Any) -> None:
    """写死档位就用它；auto 有探测结果就从探测档起步，没探测就从最严的起步。"""
    engine = fake_engine(script=[])
    assert provider_factory(engine, mode="tools").modes() == ("tools", "md_json")
    assert provider_factory(engine).modes() == ("json_schema", "tools", "md_json")
    probed = provider_factory(engine, capabilities=BackendCapabilities(tools=True))
    assert probed.modes() == ("tools", "md_json")


def test_concurrency_gate_serialises_calls(provider_factory: Any, result_factory: Any, task_factory: Any) -> None:
    """max_concurrency=1 时，两个线程同时来也只有一个能进引擎。"""

    class Probe:
        """记录「同时在跑的调用条数」的假引擎。"""

        name = "probe"

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.active = 0
            self.peak = 0

        def extract(self, request: Any) -> Any:
            """输入：一次请求。输出：成功结果。预期行为：记录并发峰值后睡一会儿。"""
            with self._lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.05)
            with self._lock:
                self.active -= 1
            return result_factory()

    probe = Probe()
    provider = provider_factory(probe, keys=("k1", "k2"), limits=limits(max_concurrency=1))
    barrier = threading.Barrier(2)

    def worker() -> None:
        """输入：无。输出：无。预期行为：和另一个线程同时冲进 extract。"""
        barrier.wait()
        provider.extract(task_factory(), "json_schema")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert probe.peak == 1


def test_chain_rejects_empty_list() -> None:
    """空链是配置错误，早报比晚报好。"""
    with pytest.raises(ValueError, match="provider 链不能为空"):
        ProviderChain([])


def test_chain_iterates_in_configuration_order(fake_engine: Any, provider_factory: Any) -> None:
    """降级顺序 = 配置顺序，遍历链必须按这个顺序来。"""
    chain = ProviderChain(
        [
            provider_factory(fake_engine(script=[]), name="a"),
            provider_factory(fake_engine(script=[]), name="b"),
        ]
    )
    assert [p.name for p in chain] == ["a", "b"]


def test_min_interval_gate_spaces_out_calls(
    fake_engine: Any, provider_factory: Any, result_factory: Any, task_factory: Any
) -> None:
    """同一个 key 的两次请求之间必须隔开配置的最小间隔，否则免费档会被打爆。"""
    engine = fake_engine(script=[result_factory(), result_factory()])
    provider = provider_factory(engine, limits=limits(min_interval_ms=200))
    started = time.monotonic()
    provider.extract(task_factory(), "json_schema")
    provider.extract(task_factory(), "json_schema")
    assert time.monotonic() - started >= 0.15  # 留点余量，不跟调度器较劲


def test_chain_usable_filters_by_model_when_mixed_is_off(fake_engine: Any, provider_factory: Any) -> None:
    """不允许混用时，只保留与第一家 model 相同的 provider，免得数据分布被换模型污染。"""
    first = provider_factory(fake_engine(script=[]), name="a", model="m1")
    second = provider_factory(fake_engine(script=[]), name="b", model="m2")
    chain = ProviderChain([first, second], allow_mixed=False)
    assert chain.names == ["a", "b"]
    assert len(chain) == 2
    assert [p.name for p in chain.usable()] == ["a"]
    assert [p.name for p in ProviderChain([first, second], allow_mixed=True).usable()] == ["a", "b"]
