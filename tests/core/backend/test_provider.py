"""provider 的运行态测试：熔断语义、限速自适应、key 剔除、并发闸门、重试退避。

全部用 `FakeEngine` 回放异常，**不发网络请求**；睡眠函数注入空实现，测试不真等。
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace


import pytest

from jc.backend.config import Limits
from jc.backend.engine import EngineResult
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
_BASE = Limits(
    requests_per_minute=0, min_interval_ms=0, max_attempts=1, rate_limit_attempts=1
)


class SlowEngine:
    """会睡的假引擎：用来量「同一家 provider 能不能同时发多个请求」。"""

    name = "slow"

    def __init__(self, delay: float) -> None:
        """输入：每次调用睡多久。输出：SlowEngine 实例。"""
        self.delay = delay
        self.peak = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def extract(self, request: object) -> object:
        """输入：一次请求。输出：成功结果。预期行为：记下峰值并发再返回。"""
        with self._lock:
            self._in_flight += 1
            self.peak = max(self.peak, self._in_flight)
        time.sleep(self.delay)
        with self._lock:
            self._in_flight -= 1
        return EngineResult(
            data={"answer": "x"},
            raw_text='{"answer":"x"}',
            usage={},
            reasoning=None,
            mode="json_schema",
        )


def test_provider_does_not_serialise_calls(provider_factory: object, task_factory: object) -> None:
    """3 个线程同时 `extract`，必须真的并行。

    输入：3 个 key、并发上限 5、每次调用占 0.15 秒。
    输出：引擎观测到的峰值并发 = 3，总耗时远小于串行的 0.45 秒。
    预期行为：**这条守的是「熔断器不许锁住调用」**——旧实现用 pybreaker，
    它的 `call()` 把网络调用放在自己的 RLock 里，这里峰值会掉到 1、耗时变成 3 倍。
    """
    engine = SlowEngine(0.15)
    provider = provider_factory(
        engine,
        keys=("k1", "k2", "k3"),
        limits=limits(max_concurrency=5, max_concurrency_per_key=1),
    )
    barrier = threading.Barrier(3)

    def worker() -> None:
        """输入：无。输出：无。预期行为：等到三个线程都就位再一起发。"""
        barrier.wait()
        assert provider.extract(task_factory(), "json_schema")[1]  # key 指纹非空 = 真跑通了

    threads = [threading.Thread(target=worker) for _ in range(3)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - started

    assert engine.peak == 3, f"请求被串行化了：峰值并发只有 {engine.peak}"
    assert elapsed < 0.4, f"耗时 {elapsed:.2f}s 接近串行（0.45s），说明调用被锁住了"


def limits(**overrides: object) -> Limits:
    """输入：限额字段覆盖项。输出：Limits。预期行为：在「全关」的基线上只改指定字段。"""
    return replace(_BASE, **overrides)


def test_only_counted_faults_are_faults() -> None:
    """熔断只该被「provider 真的坏了」触发；模型答错和我们自己的 bug 都不算。"""
    assert _is_not_fault(DraftInvalidError("答非所问")) is True
    assert _is_not_fault(SchemaUnsupportedError("400")) is True
    assert _is_not_fault(ValueError("认不出的异常")) is True
    assert _is_not_fault(TransportError("500")) is False
    assert _is_not_fault(RateLimitError("429")) is True  # 限流不是故障：它有自己的退避


def test_transport_failure_is_retried_until_success(
    fake_engine: object, provider_factory: object, result_factory: object, task_factory: object
) -> None:
    """网络抖动要重试：两次失败后成功，尝试次数必须是 3。"""
    engine = fake_engine(script=[TransportError("连接超时"), TransportError("连接超时"), result_factory()])
    provider = provider_factory(engine, limits=limits(max_attempts=3))
    result, key_fp, attempts = provider.extract(task_factory(), "json_schema")
    assert attempts == 3
    assert key_fp == fingerprint("k1")
    assert result.data == {"answer": "x"}


def test_transport_failure_is_raised_after_attempts_run_out(
    fake_engine: object, provider_factory: object, task_factory: object
) -> None:
    """重试用尽要原样抛出最后一个异常，不能吞掉。"""
    engine = fake_engine(script=[TransportError("500")] * 4)
    provider = provider_factory(engine, limits=limits(max_attempts=2))
    with pytest.raises(TransportError):
        provider.extract(task_factory(), "json_schema")
    assert len(engine.calls) == 2


def test_schema_unsupported_is_not_retried(
    fake_engine: object, provider_factory: object, task_factory: object
) -> None:
    """「服务端不吃这一档」不是故障，重试同一档纯属浪费——交给上层降档。"""
    engine = fake_engine(script=[SchemaUnsupportedError("400")] * 3)
    provider = provider_factory(engine, limits=limits(max_attempts=3))
    with pytest.raises(SchemaUnsupportedError):
        provider.extract(task_factory(), "json_schema")
    assert len(engine.calls) == 1


def test_model_misbehaviour_never_opens_the_breaker(
    fake_engine: object, provider_factory: object, task_factory: object
) -> None:
    """连续 4 次「模型答非所问」也不许熔断（fail_max=2），否则弱模型会被误杀。"""
    engine = fake_engine(script=[DraftInvalidError("答非所问")] * 4)
    provider = provider_factory(engine, limits=limits(fail_max=2))
    for _ in range(4):
        with pytest.raises(DraftInvalidError):
            provider.extract(task_factory(), "json_schema")
    assert provider.breaker_state == "closed"


def test_transport_failures_do_open_the_breaker(
    fake_engine: object, provider_factory: object, task_factory: object
) -> None:
    """连续 fail_max 次真故障要打开熔断，之后直接报「不可用」而**不发请求**。

    第 fail_max 次失败**原样抛它自己的异常**（真实原因不能被吃掉），
    从下一次起才被熔断闸门挡住——这样日志里始终能看到「上一次到底为什么挂」。
    """
    engine = fake_engine(script=[TransportError("500")] * 5)
    provider = provider_factory(engine, limits=limits(fail_max=3))
    for _ in range(3):
        with pytest.raises(TransportError):
            provider.extract(task_factory(), "json_schema")
    assert provider.breaker_state == "open"
    with pytest.raises(ProviderUnavailable, match="正在熔断"):
        provider.extract(task_factory(), "json_schema")
    assert provider.breaker_state == "open"
    assert len(engine.calls) == 3  # 熔断打开后一次请求都没再发出去


def test_rate_limit_halves_effective_rpm(
    fake_engine: object, provider_factory: object, task_factory: object
) -> None:
    """吃到 429 就把 RPM 减半——服务商不回限速头，只能自己收敛。"""
    engine = fake_engine(script=[RateLimitError("429")])
    provider = provider_factory(engine, limits=limits(requests_per_minute=40))
    with pytest.raises(RateLimitError):
        provider.extract(task_factory(), "json_schema")
    assert provider.effective_rpm == 20


def test_success_eases_rpm_back_towards_the_cap(
    fake_engine: object, provider_factory: object, result_factory: object, task_factory: object
) -> None:
    """降过速之后每次成功抬一点，但绝不超过配置值。"""
    engine = fake_engine(script=[RateLimitError("429"), result_factory()])
    provider = provider_factory(
        engine, limits=limits(requests_per_minute=40, max_attempts=2, rate_limit_attempts=2)
    )
    provider.extract(task_factory(), "json_schema")
    assert 20 < provider.effective_rpm <= 40


def test_retry_after_wins_over_computed_backoff(
    fake_engine: object, provider_factory: object, result_factory: object, task_factory: object
) -> None:
    """429 带 Retry-After 时以服务端要求为准，不跟服务端讨价还价。

    输入：一个带 `Retry-After: 7` 的 429，然后一个成功结果。
    输出：无。预期行为：**两个地方**都按 7 秒来——这条请求自己的退避，和整家 provider
    的退避窗口（并发下别的线程也得等）。测试里的 sleep 是假的、不推进时钟，
    所以第二次正好也是 7 秒整。
    """
    sleeps: list[float] = []
    engine = fake_engine(script=[RateLimitError("429", retry_after_s=7.0), result_factory()])
    provider = provider_factory(
        engine,
        limits=limits(requests_per_minute=40, max_attempts=2, rate_limit_attempts=2),
        sleep=sleeps.append,
    )
    provider.extract(task_factory(), "json_schema")
    assert sleeps == [7.0, 7.0]


def test_auth_error_retires_the_key(
    fake_engine: object, provider_factory: object, result_factory: object, task_factory: object
) -> None:
    """401 只说明这个 key 废了，不是 provider 挂了：换下一个 key 继续。"""
    engine = fake_engine(script=[AuthError("401"), result_factory()])
    provider = provider_factory(engine, keys=("k1", "k2"), limits=limits())
    with pytest.raises(AuthError):
        provider.extract(task_factory(), "json_schema")
    assert provider.available is True
    _, key_fp, _ = provider.extract(task_factory(), "json_schema")
    assert key_fp == fingerprint("k2")


def test_daily_quota_retires_the_key(
    fake_engine: object, provider_factory: object, result_factory: object, task_factory: object
) -> None:
    """每日配额烧干是「这个 key 今天没了」：降速救不回来，必须换 key。

    输入：一个带 `PerDay` 标记的 429，然后一个成功结果。
    输出：无。预期行为：第一次抛 429 且该 key 退场，第二次自动换到第二个 key。
    """
    engine = fake_engine(
        script=[
            RateLimitError("429 RESOURCE_EXHAUSTED quotaId=GenerateRequestsPerDayPerProject"),
            result_factory(),
        ]
    )
    provider = provider_factory(engine, keys=("k1", "k2"), limits=limits())
    with pytest.raises(RateLimitError):
        provider.extract(task_factory(), "json_schema")
    assert provider.available is True
    _, key_fp, _ = provider.extract(task_factory(), "json_schema")
    assert key_fp == fingerprint("k2")


def test_minute_rate_limit_keeps_the_key(
    fake_engine: object, provider_factory: object, result_factory: object, task_factory: object
) -> None:
    """分钟级 429 只降速、不退 key——否则一波限流就把好 key 全扔了。"""
    engine = fake_engine(script=[RateLimitError("429 Too Many Requests"), result_factory()])
    provider = provider_factory(engine, keys=("k1",), limits=limits())
    with pytest.raises(RateLimitError):
        provider.extract(task_factory(), "json_schema")
    assert provider.available is True
    _, key_fp, _ = provider.extract(task_factory(), "json_schema")
    assert key_fp == fingerprint("k1")


def test_rate_limit_gets_a_wider_retry_budget(
    fake_engine: object, provider_factory: object, result_factory: object, task_factory: object
) -> None:
    """429 是「等一会儿就能好」，重试预算必须比普通网络抖动宽。

    输入：连续 4 个 429，然后一个成功结果；普通预算 2、限流预算 6。
    输出：无。预期行为：第 5 次尝试成功——预算按 `rate_limit_attempts` 走，
    不被 `max_attempts=2` 掐掉。
    """
    engine = fake_engine(script=[RateLimitError("429")] * 4 + [result_factory()])
    provider = provider_factory(engine, limits=limits(max_attempts=2, rate_limit_attempts=6))
    _, _, attempts = provider.extract(task_factory(), "json_schema")
    assert attempts == 5


def test_rate_limit_storm_never_opens_the_breaker(
    fake_engine: object, provider_factory: object, task_factory: object
) -> None:
    """一波 429 不该把 provider 判死——熔断一开就再没机会退避重试了。

    输入：连续 5 个 429，熔断阈值 3。
    输出：无。预期行为：预算用尽后原样抛 429，但熔断仍是 closed、请求真的发了 5 次。
    """
    engine = fake_engine(script=[RateLimitError("429")] * 5)
    provider = provider_factory(engine, limits=limits(fail_max=3, rate_limit_attempts=5))
    with pytest.raises(RateLimitError):
        provider.extract(task_factory(), "json_schema")
    assert provider.breaker_state == "closed"
    assert len(engine.calls) == 5


def test_rate_limit_backs_the_whole_provider_off(
    fake_engine: object,
    provider_factory: object,
    result_factory: object,
    task_factory: object,
    fake_clock: object,
) -> None:
    """一个线程吃到 429，整家 provider 一起退避——否则并发的别的线程接着撞。

    输入：假时钟停在 0，脚本是「429、429、成功」，限流预算 3。
    输出：无。预期行为：两次退避窗口按 1 → 2 秒指数增长，且窗口时长真的交给了睡眠函数。
    """
    sleeps: list[float] = []
    engine = fake_engine(script=[RateLimitError("429"), RateLimitError("429"), result_factory()])
    provider = provider_factory(
        engine,
        limits=limits(rate_limit_attempts=3),
        sleep=sleeps.append,
        clock=fake_clock(),
    )
    provider.extract(task_factory(), "json_schema")
    assert sleeps.count(1.0) == 1  # 第一次退避：2^0
    assert sleeps.count(2.0) == 1  # 第二次退避：2^1
    assert provider.backoff_remaining_s == 2.0


def test_retry_after_also_sets_the_provider_backoff(
    fake_engine: object,
    provider_factory: object,
    result_factory: object,
    task_factory: object,
    fake_clock: object,
) -> None:
    """服务端给了 `Retry-After` 就按它来，别的线程也一样等这么久。"""
    engine = fake_engine(script=[RateLimitError("429", retry_after_s=7.0), result_factory()])
    provider = provider_factory(
        engine,
        limits=limits(rate_limit_attempts=2),
        sleep=lambda _seconds: None,
        clock=fake_clock(),
    )
    provider.extract(task_factory(), "json_schema")
    assert provider.backoff_remaining_s == 7.0


def test_success_resets_the_rate_limit_streak(
    fake_engine: object,
    provider_factory: object,
    result_factory: object,
    task_factory: object,
    fake_clock: object,
) -> None:
    """通了就说明限流过去了：下一次再挨 429，退避长度从头算，不接着往上翻。"""
    engine = fake_engine(
        script=[
            RateLimitError("429"),
            result_factory(),
            RateLimitError("429"),
            result_factory(),
        ]
    )
    provider = provider_factory(
        engine,
        limits=limits(rate_limit_attempts=4),
        sleep=lambda _seconds: None,
        clock=fake_clock(),
    )
    provider.extract(task_factory(), "json_schema")
    assert provider.backoff_remaining_s == 1.0


def test_pool_empty_raises_auth_error(
    fake_engine: object, provider_factory: object, task_factory: object
) -> None:
    """唯一的 key 废掉后，这家就没戏了，报错要说人话。"""
    engine = fake_engine(script=[AuthError("401")])
    provider = provider_factory(engine, keys=("k1",), limits=limits())
    with pytest.raises(AuthError):
        provider.extract(task_factory(), "json_schema")
    assert provider.available is False
    with pytest.raises(AuthError, match="key 池已空"):
        provider.extract(task_factory(), "json_schema")


def test_modes_follow_configuration(fake_engine: object, provider_factory: object) -> None:
    """写死档位就用它；auto 有探测结果就从探测档起步，没探测就从最严的起步。"""
    engine = fake_engine(script=[])
    assert provider_factory(engine, mode="tools").modes() == ("tools", "md_json")
    assert provider_factory(engine).modes() == ("json_schema", "tools", "md_json")
    probed = provider_factory(engine, capabilities=BackendCapabilities(tools=True))
    assert probed.modes() == ("tools", "md_json")


def test_concurrency_gate_serialises_calls(provider_factory: object, result_factory: object, task_factory: object) -> None:
    """max_concurrency=1 时，两个线程同时来也只有一个能进引擎。"""

    class Probe:
        """记录「同时在跑的调用条数」的假引擎。"""

        name = "probe"

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.active = 0
            self.peak = 0

        def extract(self, request: object) -> object:
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


def test_chain_iterates_in_configuration_order(fake_engine: object, provider_factory: object) -> None:
    """降级顺序 = 配置顺序，遍历链必须按这个顺序来。"""
    chain = ProviderChain(
        [
            provider_factory(fake_engine(script=[]), name="a"),
            provider_factory(fake_engine(script=[]), name="b"),
        ]
    )
    assert [p.name for p in chain] == ["a", "b"]


def test_min_interval_gate_spaces_out_calls(
    fake_engine: object, provider_factory: object, result_factory: object, task_factory: object
) -> None:
    """同一个 key 的两次请求之间必须隔开配置的最小间隔，否则免费档会被打爆。"""
    engine = fake_engine(script=[result_factory(), result_factory()])
    provider = provider_factory(engine, limits=limits(min_interval_ms=200))
    started = time.monotonic()
    provider.extract(task_factory(), "json_schema")
    provider.extract(task_factory(), "json_schema")
    assert time.monotonic() - started >= 0.15  # 留点余量，不跟调度器较劲


def test_chain_usable_filters_by_model_when_mixed_is_off(fake_engine: object, provider_factory: object) -> None:
    """不允许混用时，只保留与第一家 model 相同的 provider，免得数据分布被换模型污染。"""
    first = provider_factory(fake_engine(script=[]), name="a", model="m1")
    second = provider_factory(fake_engine(script=[]), name="b", model="m2")
    chain = ProviderChain([first, second], allow_mixed=False)
    assert chain.names == ["a", "b"]
    assert len(chain) == 2
    assert [p.name for p in chain.usable()] == ["a"]
    assert [p.name for p in ProviderChain([first, second], allow_mixed=True).usable()] == ["a", "b"]
