"""provider 的运行态与降级链（PLAN 3.1.3 / 3.1.4 / 3.1.6）。

一个 provider 把四件事组合起来：
- **key 池**：`keys.KeyPool`（手写——没有库管「N 个 key 挑哪个」）；
- **限速**：`pyrate-limiter`，provider 级滑动窗口 RPM + 单 key 最小间隔；
- **熔断**：`breaker.Breaker`（手写，见那里的说明：pybreaker 会把被保护的调用锁在它自己的
  RLock 里，等于把同一家的请求串行化），`fail_max` / `reset_timeout_s` 对应 PLAN 的
  「连续 3 次 / 冷却 60 秒」；关键是用 `_is_not_fault` 把「模型答错」排除在故障之外；
- **重试退避**：`tenacity` 指数退避 + 抖动；429 带 `Retry-After` 时以服务端要求为准，
  并且限流走**更宽的重试预算**（`_RetryBudget`：`rate_limit_attempts` > `max_attempts`）——
  一波 429 不该把整批掐死；挨了 429 还会给**整家 provider** 挂一个指数退避窗口
  （`_wait_backoff`），并发下别的线程也不会接着撞同一家。

输入/输出：`Provider.extract()` 收任务与档位，出 `(EngineResult, key 指纹, 尝试次数)`。
预期行为：所有失败都转成 `types` 里的分类异常，上层据此决定降档 / 换 provider / 记账。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Final

from pyrate_limiter import Duration, InMemoryBucket, Limiter, Rate
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    wait_exponential_jitter,
)
from tenacity.stop import stop_base

from .breaker import Breaker
from .config import ProviderConfig
from .engine import Engine, EngineRequest, EngineResult
from .keys import KeyPool, fingerprint
from .types import (
    MODE_AUTO,
    AuthError,
    BackendCapabilities,
    BackendError,
    GenerationTask,
    ProviderUnavailable,
    RateLimitError,
    TransportError,
    modes_from,
)


def _is_not_fault(exc: BaseException) -> bool:
    """输入：异常。输出：True 表示「不是 provider 故障、不计入熔断」。

    预期行为：只有 `types` 里 `counts_as_fault` 为真的才算故障；认不出的异常一律不算，
    免得我们自己的 bug 把 provider 误熔断。这是熔断器记账前的判据。
    """
    return not (isinstance(exc, BackendError) and exc.counts_as_fault)


#: 服务端把「这个 key 今天没了」写进报错正文时的标记（各家措辞不一样，只认最常见的几种）。
_DAILY_QUOTA_MARKERS: Final[tuple[str, ...]] = (
    "perday",
    "per day",
    "per-day",
    "daily quota",
    "daily limit",
)


def _is_daily_quota(exc: RateLimitError) -> bool:
    """输入：一个 429 异常。输出：它是不是「每日配额已用尽」。

    预期行为：这两种 429 的救法完全不同——**分钟级限流要降速，每日配额要退 key**。
    降速救不回每日配额（等一小时也没用），退 key 又会误伤只是被限流的好 key。
    认不出来时按分钟级处理（保守：降速最多慢一点，退 key 会白扔一个 key）。
    """
    text = str(exc).lower()
    return any(marker in text for marker in _DAILY_QUOTA_MARKERS)


def _wait_honouring_retry_after(base: object) -> object:
    """输入：tenacity 的退避策略。输出：包了一层的退避策略。

    预期行为：异常带 `retry_after_s`（来自 429 的 `Retry-After`）且比算出来的退避更长时，
    以服务端要求为准——不跟服务端讨价还价。
    """

    def _wait(state: RetryCallState) -> float:
        delay = float(base(state))
        outcome = state.outcome
        exc = outcome.exception() if outcome is not None else None
        retry_after = getattr(exc, "retry_after_s", None)
        if isinstance(retry_after, (int, float)) and retry_after > delay:
            return float(retry_after)
        return delay

    return _wait


def _make_limiter(limit: int, interval_ms: int) -> Limiter:
    """输入：次数上限与时间窗（毫秒）。输出：pyrate-limiter 的 Limiter。

    预期行为：一个限速窗口一个桶；`Rate(1, 1000)` 就是「每秒最多一次」。
    """
    return Limiter(InMemoryBucket([Rate(limit, interval_ms)]))


class _RetryBudget(stop_base):
    """按失败类别给不同重试预算的停止策略（PLAN 3.1.6）。

    输入：tenacity 的重试状态。输出：True = 预算用尽、该停下。
    预期行为：429 是「等一会儿就能好」，预算取 `rate_limit_attempts`（默认比
    `max_attempts` 宽）；其余可重试失败（网络 / 5xx / 认证）仍用 `max_attempts`。
    """

    def __init__(self, max_attempts: int, rate_limit_attempts: int) -> None:
        """输入：普通预算与限流预算。输出：停止策略实例。"""
        super().__init__()
        self._max_attempts = max_attempts
        self._rate_limit_attempts = rate_limit_attempts

    def __call__(self, retry_state: RetryCallState) -> bool:
        """输入：一次重试的状态。输出：是否停止。预期行为：按本次失败的类别挑预算。"""
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None else None
        limit = self._rate_limit_attempts if isinstance(exc, RateLimitError) else self._max_attempts
        return retry_state.attempt_number >= limit


class Provider:
    """一个 provider 的运行态（线程安全）。

    参数：`config` 静态配置；`keys` 该家的 key 元组；`engine` 结构化输出引擎；
    `capabilities` 可选的能力探测结果（决定 `auto` 档从哪起步）；
    `sleep` 退避用的睡眠函数（测试注入空实现，免得真等）；
    `clock` 单调时钟（测试注入假时钟，才量得出退避窗口）。
    """

    def __init__(
        self,
        config: ProviderConfig,
        keys: Sequence[str],
        engine: Engine,
        *,
        capabilities: BackendCapabilities | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """输入：配置、key、引擎、可选探测结果、睡眠函数、时钟。输出：Provider 实例。"""
        self.config = config
        self._engine = engine
        self._capabilities = capabilities
        self._sleep = sleep
        self._clock = clock
        self._pool = KeyPool(keys, config.limits.max_concurrency_per_key)
        self._gate = threading.BoundedSemaphore(config.limits.max_concurrency)
        self._rpm_lock = threading.Lock()
        self._rpm_cap = config.limits.requests_per_minute
        self._rpm_current = config.limits.requests_per_minute
        self._rpm = _make_limiter(self._rpm_current, Duration.MINUTE) if self._rpm_current > 0 else None
        #: 429 之后的「整家一起退避」窗口：一个线程被限流，别的线程也不该接着撞。
        #: `_rl_streak` 是连续被限流的次数，决定窗口长度（指数），成功一次就清零。
        self._backoff_until = 0.0
        self._rl_streak = 0
        self._interval = (
            _make_limiter(1, config.limits.min_interval_ms)
            if config.limits.min_interval_ms > 0
            else None
        )
        self._breaker = Breaker(
            fail_max=config.limits.fail_max,
            reset_timeout_s=config.limits.cooldown_s,
        )

    @property
    def name(self) -> str:
        """provider 名。"""
        return self.config.name

    @property
    def model(self) -> str:
        """该家使用的模型名。"""
        return self.config.model

    @property
    def available(self) -> bool:
        """这家现在还有没有戏：key 池非空。

        预期行为：**熔断状态不在这里判断**——冷却到点是在 `allow()` 里推进的，
        所以让它自己说话，熔断打开时 `extract()` 会抛 `ProviderUnavailable`。
        """
        return self._pool.available() > 0

    @property
    def breaker_state(self) -> str:
        """当前熔断状态（closed / open / half-open），供记账与调试。"""
        return self._breaker.state

    @property
    def effective_rpm(self) -> int:
        """当前生效的 RPM（可能被 429 自适应减半过）。"""
        with self._rpm_lock:
            return self._rpm_current

    @property
    def backoff_remaining_s(self) -> float:
        """输入：无。输出：还差多少秒才该再打这家。预期行为：没在退避时给 0。"""
        return max(0.0, self._backoff_until - self._clock())

    def modes(self) -> tuple[str, ...]:
        """输入：无。输出：本次要依次尝试的档位序列。

        预期行为：配置写死档位就用它；`auto` 时若探测过能力就从探测出的档位起步，
        没探测过则从最严的 `json_schema` 起步。
        """
        if self.config.mode != MODE_AUTO:
            return modes_from(self.config.mode)
        if self._capabilities is not None:
            return modes_from(self._capabilities.starting_mode())
        return modes_from(MODE_AUTO)

    def _retrying(self) -> Retrying:
        """输入：无。输出：配好的 tenacity 重试器。

        预期行为：只重试**可重试**的失败（网络 / 5xx / 429 / 认证）；「模型答错」与
        「这一档 schema 不支持」不重试，直接交给上层降档或换 provider。
        预算由 `_RetryBudget` 定：429 用 `rate_limit_attempts`，其余用 `max_attempts`。
        """
        limits = self.config.limits
        return Retrying(
            stop=_RetryBudget(limits.max_attempts, limits.rate_limit_attempts),
            wait=_wait_honouring_retry_after(
                wait_exponential_jitter(initial=1.0, max=max(1.0, limits.cooldown_s))
            ),
            retry=retry_if_exception_type((RateLimitError, TransportError, AuthError)),
            reraise=True,
            sleep=self._sleep,
        )

    def _wait_limits(self, key: str) -> None:
        """输入：本次要用的 key。输出：无。预期行为：阻塞到 RPM 与最小间隔都满足。"""
        if self._rpm is not None:
            self._rpm.try_acquire("provider", blocking=True)
        if self._interval is not None:
            self._interval.try_acquire(key, blocking=True)

    def _wait_backoff(self) -> None:
        """输入：无。输出：无。预期行为：429 之后整家 provider 一起等到窗口结束。

        只有挨打的那个线程自己退避是不够的：并发下别的线程会接着撞同一家端点，
        429 就一直消不掉。所以退避窗口挂在 provider 上，谁进来都先等。
        """
        remaining = self._backoff_until - self._clock()
        if remaining > 0:
            self._sleep(remaining)

    def _on_rate_limited(self, retry_after_s: float | None = None) -> None:
        """输入：可选的 `Retry-After` 秒数。输出：无。

        预期行为：两件事一起做——① RPM 减半（下限 1），下次请求起生效；② 记一个
        **整家 provider 一起等**的退避窗口，长度按连续被限流的次数指数增长（1、2、4…
        秒），封顶 `cooldown_s`；服务端给了 `Retry-After` 就以它为准（不跟服务端讨价还价）。

        这是 PLAN 3.1.6 的自适应降速——NVIDIA NIM 不回 `x-ratelimit-*`，拿不到官方口径，
        配置里的 RPM 只能是起点。
        """
        with self._rpm_lock:
            self._rpm_current = max(1, self._rpm_current // 2)
            self._rpm = _make_limiter(self._rpm_current, Duration.MINUTE)
            self._rl_streak += 1
            window = retry_after_s
            if window is None:
                window = min(self.config.limits.cooldown_s, float(2 ** (self._rl_streak - 1)))
            self._backoff_until = max(self._backoff_until, self._clock() + max(0.0, window))

    def _on_success(self) -> None:
        """输入：无。输出：无。预期行为：RPM 低于配置值时每次成功抬回 10%，封顶配置值。"""
        with self._rpm_lock:
            self._rl_streak = 0  # 通了就说明限流过去了，退避长度从头算
            if self._rpm_current >= self._rpm_cap:
                return
            step = max(1, self._rpm_cap // 10)
            self._rpm_current = min(self._rpm_cap, self._rpm_current + step)
            self._rpm = _make_limiter(self._rpm_current, Duration.MINUTE)

    def _extract_once(self, task: GenerationTask, mode: str) -> tuple[EngineResult, str]:
        """输入：任务与档位。输出：(结果, key 指纹)。预期行为：走一次完整的闸门 + 熔断调用。"""
        with self._gate:
            with self._pool.slot() as key:
                self._wait_backoff()
                self._wait_limits(key)
                request = EngineRequest(
                    base_url=self.config.base_url,
                    api_key=key,
                    model=self.config.model,
                    prompt=task.prompt,
                    response_model=task.response_model,
                    mode=mode,
                    timeout_s=self.config.timeout_s,
                    max_retries=self.config.max_retries,
                )
                # 熔断闸门：判定与记账各占一次短锁，**网络调用不在锁里**（见 breaker.py 的说明）。
                if not self._breaker.allow():
                    raise ProviderUnavailable(
                        f"{self.name} 正在熔断：连续 {self.config.limits.fail_max} 次故障，"
                        f"冷却 {self.config.limits.cooldown_s} 秒中"
                    )
                try:
                    result = self._engine.extract(request)
                except BaseException as exc:
                    if not _is_not_fault(exc):
                        self._breaker.record_fault()
                    if isinstance(exc, AuthError):
                        self._pool.mark_invalid(key)  # 这个 key 废了，下次换一个
                    elif isinstance(exc, RateLimitError):
                        if _is_daily_quota(exc):
                            self._pool.mark_invalid(key)  # 今天这个 key 没了，换下一个
                        else:
                            self._on_rate_limited(exc.retry_after_s)
                    raise
                self._breaker.record_ok()
                self._on_success()
                return result, fingerprint(key)

    def extract(self, task: GenerationTask, mode: str) -> tuple[EngineResult, str, int]:
        """输入：任务与档位。输出：(结果, key 指纹, 实际尝试次数)。

        预期行为：按 `max_attempts` 重试可重试的失败；重试用尽原样抛出最后一个异常。
        """
        attempts = 0
        for attempt in self._retrying():
            with attempt:
                attempts += 1
                result, key_fp = self._extract_once(task, mode)
                return result, key_fp, attempts
        raise ProviderUnavailable(f"{self.name}：重试用尽仍未成功")  # pragma: no cover


class ProviderChain:
    """有序的 provider 列表（PLAN 3.1.4）。

    参数：`providers` 有序列表；`allow_mixed` 是否允许降级到**不同 model** 的 provider。
    预期行为：默认允许（主人的要求是「全部提供商的都挂了才停下，每家可以配不同 model」），
    但每家都记账，事后能按 provider 切片；想要「一家到底」就设 `allow_mixed=False`。
    """

    def __init__(self, providers: Sequence[Provider], *, allow_mixed: bool = True) -> None:
        """输入：provider 序列与混用开关。输出：ProviderChain 实例。"""
        if not providers:
            raise ValueError("provider 链不能为空")
        self._providers = list(providers)
        self.allow_mixed = allow_mixed

    def __iter__(self) -> Iterator[Provider]:
        """按配置顺序迭代。"""
        return iter(self._providers)

    def __len__(self) -> int:
        """provider 数量。"""
        return len(self._providers)

    @property
    def names(self) -> list[str]:
        """全部 provider 名（按顺序）。"""
        return [p.name for p in self._providers]

    def usable(self) -> list[Provider]:
        """输入：无。输出：按顺序排列的、当前有 key 可用的 provider。

        预期行为：`allow_mixed=False` 时只保留与第一家 **model 相同**的那些，
        这样降级不会把数据分布换成另一个模型。
        """
        live = [p for p in self._providers if p.available]
        if self.allow_mixed or not live:
            return live
        first_model = live[0].model
        return [p for p in live if p.model == first_model]
