"""后端层测试的公共夹具（PLAN 3.1）。

这里所有「外部」都被 mock 掉：假引擎负责回放脚本、假 openai 客户端负责回放响应，
所以整个测试套件**不发任何网络请求**，也不需要真 key。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue



import pytest
from pydantic import BaseModel

import time

from jc.backend.config import Limits, ProviderConfig
from jc.backend.engine import EngineRequest, EngineResult
from jc.backend.provider import Provider
from jc.backend.types import BackendCapabilities, GenerationTask


class DummyAnswer(BaseModel):
    """测试用的最小草稿模型：只有一个字符串字段。"""

    answer: str


def make_task(task_id: str = "t1", **overrides: object) -> GenerationTask:
    """输入：任务 id 与字段覆盖项。输出：一个最小的 GenerationTask。

    预期行为：不依赖任何 workflow，任何后端测试都能直接拿来用。
    """
    base: dict[str, JsonValue] = {
        "task_id": task_id,
        "kind": "choice",
        "domain": "test",
        "prompt": "随便问一个选择题",
        "response_model": DummyAnswer,
    }
    base.update(overrides)
    return GenerationTask(**base)


@pytest.fixture
def task_factory() -> object:
    """工厂夹具：`task_factory("t2")` 造一个最小任务。"""
    return make_task


class FakeEngine:
    """按脚本回放的假引擎。

    输入：`script` 是待回放项列表（`EngineResult` 实例，或异常实例 / 异常类）。
    输出：`extract()` 依次吐出脚本项。
    预期行为：脚本用完且没给 default 就报错，避免测试因为「没调用」而静默通过。
    """

    name = "fake"

    def __init__(self, script: object = (), default: object = None) -> None:
        """输入：脚本序列与兜底结果。输出：FakeEngine 实例。"""
        self.script: list[JsonValue] = list(script)
        self.default = default
        self.calls: list[EngineRequest] = []

    def extract(self, request: EngineRequest) -> EngineResult:
        """输入：一次请求。输出：脚本里的下一项（或抛脚本里的异常）。"""
        self.calls.append(request)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, type) and issubclass(item, BaseException):
                raise item("脚本安排的失败")
            return item
        if self.default is not None:
            return self.default
        raise AssertionError("FakeEngine 的脚本用完了")


def ok_result(data: dict[str, JsonValue] | None = None, mode: str = "json_schema") -> EngineResult:
    """造一个成功的引擎结果。输入：可选数据与档位。输出：EngineResult。"""
    return EngineResult(
        data=data if data is not None else {"answer": "x"},
        raw_text='{"answer":"x"}',
        usage={"total_tokens": 3},
        reasoning=None,
        mode=mode,
    )


@pytest.fixture
def fake_engine() -> type[FakeEngine]:
    """工厂夹具：`fake_engine(script=[...])` 造一个 FakeEngine。"""
    return FakeEngine


@pytest.fixture
def result_factory() -> object:
    """工厂夹具：`result_factory({"answer": "a"})` 造成功结果。"""
    return ok_result


@pytest.fixture
def no_sleep() -> object:
    """退避用的空睡眠函数：测试里不真等。"""
    return lambda _seconds: None


class FakeClock:
    """可推进的假单调时钟：只有显式改 `now` 才走字，用来量退避窗口。"""

    def __init__(self, now: float = 0.0) -> None:
        """输入：起始刻度。输出：FakeClock 实例。"""
        self.now = now

    def __call__(self) -> float:
        """输入：无。输出：当前刻度。预期行为：不推进就一直是同一个值。"""
        return self.now


@pytest.fixture
def fake_clock() -> type[FakeClock]:
    """工厂夹具：`fake_clock()` 造一个停在 0 的假时钟。"""
    return FakeClock


def make_provider(
    engine: object,
    *,
    name: str = "fake",
    keys: tuple[str, ...] = ("k1",),
    model: str = "fake-model",
    mode: str = "auto",
    limits: Limits | None = None,
    capabilities: BackendCapabilities | None = None,
    sleep: object = None,
    clock: object = None,
) -> Provider:
    """造一个 provider，默认关掉限速与真实睡眠。

    输入：假引擎、可选名字 / key / 模型 / 档位 / 限额 / 能力 / 睡眠函数 / 时钟。
    输出：Provider 实例。
    预期行为：`requests_per_minute=0` 与 `min_interval_ms=0` 让限速器完全不介入，
    测试只测被测的那件事，不被限速拖慢。
    """
    return Provider(
        ProviderConfig(
            name=name,
            base_url="https://example.test/v1",
            model=model,
            mode=mode,
            limits=limits
            or Limits(
                requests_per_minute=0,
                min_interval_ms=0,
                max_attempts=1,
                rate_limit_attempts=1,  # 「关掉重试」的语义要覆盖限流那一档
            ),
        ),
        keys,
        engine,
        capabilities=capabilities,
        sleep=sleep or (lambda _seconds: None),
        clock=clock or time.monotonic,
    )


@pytest.fixture
def provider_factory() -> object:
    """工厂夹具：`provider_factory(fake_engine(script=[...]))` 造一个 provider。"""
    return make_provider
