"""引擎测试：异常分类、原始响应拆解、instructor 包装（客户端全 mock，不发网络）。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from instructor.core import FailedAttempt, IncompleteOutputException, InstructorRetryException
from pydantic import BaseModel, ValidationError

from jc.backend import engine as engine_module
from jc.backend.config import Limits, ProviderConfig
from jc.backend.engine import EngineRequest, InstructorEngine, _split_raw, classify_exception
from jc.backend.types import (
    MODE_JSON_SCHEMA,
    MODE_MD_JSON,
    MODE_TOOLS,
    AuthError,
    BackendError,
    DraftInvalidError,
    RateLimitError,
    SchemaUnsupportedError,
    TransportError,
)


class Answer(BaseModel):
    """测试用的最小草稿模型。"""

    answer: str


def status_error(cls: type, status: int, headers: dict[str, str] | None = None) -> Any:
    """输入：openai 异常类、HTTP 状态、响应头。输出：造好的异常实例。"""
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx2.Response(status, request=request, headers=headers or {})
    return cls("boom", response=response, body=None)


def retry_wrapper(
    cause: BaseException | None = None, *, only_attempts: bool = False
) -> InstructorRetryException:
    """造一个 instructor 的「重试用尽」异常。

    输入：底层原因异常；`only_attempts=True` 时只填 `failed_attempts`、不挂 `__cause__`。
    输出：InstructorRetryException。
    预期行为：默认把原因挂在 `__cause__` 上，与 instructor 真实的
    `raise InstructorRetryException(...) from last_exception` 保持一致。
    """
    if cause is not None and only_attempts:
        return InstructorRetryException(
            "Max retries exceeded",
            n_attempts=1,
            total_usage=0,
            failed_attempts=[FailedAttempt(attempt_number=1, exception=cause, completion=None)],
        )
    exc = InstructorRetryException("Max retries exceeded", n_attempts=1, total_usage=0)
    exc.__cause__ = cause
    return exc


def timeout_error() -> Any:
    """输入：无。输出：一个读超时的 openai 异常。预期行为：当「被包起来的底层原因」用。"""
    return openai.APITimeoutError(request=httpx2.Request("POST", "https://example.test/v1"))


def test_classify_auth_failures() -> None:
    assert isinstance(classify_exception(status_error(openai.AuthenticationError, 401)), AuthError)
    assert isinstance(classify_exception(status_error(openai.PermissionDeniedError, 403)), AuthError)


def test_classify_rate_limit_carries_retry_after() -> None:
    exc = status_error(openai.RateLimitError, 429, {"retry-after": "7"})
    classified = classify_exception(exc)
    assert isinstance(classified, RateLimitError)
    assert classified.retry_after_s == 7.0


def test_classify_rate_limit_tolerates_unusable_retry_after() -> None:
    """`Retry-After` 是日期串或干脆没有时给 None，绝不能因为读不懂就崩。"""
    unparsable = status_error(openai.RateLimitError, 429, {"retry-after": "Wed, 21 Oct 2015"})
    assert classify_exception(unparsable).retry_after_s is None
    assert classify_exception(status_error(openai.RateLimitError, 429)).retry_after_s is None


def test_classify_schema_unsupported() -> None:
    assert isinstance(classify_exception(status_error(openai.BadRequestError, 400)), SchemaUnsupportedError)
    assert isinstance(
        classify_exception(status_error(openai.UnprocessableEntityError, 422)),
        SchemaUnsupportedError,
    )


def test_classify_transport_failures() -> None:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    assert isinstance(classify_exception(openai.APIConnectionError(request=request)), TransportError)
    assert isinstance(classify_exception(openai.APITimeoutError(request=request)), TransportError)
    assert isinstance(
        classify_exception(status_error(openai.InternalServerError, 500)), TransportError
    )


def test_classify_model_misbehaviour_is_not_a_fault() -> None:
    """模型答错 / 解析失败绝不能算 provider 故障，否则会误熔断。"""
    for exc in (
        IncompleteOutputException(),
        # n_attempts / total_usage 是 instructor 的必填关键字参数，随便给个值即可
        InstructorRetryException("failed", n_attempts=1, total_usage=0),
        ValueError("未知错误"),
    ):
        classified = classify_exception(exc)
        assert classified.counts_as_fault is False
    assert isinstance(classify_exception(IncompleteOutputException()), DraftInvalidError)


def test_classify_keeps_backend_errors_as_is() -> None:
    original = TransportError("网络挂了")
    assert classify_exception(original) is original


def test_classify_unwraps_transport_cause_hidden_in_retry_wrapper() -> None:
    """实测坑：NIM 端点读超时 / 504 被 instructor 包成「重试用尽」，必须还原成故障。

    输入：底层是读超时与 504 的重试异常。
    输出：分类结果。
    预期行为：两者都是 `TransportError`（`counts_as_fault` 为真）——否则重试、熔断、
    provider 降级全都不触发，整批会一直打一个已经挂掉的端点，还把原因记成「草稿不合格」。
    """
    for cause in (timeout_error(), status_error(openai.InternalServerError, 504)):
        classified = classify_exception(retry_wrapper(cause))
        assert isinstance(classified, TransportError)
        assert classified.counts_as_fault is True


def test_classify_unwraps_rate_limit_cause_keeping_retry_after() -> None:
    """429 被包起来也要还原，而且 `Retry-After` 不能丢——退避与自适应降速都靠它。"""
    wrapped = retry_wrapper(status_error(openai.RateLimitError, 429, {"retry-after": "7"}))
    classified = classify_exception(wrapped)
    assert isinstance(classified, RateLimitError)
    assert classified.retry_after_s == 7.0


def test_classify_unwraps_auth_and_schema_causes() -> None:
    """认证失败要还原成 `AuthError`（key 得被剔除），400 要还原成 `SchemaUnsupportedError`（该降档）。"""
    auth = classify_exception(retry_wrapper(status_error(openai.AuthenticationError, 401)))
    schema = classify_exception(retry_wrapper(status_error(openai.BadRequestError, 400)))
    assert isinstance(auth, AuthError)
    assert isinstance(schema, SchemaUnsupportedError)


def test_classify_keeps_validation_cause_as_draft_invalid() -> None:
    """底层真是「模型答错」时口径不变：`DraftInvalidError`，且绝不算 provider 故障。"""
    try:
        Answer(answer=1)
    except ValidationError as exc:
        validation: BaseException = exc
    classified = classify_exception(retry_wrapper(validation))
    assert isinstance(classified, DraftInvalidError)
    assert classified.counts_as_fault is False


def test_classify_reads_failed_attempts_when_cause_is_missing() -> None:
    """没有 `__cause__`（别的 instructor 版本）时退回 `failed_attempts` 的最后一条。"""
    wrapped = retry_wrapper(timeout_error(), only_attempts=True)
    assert isinstance(classify_exception(wrapped), TransportError)


def test_classify_survives_self_referential_retry_wrapper() -> None:
    """`__cause__` 指向自己（或成环）时必须收敛，不能递归到爆栈。"""
    exc = InstructorRetryException("boom", n_attempts=1, total_usage=0)
    exc.__cause__ = exc
    assert isinstance(classify_exception(exc), DraftInvalidError)


def test_extract_unwraps_wrapped_transport_errors(patched: dict[str, Any]) -> None:
    """端到端走生产路径：客户端抛「重试用尽（底层读超时）」，引擎必须抛 `TransportError`。"""
    patched["next_client"] = FakeClient(error=retry_wrapper(timeout_error()))
    with pytest.raises(TransportError):
        InstructorEngine().extract(request_for())


def test_classify_pydantic_validation_error() -> None:
    try:
        Answer(answer=1)
    except ValidationError as exc:
        classified = classify_exception(exc)
    assert isinstance(classified, DraftInvalidError)


def test_split_raw_extracts_text_usage_and_reasoning() -> None:
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"answer":"hi"}', reasoning="think hard")
            )
        ],
        usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 5}),
    )
    text, usage, reasoning = _split_raw(raw)
    assert text == '{"answer":"hi"}'
    assert usage == {"total_tokens": 5}
    assert reasoning == "think hard"


def test_split_raw_tolerates_missing_fields() -> None:
    assert _split_raw(SimpleNamespace()) == ("", {}, None)


class FakeClient:
    """假的 instructor 客户端：回放一个 (parsed, raw) 或一个异常。"""

    def __init__(self, parsed: Any = None, raw: Any = None, error: BaseException | None = None):
        self.parsed = parsed
        self.raw = raw
        self.error = error
        self.kwargs: dict[str, Any] = {}

    def create_with_completion(self, **kwargs: Any) -> tuple[Any, Any]:
        """输入：调用参数。输出：(parsed, raw)。预期行为：记下参数供断言。"""
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.parsed, self.raw


def make_raw() -> Any:
    """造一个最小的原始 completion。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"hi"}'))],
        usage=None,
    )


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """把 openai 客户端与 instructor.from_openai 都换掉，返回可观察的盒子。"""
    box: dict[str, Any] = {"modes": [], "clients": []}

    def fake_from_openai(_client: Any, mode: Any) -> FakeClient:
        box["modes"].append(mode)
        fake = box.get("next_client") or FakeClient(Answer(answer="hi"), make_raw())
        box["clients"].append(fake)
        return fake

    monkeypatch.setattr(engine_module.openai, "OpenAI", lambda **_kw: object())
    monkeypatch.setattr(engine_module.instructor, "from_openai", fake_from_openai)
    return box


def request_for(mode: str = MODE_JSON_SCHEMA, key: str = "k1") -> EngineRequest:
    """造一次引擎请求。"""
    return EngineRequest(
        base_url="https://example.test/v1",
        api_key=key,
        model="openai/gpt-oss-20b",
        prompt="say hi",
        response_model=Answer,
        mode=mode,
    )


def test_extract_returns_parsed_data(patched: dict[str, Any]) -> None:
    result = InstructorEngine().extract(request_for())
    assert result.data == {"answer": "hi"}
    assert result.raw_text == '{"answer":"hi"}'
    assert result.mode == MODE_JSON_SCHEMA


def test_extract_never_sends_max_tokens(patched: dict[str, Any]) -> None:
    """PLAN 3.1.2：永不发 max_tokens，字段根本不该出现。"""
    engine = InstructorEngine()
    engine.extract(request_for())
    kwargs = patched["clients"][0].kwargs
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs


def tool_calls_body() -> dict[str, Any]:
    """输入：无。输出：一份与真实端点同形的完成体（既有 tool_calls 也有 JSON 文本）。

    预期行为：给 MockTransport 用；tools / json_schema / md_json 三档都能从它解析出
    `{"answer": "hi"}`。
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-20b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"answer": "hi"}',
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Answer", "arguments": '{"answer": "hi"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_real_instructor_request_body_has_no_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLAN 3.1.2 的端到端守卫：真 instructor 走一遍 HTTP 序列化，请求体里不许有 max_tokens。

    输入：无。输出：无。预期行为：上面那条测试把 instructor 整个换成了假客户端，只能证明
    **我们**没传；这条补另一半——真库自己也不会偷偷塞。必须用真库验，因为「默认塞上限」
    正是上游会干的事（instructor 的 v2 anthropic 分支就写死 `max_tokens=4096`）。
    抓取器的灵敏度已自检：主动注入 `max_tokens` 时它看得见。
    """
    seen: list[dict[str, Any]] = []
    # 先把真类捞出来：`engine_module.openai` 就是全局 openai 模块，patch 之后
    # 再写 `openai.OpenAI` 会指回自己，直接无限递归。
    real_openai = openai.OpenAI

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=tool_calls_body())

    def fake_openai(**kwargs: Any) -> Any:
        """输入：引擎给的构造参数。输出：真 openai 客户端，但 HTTP 层换成 MockTransport。"""
        return real_openai(
            api_key=kwargs["api_key"],
            base_url=kwargs["base_url"],
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        )

    class OpenAIModuleProxy:
        """`jc.backend.engine` 眼里的 `openai` 模块替身：只换 `OpenAI`，其余属性透传。

        预期行为：**不能**直接 patch 真模块上的 `OpenAI` —— instructor 自己也要
        `isinstance(client, openai.OpenAI)`，把它换成函数就不再是类型了（实测直接 TypeError）。
        """

        OpenAI = staticmethod(fake_openai)

        def __getattr__(self, name: str) -> Any:
            return getattr(openai, name)

    monkeypatch.setattr(engine_module, "openai", OpenAIModuleProxy())
    result = InstructorEngine().extract(request_for(MODE_TOOLS))
    assert result.data == {"answer": "hi"}
    body = seen[-1]
    assert "tools" in body  # 证明确实走的是真 tools 档，不是被 mock 掉的空转
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body


def test_no_max_tokens_knob_anywhere() -> None:
    """PLAN 3.1.2：不是「默认不发」，而是「根本没这个开关」。

    输入：无。输出：无。预期行为：请求对象、provider 配置、限额配置三处都不存在
    `max_tokens` 字段——用户没有任何入口能把上限塞进请求。
    """
    for cls in (EngineRequest, ProviderConfig, Limits):
        names = set(cls.__dataclass_fields__)
        assert "max_tokens" not in names, cls
        assert "max_completion_tokens" not in names, cls


def test_extract_always_sends_the_model(patched: dict[str, Any]) -> None:
    """model 必须显式传下去——漏了它，真实端点会直接拒收请求。

    这个断言是补的：假客户端不校验参数，所以漏传 model 也能全绿，
    只有真打端点才会暴露（实测报「Missing required arguments」）。
    """
    engine = InstructorEngine()
    engine.extract(request_for())
    assert patched["clients"][0].kwargs["model"] == "openai/gpt-oss-20b"


def test_extract_maps_our_mode_to_instructor_mode(patched: dict[str, Any]) -> None:
    engine = InstructorEngine()
    engine.extract(request_for(MODE_TOOLS))
    engine.extract(request_for(MODE_MD_JSON))
    import instructor

    assert patched["modes"] == [instructor.Mode.TOOLS, instructor.Mode.MD_JSON]


def test_clients_are_cached_per_endpoint_key_and_mode(patched: dict[str, Any]) -> None:
    engine = InstructorEngine()
    engine.extract(request_for(key="k1"))
    engine.extract(request_for(key="k1"))
    engine.extract(request_for(key="k2"))
    assert len(patched["clients"]) == 2


def test_extract_classifies_errors(patched: dict[str, Any]) -> None:
    patched["next_client"] = FakeClient(error=status_error(openai.RateLimitError, 429))
    with pytest.raises(RateLimitError):
        InstructorEngine().extract(request_for())


def test_extract_rejects_unknown_mode(patched: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="未知的结构化输出档位"):
        InstructorEngine().extract(request_for("yaml"))


def test_extract_passes_system_message_when_given(patched: dict[str, Any]) -> None:
    request = EngineRequest(
        base_url="https://example.test/v1",
        api_key="k",
        model="m",
        prompt="say hi",
        response_model=Answer,
        mode=MODE_JSON_SCHEMA,
        system="you are a cat",
    )
    InstructorEngine().extract(request)
    messages = patched["clients"][0].kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "you are a cat"}
