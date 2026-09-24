"""引擎测试：异常分类、原始响应拆解、instructor 包装（客户端全 mock，不发网络）。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from instructor.core import IncompleteOutputException, InstructorRetryException
from pydantic import BaseModel, ValidationError

from jc.backend import engine as engine_module
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
    """PLAN 3.1.2：默认不发 max_tokens，字段根本不该出现。"""
    engine = InstructorEngine()
    engine.extract(request_for())
    kwargs = patched["clients"][0].kwargs
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs


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
