"""结构化输出引擎：用 `instructor` 打三档（PLAN 3.1.1）。

三档对应 instructor 自己的 Mode，我们**不自己拼请求体**：

| 我们的档位 | instructor Mode | 含义 |
|---|---|---|
| `json_schema` | `JSON_SCHEMA` | 服务端强约束（最稳） |
| `tools` | `TOOLS` | 工具调用（「取巧」那条路） |
| `md_json` | `MD_JSON` | 提示词 + 从文本里抠 JSON（最兜底） |

采样参数**永不发 `max_tokens`**、也不给开关（PLAN 3.1.2）：推理模型会把预算烧在 `reasoning` 上，
`max_tokens` 越小越容易「想一半被砍死」。

输入/输出：`EngineRequest` → `EngineResult`。预期行为：所有异常都转成 `types` 里的分类异常，
上层靠 `counts_as_fault` 决定要不要计入熔断。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import instructor
import openai
from instructor.core import (
    IncompleteOutputException,
    InstructorRetryException,
    ResponseParsingError,
)
from pydantic import ValidationError

from .keys import fingerprint
from .types import (
    MODE_JSON_SCHEMA,
    MODE_MD_JSON,
    MODE_TOOLS,
    AuthError,
    BackendError,
    DraftInvalidError,
    RateLimitError,
    SchemaUnsupportedError,
    StructuredModel,
    TransportError,
)

#: 我们的档位 → instructor 的 Mode。
MODE_TO_INSTRUCTOR: dict[str, JsonValue] = {
    MODE_JSON_SCHEMA: instructor.Mode.JSON_SCHEMA,
    MODE_TOOLS: instructor.Mode.TOOLS,
    MODE_MD_JSON: instructor.Mode.MD_JSON,
}


@dataclass(frozen=True)
class EngineRequest:
    """一次结构化输出请求。

    字段：`response_model` 必须是 pydantic 模型（instructor 的硬要求）；
    `max_retries` 交给 instructor 做「校验失败回喂重试」；`system` 可选。
    """

    base_url: str
    api_key: str
    model: str
    prompt: str
    response_model: type[StructuredModel]
    mode: str
    timeout_s: float = 120.0
    max_retries: int = 3
    system: str | None = None


@dataclass(frozen=True)
class EngineResult:
    """一次结构化输出的结果。字段：`data` 解析后的 dict；`raw_text` 原始文本。"""

    data: dict[str, JsonValue]
    raw_text: str
    usage: dict[str, JsonValue]
    reasoning: str | None
    mode: str


@runtime_checkable
class Engine(Protocol):
    """结构化输出引擎协议。"""

    name: str

    def extract(self, request: EngineRequest) -> EngineResult:
        """输入：一次请求。输出：结果。预期行为：失败时抛分类好的 `BackendError` 子类。"""


def _retry_after_s(exc: openai.APIStatusError) -> float | None:
    """输入：一个 openai 状态异常。输出：`Retry-After` 秒数，读不到给 None。"""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


#: instructor 的「重试用尽」里，只有这几类原因**不是**「模型答错」，要还原后原样上抛：
#: 认证（这个 key 废了）、限流（该退避 / 换 key）、不吃这一档 schema（该降档）、
#: 端点故障（该重试 / 换下一家）。
_NON_DRAFT_ERRORS = (AuthError, RateLimitError, SchemaUnsupportedError, TransportError)


def _unwrap_retry_cause(exc: InstructorRetryException) -> BaseException | None:
    """输入：instructor 的「重试用尽」异常。输出：最后一次真正抛出的异常，取不到给 None。

    预期行为：优先用 `__cause__`（instructor 是 `raise InstructorRetryException(...) from
    last_exception`），没有就退回 `failed_attempts` 的最后一条。**沿链走到底，并记下已访问
    的 id**：异常可能被嵌套包好几层，也可能自引用成环，不能无限递归。
    """
    seen = {id(exc)}
    current: BaseException | None = exc
    while isinstance(current, InstructorRetryException):
        cause: BaseException | None = current.__cause__
        if cause is None:
            attempts = getattr(current, "failed_attempts", None) or []
            cause = getattr(attempts[-1], "exception", None) if attempts else None
        if cause is None or id(cause) in seen:
            return None
        seen.add(id(cause))
        current = cause
    return current


def classify_exception(exc: BaseException) -> BackendError:
    """输入：任意异常。输出：分类好的 `BackendError` 子类。

    预期行为：**「模型答错」绝不能算 provider 故障**——所以解析 / 校验类失败统一归到
    `DraftInvalidError`（`counts_as_fault = False`）；只有网络、5xx、429 才算故障。
    `InstructorRetryException` 会**先还原底层原因**再分类（见 `_unwrap_retry_cause`）。
    已经是 `BackendError` 的原样返回；认不出来的异常归到基类（不算故障），避免误熔断。
    """
    if isinstance(exc, BackendError):
        return exc
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return AuthError(f"认证失败：{exc}")
    if isinstance(exc, openai.RateLimitError):
        return RateLimitError(f"被限流：{exc}", retry_after_s=_retry_after_s(exc))
    if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        return SchemaUnsupportedError(f"服务端不接受这次请求（多半是不吃这一档 schema）：{exc}")
    if isinstance(
        exc,
        (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError),
    ):
        return TransportError(f"网络或服务端故障：{exc}")
    if isinstance(exc, InstructorRetryException):
        # 实测坑（NVIDIA NIM）：端点读超时 / 504 会被 instructor 包成「重试用尽」，
        # 若一律归到 DraftInvalidError，重试、熔断、降级全都不触发，还会把原因记成
        # 「模型没给出合格草稿」。所以先把底层原因还原出来，只有真答错才算草稿不合格。
        cause = _unwrap_retry_cause(exc)
        if cause is not None:
            classified = classify_exception(cause)
            if isinstance(classified, _NON_DRAFT_ERRORS):
                return classified
        return DraftInvalidError(f"模型没给出合格草稿：{exc}")
    if isinstance(
        exc,
        (
            IncompleteOutputException,
            ResponseParsingError,
            ValidationError,
        ),
    ):
        return DraftInvalidError(f"模型没给出合格草稿：{exc}")
    return BackendError(f"未分类的后端异常（{type(exc).__name__}）：{exc}")


def _split_raw(raw: object) -> tuple[str, dict[str, JsonValue], str | None]:
    """输入：instructor 交回的原始 completion。输出：(原始文本, usage, 推理文本)。

    预期行为：字段缺失一律降级成空值，不抛异常。`reasoning` 单独取出，
    **不回喂给模型**（否则等于把思考过程喂回去）。
    """
    text = ""
    reasoning: str | None = None
    usage: dict[str, JsonValue] = {}
    choices = getattr(raw, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            text = content if isinstance(content, str) else ""
            for field_name in ("reasoning", "reasoning_content"):
                value = getattr(message, field_name, None)
                if isinstance(value, str) and value:
                    reasoning = value
                    break
    raw_usage = getattr(raw, "usage", None)
    if raw_usage is not None:
        dump = getattr(raw_usage, "model_dump", None)
        usage = dump() if callable(dump) else dict(raw_usage)
    return text, usage, reasoning


class InstructorEngine:
    """用 `instructor` 做结构化输出。

    参数：`name` 引擎名（写进草稿的 `backend` 字段）。
    预期行为：按 (base_url, key 指纹, 档位) 缓存客户端；缓存键用指纹而非明文 key。
    """

    name = "instructor"

    def __init__(self, name: str = "instructor") -> None:
        """输入：引擎名。输出：InstructorEngine 实例。"""
        self.name = name
        self._clients: dict[tuple[str, str, str], object] = {}

    def _client(self, request: EngineRequest) -> object:
        """输入：请求。输出：绑定到该 (端点, key, 档位) 的 instructor 客户端。

        预期行为：命中缓存直接返回；未命中则新建并缓存。缓存键里的 key 只存指纹。
        """
        if request.mode not in MODE_TO_INSTRUCTOR:
            raise ValueError(f"未知的结构化输出档位：{request.mode!r}")
        cache_key = (request.base_url, fingerprint(request.api_key), request.mode)
        client = self._clients.get(cache_key)
        if client is None:
            raw = openai.OpenAI(
                api_key=request.api_key,
                base_url=request.base_url,
                timeout=request.timeout_s,
            )
            client = instructor.from_openai(raw, mode=MODE_TO_INSTRUCTOR[request.mode])
            self._clients[cache_key] = client
        return client

    def extract(self, request: EngineRequest) -> EngineResult:
        """输入：一次请求。输出：EngineResult。

        预期行为：校验失败由 instructor 按 `max_retries` 回喂重试；重试耗尽抛
        `InstructorRetryException`，这里统一转成 `DraftInvalidError`。
        **永不传 `max_tokens`**，请求体里没这个字段、也没这个开关（PLAN 3.1.2）。
        """
        client = self._client(request)
        messages: list[dict[str, JsonValue]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        try:
            parsed, raw = client.create_with_completion(
                response_model=request.response_model,
                messages=messages,
                model=request.model,
                max_retries=request.max_retries,
            )
        except Exception as exc:  # noqa: BLE001 —— 一律分类后再抛，上层靠分类决策
            raise classify_exception(exc) from exc
        text, usage, reasoning = _split_raw(raw)
        return EngineResult(
            data=parsed.model_dump(),
            raw_text=text,
            usage=usage,
            reasoning=reasoning,
            mode=request.mode,
        )
