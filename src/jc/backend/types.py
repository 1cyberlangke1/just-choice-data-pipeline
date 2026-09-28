"""后端层公共类型：任务、草稿、能力、协议与**错误分类**（PLAN 3.1）。

三条要点：
- 后端**只出草稿**，绝不碰契约（编号 / key / `type` 只由 Assembler 产生）；
- 错误必须分类：熔断（3.1.3）要区分「provider 故障」与「草稿不合格」。把「模型答错」
  算成故障，会把一个正常但能力弱的 provider 误熔断。判据就是类属性 `counts_as_fault`；
- 采样参数**永不发 `max_tokens`**、也不给开关（3.1.2）；推理字段单独存、不回喂。

输入/输出：本模块只有数据类、协议常量与异常类型；无副作用、不发网络请求。
**刻意不 import pydantic**：草稿模型是 workflow 提供的，这里只用 `type[StructuredModel]` 作注解，
这样核心层即使没装 `[llm]` extra 也能 import 本模块。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

# ---- 结构化输出的三档（PLAN 3.1.1）------------------------------------------

MODE_JSON_SCHEMA: Final[str] = "json_schema"
MODE_TOOLS: Final[str] = "tools"
MODE_MD_JSON: Final[str] = "md_json"
MODE_AUTO: Final[str] = "auto"

#: 降级顺序：服务端强约束 → 工具调用 → 提示词 + 本地校验。
MODE_CHAIN: Final[tuple[str, ...]] = (MODE_JSON_SCHEMA, MODE_TOOLS, MODE_MD_JSON)

ALL_MODES: Final[tuple[str, ...]] = (MODE_AUTO, *MODE_CHAIN)


def modes_from(mode: str) -> tuple[str, ...]:
    """输入：配置的档位（`auto` 或三档之一）。输出：从该档降到 `MD_JSON` 的档位序列。

    预期行为：`auto` 从最严的 `json_schema` 起步；未知档位直接报错，不静默兜底。
    """
    if mode == MODE_AUTO:
        return MODE_CHAIN
    if mode not in MODE_CHAIN:
        raise ValueError(f"未知的结构化输出档位：{mode!r}（可选：{', '.join(ALL_MODES)}）")
    return MODE_CHAIN[MODE_CHAIN.index(mode) :]


# ---- 错误分类（熔断的判据）--------------------------------------------------


class BackendError(Exception):
    """后端层所有错误的基类。

    类属性 `counts_as_fault` 决定这次失败是否计入 provider 熔断计数（PLAN 3.1.3）。
    """

    counts_as_fault: bool = False
    #: 这次失败是不是「整批该停下」。通用层（`jc.pipeline`）只看这个类属性、
    #: 不 import 任何后端模块，所以任何后端实现都能用同一套判据。
    fatal_for_run: bool = False


class TransportError(BackendError):
    """网络超时 / 连接失败 / 5xx。**算 provider 故障**，可重试。"""

    counts_as_fault = True


class RateLimitError(TransportError):
    """429。**算 provider 故障**，可重试，且优先换 key。"""

    counts_as_fault = True

    def __init__(self, msg: str, retry_after_s: float | None = None) -> None:
        """输入：人话说明、可选的 `Retry-After` 秒数。输出：异常实例。"""
        super().__init__(msg)
        self.retry_after_s = retry_after_s


class AuthError(BackendError):
    """401 / 403：这个 key 不行。**不算 provider 故障**（换个 key 可能就好）。"""


class SchemaUnsupportedError(BackendError):
    """400 / 422：服务端不吃这一档 schema。**不算故障**，应当降档重试。"""


class DraftInvalidError(BackendError):
    """响应拿到了，但内容不是合法草稿（解析失败 / 答非所问）。

    **绝不算 provider 故障**——否则一个正常但能力弱的 provider 会被误熔断。
    """


class ProviderUnavailable(BackendError):
    """该 provider 当前不可用（熔断打开 / key 池空）。不算故障，直接顺延下一家。"""


class AllProvidersDown(BackendError):
    """所有 provider 都不可用——这是整批停下的唯一条件（PLAN 3.1.4）。"""

    #: 整批停下：继续跑只会产出一整个「全被拒」的假数据集，还白烧配额。
    fatal_for_run = True


# ---- 数据类 -----------------------------------------------------------------


@dataclass(frozen=True)
class Seed:
    """一块素材。含义由 workflow 决定，后端只负责把它拼进提示词。

    字段：`kind` 素材类型（如 `table` / `email`），`payload` 可 JSON 序列化的内容。
    """

    kind: str
    payload: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationTask:
    """一次生成请求的完整输入。

    字段：`task_id` 全局唯一且**可由 run_seed 确定性重算**（断点续跑靠它）；
    `kind` 题型（choice/noul/score）；`prompt` 提示词；
    `response_model` **必须是一个 pydantic 模型**（instructor 的硬要求，见 3.1.1）；
    `seeds` 素材；`spec` workflow 私有的真值规格。

    属性 `schema` 由 `response_model` 现算，供能力探测与记账使用。
    """

    task_id: str
    kind: str
    domain: str
    prompt: str
    response_model: type[StructuredModel]
    seeds: tuple[Seed, ...] = ()
    spec: dict[str, JsonValue] = field(default_factory=dict)

    @property
    def schema(self) -> dict[str, JsonValue]:
        """输入：无。输出：`response_model` 的 JSON Schema（dict）。"""
        return dict(self.response_model.model_json_schema())


@dataclass(frozen=True)
class Draft:
    """后端交回来的草稿 + 记账信息。

    字段：`data` 解析后的对象；`raw_text` 原始响应文本；`provider` / `model` / `mode`
    用于事后按来源切片；`key_fingerprint` 是 key 的 sha256 前 8 位（**不存明文**）；
    `reasoning` 单独存，**不回喂**；`usage` 服务端用量；`attempts` 实际尝试次数。
    """

    data: dict[str, JsonValue]
    raw_text: str
    backend: str
    provider: str
    model: str
    mode: str
    usage: dict[str, JsonValue] = field(default_factory=dict)
    key_fingerprint: str = ""
    attempts: int = 1
    reasoning: str | None = None
    cached: bool = False


@dataclass(frozen=True)
class BackendCapabilities:
    """一个端点的能力探测结果（PLAN 3.1.1）。

    只决定「从哪一档起步」，**不替代本地校验**——探测返 200 不等于服务端真强制了 schema。
    """

    plain: bool = False
    json_object: bool = False
    json_schema: bool = False
    tools: bool = False
    notes: tuple[str, ...] = ()

    def starting_mode(self) -> str:
        """返回该端点应当起步的档位（都不支持就退到 `MD_JSON`）。"""
        if self.json_schema:
            return MODE_JSON_SCHEMA
        if self.tools:
            return MODE_TOOLS
        return MODE_MD_JSON


@runtime_checkable
class StructuredModel(Protocol):
    """结构化输出的模型：后端只用到类方法 `model_json_schema`。

    输入/输出：无（纯协议）。
    预期行为：让 `GenerationTask.response_model` 不必写成 `type[Any]`——`Any` 在本仓库
    是被禁的（见 `tests/core/test_no_any.py`）；这里也刻意不 import pydantic。
    """

    @classmethod
    def model_json_schema(cls) -> dict[str, JsonValue]: ...


@runtime_checkable
class GenerationBackend(Protocol):
    """生成后端协议：给任务，出草稿。"""

    name: str
    capabilities: BackendCapabilities

    def generate(self, task: GenerationTask) -> Draft:
        """输入：一个任务。输出：草稿。预期行为：网络与限流错误在内部消化。"""

    def generate_many(self, tasks: list[GenerationTask]) -> list[Draft]:
        """输入：任务列表。输出：草稿列表（顺序与入参一致）。"""
