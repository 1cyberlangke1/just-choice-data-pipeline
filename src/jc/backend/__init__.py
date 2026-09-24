"""jc.backend：LLM 生成后端（PLAN 3.1）。

分层：
- `types`：任务 / 草稿 / 能力 / **错误分类**（熔断的判据）；
- `keys`：多 key 负载均衡（最少在途优先）；
- `config`：TOML 配置；
- `provider`：provider = key 池 + 限速（pyrate-limiter）+ 熔断（pybreaker）；
- `engine`：`instructor` 包装（结构化输出三档降级）；
- `probe`：端点能力探测；
- `progress`：断点续跑（追加写、先产物后进度）；
- `llm`：把上面这些拼成一个 `GenerationBackend`。

本包依赖 `instructor` / `pydantic` / `openai` / `pyrate-limiter` / `pybreaker`，
装在可选 extra `[llm]` 里；核心层（`jc.contract` / `jc.validate` / `jc.jsonio`）不依赖它们。
"""

from __future__ import annotations

from .config import (
    BackendsConfig,
    Limits,
    ProviderConfig,
    load_config,
    resolve_all_keys,
    resolve_keys,
)
from .engine import Engine, EngineRequest, EngineResult, InstructorEngine, classify_exception
from .keys import KeyPool, fingerprint
from .llm import LLMBackend, build_backend
from .probe import probe_capabilities
from .progress import (
    STATUS_FAILED,
    STATUS_INVALID,
    STATUS_OK,
    ProgressEntry,
    ProgressLog,
    RunSink,
)
from .provider import Provider, ProviderChain
from .types import (
    ALL_MODES,
    MODE_AUTO,
    MODE_CHAIN,
    MODE_JSON_SCHEMA,
    MODE_MD_JSON,
    MODE_TOOLS,
    AllProvidersDown,
    AuthError,
    BackendCapabilities,
    BackendError,
    Draft,
    DraftInvalidError,
    GenerationBackend,
    GenerationTask,
    ProviderUnavailable,
    RateLimitError,
    SchemaUnsupportedError,
    Seed,
    TransportError,
    modes_from,
)

__all__ = [
    "ALL_MODES",
    "AllProvidersDown",
    "AuthError",
    "BackendCapabilities",
    "BackendError",
    "BackendsConfig",
    "Draft",
    "DraftInvalidError",
    "Engine",
    "EngineRequest",
    "EngineResult",
    "GenerationBackend",
    "GenerationTask",
    "InstructorEngine",
    "KeyPool",
    "LLMBackend",
    "Limits",
    "MODE_AUTO",
    "MODE_CHAIN",
    "MODE_JSON_SCHEMA",
    "MODE_MD_JSON",
    "MODE_TOOLS",
    "ProgressEntry",
    "ProgressLog",
    "Provider",
    "ProviderChain",
    "ProviderConfig",
    "ProviderUnavailable",
    "RateLimitError",
    "RunSink",
    "STATUS_FAILED",
    "STATUS_INVALID",
    "STATUS_OK",
    "SchemaUnsupportedError",
    "Seed",
    "TransportError",
    "build_backend",
    "classify_exception",
    "fingerprint",
    "load_config",
    "modes_from",
    "probe_capabilities",
    "resolve_all_keys",
    "resolve_keys",
]
