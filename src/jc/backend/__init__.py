"""jc.backend：LLM 生成后端（PLAN 3.1）。

分层：
- `types`：任务 / 草稿 / 能力 / **错误分类**（熔断的判据）；
- `keys`：多 key 负载均衡（最少在途优先）；
- `config`：TOML 配置；
- `provider`：provider = key 池 + 限速（pyrate-limiter）+ 熔断（自己写的 `breaker.Breaker`）；
- `engine`：`instructor` 包装（结构化输出三档降级）；
- `probe`：端点能力探测；
- `progress`：断点续跑（追加写、先产物后进度）；
- `llm`：把上面这些拼成一个 `GenerationBackend`。

本包依赖 `instructor` / `pydantic` / `openai` / `pyrate-limiter`，
装在可选 extra `[llm]` 里；核心层（`jc.contract` / `jc.validate` / `jc.jsonio`）不依赖它们。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Final

from ..optional import llm_extra_hint
from .config import (
    BackendsConfig,
    Limits,
    ProviderConfig,
    load_config,
    resolve_all_keys,
    resolve_keys,
)
from .keys import KeyPool, fingerprint
from .progress import (
    STATUS_FAILED,
    STATUS_INVALID,
    STATUS_OK,
    ProgressEntry,
    ProgressLog,
    RunSink,
)
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

#: 取用时才 import 的名字 → (子模块, 该子模块里的名字)。
#:
#: 这些子模块要 `instructor` / `openai` / `pyrate-limiter` / `tenacity`（可选 extra `[llm]`），
#: 所以**不在这里 import**：`jc.pipeline` 只用 `backend.types` 里的类型标注，模块级 import
#: 会把整包拖进来，于是「核心层与纯构造 workflow 只用标准库」这句承诺在裸装（只
#: `pip install -e .`）下会变成一句 `ModuleNotFoundError: No module named 'instructor'`——
#: 连 `python -m jc workflows` 都起不来（2026-10-04 冒烟实测）。
#: `__getattr__` 是 PEP 562 给模块留的钩子：属性取不到时才走它，所以「只有真的要用后端」
#: 才会付出装依赖的代价；这不是反射，是标准库认的模块级协议。
LAZY_NAMES: Final[dict[str, tuple[str, str]]] = {
    "Engine": (".engine", "Engine"),
    "EngineRequest": (".engine", "EngineRequest"),
    "EngineResult": (".engine", "EngineResult"),
    "InstructorEngine": (".engine", "InstructorEngine"),
    "classify_exception": (".engine", "classify_exception"),
    "LLMBackend": (".llm", "LLMBackend"),
    "build_backend": (".llm", "build_backend"),
    "probe_capabilities": (".probe", "probe_capabilities"),
    "Provider": (".provider", "Provider"),
    "ProviderChain": (".provider", "ProviderChain"),
}

if TYPE_CHECKING:  # 只为类型检查与读代码：运行时由下面的 __getattr__ 供货
    from .engine import Engine, EngineRequest, EngineResult, InstructorEngine, classify_exception
    from .llm import LLMBackend, build_backend
    from .probe import probe_capabilities
    from .provider import Provider, ProviderChain


def __getattr__(name: str) -> object:
    """输入：属性名。输出：那个名字（必要时才 import 对应的子模块）。

    预期行为：名字在 `LAZY_NAMES` 里就 import 它的子模块；缺 `[llm]` 时报**人话**
    （指名缺哪个包、怎么装），而不是裸的 `ModuleNotFoundError`；其余名字照常报
    `AttributeError`——不报的话 `hasattr` / `getattr(..., default)` 这类探测会永远为真。
    """
    target = LAZY_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    try:
        module = importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        hint = llm_extra_hint(exc)
        if hint is None:
            raise
        raise ModuleNotFoundError(f"{hint}（原报错：{exc}）", name=exc.name) from exc
    return getattr(module, attribute)

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
