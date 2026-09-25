"""LLM 后端：把 provider 链 + 引擎 + 进度拼成一个 `GenerationBackend`（PLAN 3.1）。

单次调用的决策顺序：

```
按顺序遍历 provider（3.1.4：全挂才停）
  └─ 按档位从严到松遍历（3.1.1：JSON_SCHEMA → TOOLS → MD_JSON）
       ├─ 服务端不吃这一档  → 降档继续
       ├─ provider 熔断 / key 用尽 → 换下一家
       ├─ 网络 / 5xx / 429（已重试用尽）→ 换下一家
       └─ 模型答非所问 → 换下一家；全都没给出合格草稿时抛 DraftInvalidError（这一条被拒，
          整批继续），只有「一家都没活着」才抛 AllProvidersDown（整批停下）
```

输入/输出：`generate()` 收任务出 `Draft`；`generate_many()` 额外负责断点续跑记账。
预期行为：`Draft` 里一定带 `provider` / `model` / `mode` / key 指纹，事后能按来源切片。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .config import BackendsConfig, resolve_keys
from .engine import Engine, InstructorEngine
from .probe import probe_capabilities
from .progress import STATUS_OK, ProgressEntry, RunSink
from .provider import Provider, ProviderChain
from .types import (
    AllProvidersDown,
    AuthError,
    BackendCapabilities,
    BackendError,
    Draft,
    DraftInvalidError,
    GenerationTask,
    ProviderUnavailable,
    SchemaUnsupportedError,
    TransportError,
)

# 把一个任务 + 草稿变成契约样本的函数（由 workflow / Assembler 提供）。
Assemble = Callable[[GenerationTask, Draft], dict[str, Any]]


class LLMBackend:
    """基于 provider 链的生成后端（实现 `GenerationBackend` 协议）。

    参数：`chain` provider 链；`engine` 引擎（已经绑在 Provider 上，这里只为签名对称）；
    `sink` 可选的产物 + 进度写入口；`clock` 单调时钟（测试可注入）。
    """

    name = "llm"

    def __init__(
        self,
        chain: ProviderChain,
        engine: Engine | None = None,
        *,
        sink: RunSink | None = None,
        capabilities: BackendCapabilities | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """输入：链、引擎、写入口、能力、时钟。输出：LLMBackend 实例。"""
        del engine
        self._chain = chain
        self._sink = sink
        self.capabilities = capabilities or BackendCapabilities()
        self._clock = clock

    @property
    def providers(self) -> list[str]:
        """按顺序列出 provider 名。"""
        return self._chain.names

    @property
    def starting_mode(self) -> str:
        """输入：无。输出：这一批**真正**会最先尝试的档位。

        预期行为：取自链上第一家的档位序列（配置写死了档位时就是那一档）。
        **不能拿 `capabilities.starting_mode()` 顶替它**——那是探测结论，探测没开时
        一律是 `md_json`，会把实际从 `json_schema` 起步的 run 记成 `md_json`。
        """
        first = next(iter(self._chain), None)
        return first.modes()[0] if first is not None else ""

    def generate(self, task: GenerationTask) -> Draft:
        """输入：一个任务。输出：草稿。

        预期行为：按 3.1.1 的档位降级与 3.1.4 的 provider 降级依次尝试。
        全部 provider 都**挂了**（熔断 / key 用尽 / 网络）才抛 `AllProvidersDown`——
        这是整批停下的唯一条件；若只是「provider 都活着、但都给不出合格草稿」，
        抛 `DraftInvalidError`（`fatal_for_run` 为假），由管线记成这一条被拒、整批继续。

        这两件事必须分开：实测踩坑——kimi-k3 在 `tools` 档偶尔不回 `tool_calls`，
        `instructor` 抛 `IndexError` 再包成 `InstructorRetryException`，被这里误判成
        「全部 provider 都不可用」，6 条样本跑到第 5 条把整批掐了（端点当场验证是健康的）。
        """
        providers = self._chain.usable()
        if not providers:
            raise AllProvidersDown(f"没有可用的 provider（配置了 {self._chain.names}）")
        last: BackendError | None = None
        junk: DraftInvalidError | None = None
        for provider in providers:
            try:
                draft = self._try_provider(provider, task)
            except DraftInvalidError as exc:
                junk = exc  # 这家活着，只是没给出合格草稿：换下一家，但不算「挂了」
                continue
            if draft is not None:
                return draft
            last = ProviderUnavailable(f"{provider.name} 这次没成")
        if junk is not None:
            raise junk
        raise AllProvidersDown(f"全部 provider 都不可用：{self._chain.names}") from last

    def _try_provider(self, provider: Provider, task: GenerationTask) -> Draft | None:
        """输入：一个 provider 与任务。输出：草稿，或 None（这家没成，该换下一家）。

        预期行为：档位从严到松依次试；「不吃这一档」继续降档，provider 级失败立刻换下一家。
        **`DraftInvalidError` 直接往外抛**——provider 是活的，该由 `generate` 决定它是
        「换下一家」还是「整批停」，这里不替它记账。
        """
        for mode in provider.modes():
            try:
                result, key_fp, attempts = provider.extract(task, mode)
            except SchemaUnsupportedError:
                continue  # 服务端不吃这一档，降档重试
            except (ProviderUnavailable, TransportError, AuthError):
                return None  # 这家没戏了，换下一家
            return Draft(
                data=result.data,
                raw_text=result.raw_text,
                backend=self.name,
                provider=provider.name,
                model=provider.model,
                mode=result.mode,
                usage=result.usage,
                key_fingerprint=key_fp,
                attempts=attempts,
                reasoning=result.reasoning,
            )
        return None

    def generate_many(
        self, tasks: Sequence[GenerationTask], *, assemble: Assemble | None = None
    ) -> list[Draft]:
        """输入：任务序列、可选的「草稿 → 契约样本」函数。输出：本次新产出的草稿列表。

        预期行为：配了 `sink` 时先按进度跳过已完成的 task_id，每产出一条就
        **先写产物、再写进度**（3.1.5 的顺序铁律）。任何任务失败都直接抛出，
        已写下的进度保留，下次重启接着跑。
        """
        done = self._sink.progress.completed() if self._sink is not None else {}
        out: list[Draft] = []
        for task in tasks:
            if task.task_id in done:
                continue
            started = self._clock()
            draft = self.generate(task)
            elapsed = self._clock() - started
            out.append(draft)
            if self._sink is not None:
                sample = (
                    assemble(task, draft)
                    if assemble is not None
                    else {"task_id": task.task_id, "draft": draft.data}
                )
                self._sink.write_sample(
                    sample,
                    ProgressEntry(
                        task_id=task.task_id,
                        status=STATUS_OK,
                        provider=draft.provider,
                        model=draft.model,
                        mode=draft.mode,
                        key_fingerprint=draft.key_fingerprint,
                        attempts=draft.attempts,
                        elapsed_s=elapsed,
                        usage=draft.usage,
                    ),
                )
        return out


def build_backend(
    config: BackendsConfig,
    *,
    sink: RunSink | None = None,
    engine: Engine | None = None,
    probe: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    env: Mapping[str, str] | None = None,
) -> LLMBackend:
    """把一份后端配置装配成可以直接用的 `LLMBackend`（PLAN 3.1.4 / 3.1.6）。

    输入：整份配置、可选写入口 / 引擎 / 探测开关 / 睡眠函数 / 环境变量映射。
    输出：装配好的 `LLMBackend`。
    预期行为：
    - key **只从环境变量读**（`keys_env` 里放的是变量名）；任何一家取不到 key 就立刻报人话，
      不静默少一家——否则配置写错会变成「偷偷只用了一半 provider」；
    - 全部 provider **共用一个引擎实例**，客户端缓存（键含端点 + key 指纹 + 档位）才有效；
    - `probe=True` 时对每家探一次能力（只拿第一个 key 探，省配额），探测结果只决定该家
      从哪一档起步；探测本身不抛异常，失败只是记一条 notes；
    - 是否允许同一 run 内混不同 model，原样取自 `config.allow_mixed_providers`
      （PLAN 3.1.4：默认「一家到底」，前一家挂了整批停并保留进度）。
    """
    shared = engine if engine is not None else InstructorEngine()
    providers: list[Provider] = []
    probed: BackendCapabilities | None = None
    for item in config.providers:
        keys = resolve_keys(item, env)
        caps = (
            probe_capabilities(
                base_url=item.base_url,
                api_key=keys[0],
                model=item.model,
                timeout_s=item.timeout_s,
            )
            if probe
            else None
        )
        if probed is None:
            probed = caps
        providers.append(Provider(item, keys, shared, capabilities=caps, sleep=sleep))
    chain = ProviderChain(providers, allow_mixed=config.allow_mixed_providers)
    return LLMBackend(chain, sink=sink, capabilities=probed)
