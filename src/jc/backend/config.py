"""后端配置：TOML → 数据类（PLAN 3.1.4 / 3.1.6）。

**key 只从环境变量读，不写进配置文件**（配置文件可能入库；`keys_env` 里放的是变量名）。

输入/输出：`load_config()` 读 TOML 返回 `BackendsConfig`；`resolve_keys()` 按 `keys_env`
从环境变量取出真实 key。预期行为：任何缺项 / 非法取值都在加载期报人话，不拖到发请求时才炸。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


from .types import ALL_MODES, MODE_AUTO


@dataclass(frozen=True)
class Limits:
    """一个 provider 的限额配置（对应 PLAN 3.1.6 的 `[provider.limits]`）。

    全部字段都有初值；初值只是**起点**，不是真理——服务商口径拿不到时（如 NVIDIA NIM
    不回 `x-ratelimit-*`），只能靠配置填 + 429 自适应降速。
    """

    max_concurrency: int = 4
    max_concurrency_per_key: int = 1
    requests_per_minute: int = 40
    min_interval_ms: int = 0
    cooldown_s: float = 60.0
    max_attempts: int = 4
    fail_max: int = 3

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任何非法取值立刻报错，不带着坏配置往下跑。"""
        if self.max_concurrency < 1:
            raise ValueError(f"max_concurrency 必须 >= 1，得到 {self.max_concurrency}")
        if self.max_concurrency_per_key < 1:
            raise ValueError(f"max_concurrency_per_key 必须 >= 1，得到 {self.max_concurrency_per_key}")
        if self.requests_per_minute < 0:
            raise ValueError(f"requests_per_minute 不能为负，得到 {self.requests_per_minute}")
        if self.min_interval_ms < 0:
            raise ValueError(f"min_interval_ms 不能为负，得到 {self.min_interval_ms}")
        if self.cooldown_s < 0:
            raise ValueError(f"cooldown_s 不能为负，得到 {self.cooldown_s}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts 必须 >= 1，得到 {self.max_attempts}")
        if self.fail_max < 1:
            raise ValueError(f"fail_max 必须 >= 1，得到 {self.fail_max}")


@dataclass(frozen=True)
class ProviderConfig:
    """一个 provider 的静态配置。每家可以配不同 model（PLAN 3.1.4）。

    字段：`mode` 为 `auto` 或三档之一；`keys_env` 是**环境变量名**（不是 key 本身）。
    """

    name: str
    base_url: str
    model: str
    mode: str = MODE_AUTO
    keys_env: str | None = None
    limits: Limits = field(default_factory=Limits)
    timeout_s: float = 120.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：名字 / 地址 / 模型非空，档位合法，超时为正。"""
        if not self.name.strip():
            raise ValueError("provider 的 name 不能为空")
        if not self.base_url.strip():
            raise ValueError(f"provider {self.name!r} 的 base_url 不能为空")
        if not self.model.strip():
            raise ValueError(f"provider {self.name!r} 的 model 不能为空")
        if self.mode not in ALL_MODES:
            raise ValueError(f"provider {self.name!r} 的 mode 非法：{self.mode!r}")
        if self.timeout_s <= 0:
            raise ValueError(f"provider {self.name!r} 的 timeout_s 必须为正")
        if self.max_retries < 0:
            raise ValueError(f"provider {self.name!r} 的 max_retries 不能为负")


@dataclass(frozen=True)
class BackendsConfig:
    """整份后端配置：有序的 provider 列表 + 是否允许同一 run 混多家（PLAN 3.1.4）。"""

    providers: tuple[ProviderConfig, ...]
    allow_mixed_providers: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：至少一家；名字不得重复。"""
        if not self.providers:
            raise ValueError("至少要配置一个 provider")
        names = [p.name for p in self.providers]
        if len(set(names)) != len(names):
            raise ValueError(f"provider 名字重复：{names}")


def _limits_from(raw: Mapping[str, JsonValue], where: str) -> Limits:
    """输入：`[provider.limits]` 的原始 dict 与报错位置。输出：Limits。

    预期行为：只认识的键才收，未知键报错（防拼写错误被静默忽略）。
    """
    known = set(Limits.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{where} 里有未知的限额键：{sorted(unknown)}（可用：{sorted(known)}）")
    return Limits(**raw)


def load_config(path: str | Path) -> BackendsConfig:
    """输入：TOML 配置文件路径。输出：BackendsConfig。

    预期行为：文件不存在 / 语法错 / 缺 `[[provider]]` 都报人话；未知顶层键也报错。
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"找不到后端配置文件：{target}")
    with target.open("rb") as handle:
        raw = tomllib.load(handle)

    unknown_top = set(raw) - {"provider", "allow_mixed_providers"}
    if unknown_top:
        raise ValueError(f"配置里有未知的顶层键：{sorted(unknown_top)}")

    raw_providers = raw.get("provider")
    if not isinstance(raw_providers, list) or not raw_providers:
        raise ValueError("配置里至少要有一个 [[provider]] 段")

    providers: list[ProviderConfig] = []
    for index, item in enumerate(raw_providers):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index + 1} 个 [[provider]] 不是一个表")
        item = dict(item)
        where = f"第 {index + 1} 个 [[provider]]"
        limits = _limits_from(item.pop("limits", {}), where)
        known = set(ProviderConfig.__dataclass_fields__) - {"limits"}
        unknown = set(item) - known
        if unknown:
            raise ValueError(f"{where} 里有未知的键：{sorted(unknown)}（可用：{sorted(known | {'limits'})}）")
        providers.append(ProviderConfig(limits=limits, **item))

    allow_mixed = raw.get("allow_mixed_providers", False)
    if not isinstance(allow_mixed, bool):
        raise ValueError("allow_mixed_providers 必须是布尔值")
    return BackendsConfig(providers=tuple(providers), allow_mixed_providers=allow_mixed)


def resolve_keys(config: ProviderConfig, env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """输入：一个 provider 配置、环境变量映射（默认 `os.environ`）。输出：该家的 key 元组。

    预期行为：`keys_env` 为空或变量不存在 / 为空 → 报人话并点名变量名；
    key 用逗号分隔，自动去空白与空项。**错误信息里绝不回显 key 本身**。
    """
    source = os.environ if env is None else env
    if not config.keys_env:
        raise ValueError(f"provider {config.name!r} 没配 keys_env，拿不到 key")
    raw = source.get(config.keys_env, "")
    keys = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not keys:
        raise ValueError(
            f"provider {config.name!r} 的环境变量 {config.keys_env} 为空或没设置（key 用逗号分隔）"
        )
    return keys


def resolve_all_keys(
    config: BackendsConfig, env: Mapping[str, str] | None = None
) -> tuple[tuple[str, ...], ...]:
    """输入：整份配置、环境变量映射。输出：与 `config.providers` 一一对应的 key 元组。"""
    return tuple(resolve_keys(p, env) for p in config.providers)
