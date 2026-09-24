"""LLM 后端的降级与断点续跑测试（PLAN 3.1.1 / 3.1.4 / 3.1.5）。

只测「决策顺序」与「记账」：档位降级、provider 降级、全挂才停、先写产物再写进度。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from jc.backend import llm as llm_module
from jc.backend.config import BackendsConfig, Limits, ProviderConfig, load_config
from jc.backend.keys import fingerprint
from jc.backend.llm import LLMBackend, build_backend
from jc.backend.progress import STATUS_OK, ProgressEntry, RunSink
from jc.backend.provider import ProviderChain
from jc.backend.types import (
    AllProvidersDown,
    AuthError,
    BackendCapabilities,
    DraftInvalidError,
    SchemaUnsupportedError,
    TransportError,
)


def test_generate_returns_draft_with_full_provenance(
    fake_engine: Any, provider_factory: Any, task_factory: Any, result_factory: Any
) -> None:
    """草稿必须带够来源信息，事后才能按 provider / model / key 切片查脏数据。"""
    provider = provider_factory(
        fake_engine(script=[result_factory(mode="json_schema")]), name="a", model="m1"
    )
    backend = LLMBackend(ProviderChain([provider]))
    draft = backend.generate(task_factory())
    assert draft.backend == "llm"
    assert draft.provider == "a"
    assert draft.model == "m1"
    assert draft.mode == "json_schema"
    assert draft.key_fingerprint == fingerprint("k1")
    assert draft.attempts == 1
    assert backend.providers == ["a"]


def test_mode_degrades_when_the_server_rejects_the_schema(
    fake_engine: Any, provider_factory: Any, task_factory: Any, result_factory: Any
) -> None:
    """服务端不吃 json_schema 就降到 tools，同一个 provider 内部解决。"""
    engine = fake_engine(script=[SchemaUnsupportedError("400"), result_factory(mode="tools")])
    provider = provider_factory(engine, name="a")
    backend = LLMBackend(ProviderChain([provider]))
    draft = backend.generate(task_factory())
    assert draft.mode == "tools"
    assert draft.provider == "a"
    assert len(engine.calls) == 2


def test_mode_degrades_all_the_way_to_prompt_json(
    fake_engine: Any, provider_factory: Any, task_factory: Any, result_factory: Any
) -> None:
    """三档全试一遍：前两档都被拒，最后落到 md_json。"""
    engine = fake_engine(
        script=[
            SchemaUnsupportedError("400"),
            SchemaUnsupportedError("400"),
            result_factory(mode="md_json"),
        ]
    )
    backend = LLMBackend(ProviderChain([provider_factory(engine, name="a")]))
    assert backend.generate(task_factory()).mode == "md_json"


def test_provider_is_dropped_when_every_mode_is_rejected(
    fake_engine: Any, provider_factory: Any, task_factory: Any
) -> None:
    """三档全被服务端拒掉 = 这家没戏；只有一家时就是「全部 provider 都不可用」。"""
    engine = fake_engine(script=[SchemaUnsupportedError("400")] * 3)
    backend = LLMBackend(ProviderChain([provider_factory(engine, name="a")]))
    with pytest.raises(AllProvidersDown):
        backend.generate(task_factory())
    assert len(engine.calls) == 3


def test_provider_degrades_to_the_next_one(
    fake_engine: Any, provider_factory: Any, task_factory: Any, result_factory: Any
) -> None:
    """前一家真挂了就顺延下一家，并在草稿里记下究竟是谁出的。"""
    first = provider_factory(fake_engine(script=[TransportError("500")]), name="a", model="m1")
    second = provider_factory(fake_engine(script=[result_factory()]), name="b", model="m2")
    backend = LLMBackend(ProviderChain([first, second]))
    draft = backend.generate(task_factory())
    assert draft.provider == "b"
    assert draft.model == "m2"


def test_model_misbehaviour_moves_to_the_next_provider(
    fake_engine: Any, provider_factory: Any, task_factory: Any, result_factory: Any
) -> None:
    """模型答非所问属于「草稿不合格」，不算 provider 故障，但这家这题救不回来了。"""
    first = provider_factory(fake_engine(script=[DraftInvalidError("答非所问")]), name="a")
    second = provider_factory(fake_engine(script=[result_factory()]), name="b")
    backend = LLMBackend(ProviderChain([first, second]))
    assert backend.generate(task_factory()).provider == "b"
    assert first.breaker_state == "closed"


def test_all_providers_down_when_no_key_is_left(
    fake_engine: Any, provider_factory: Any, task_factory: Any
) -> None:
    """key 全废 → 这家不可用 → 整批停下的唯一条件。"""
    provider = provider_factory(fake_engine(script=[AuthError("401")]), name="a", keys=("k1",))
    with pytest.raises(AuthError):
        provider.extract(task_factory(), "json_schema")
    assert provider.available is False
    backend = LLMBackend(ProviderChain([provider]))
    with pytest.raises(AllProvidersDown, match="没有可用的 provider"):
        backend.generate(task_factory())


def test_all_providers_down_when_every_provider_fails(
    fake_engine: Any, provider_factory: Any, task_factory: Any
) -> None:
    """两家都没成 → 报「全部 provider 都不可用」，并把名单带上方便排查。"""
    first = provider_factory(fake_engine(script=[DraftInvalidError("答非所问")]), name="a")
    second = provider_factory(fake_engine(script=[DraftInvalidError("答非所问")]), name="b")
    backend = LLMBackend(ProviderChain([first, second]))
    with pytest.raises(AllProvidersDown, match="全部 provider 都不可用"):
        backend.generate(task_factory())


def test_generate_many_skips_finished_tasks(
    tmp_path: Any, fake_engine: Any, provider_factory: Any, task_factory: Any, result_factory: Any
) -> None:
    """重启续跑：进度里已有的 task_id 直接跳过，不再花配额。"""
    sink = RunSink(tmp_path / "data.jsonl", tmp_path / "progress.jsonl")
    sink.progress.append(ProgressEntry(task_id="t1", status=STATUS_OK, row=1))
    engine = fake_engine(script=[result_factory()])
    backend = LLMBackend(ProviderChain([provider_factory(engine, name="a")]), sink=sink)
    drafts = backend.generate_many([task_factory("t1"), task_factory("t2")])
    assert [d.provider for d in drafts] == ["a"]
    assert len(engine.calls) == 1


def test_generate_many_writes_sample_then_progress(
    tmp_path: Any, fake_engine: Any, provider_factory: Any, task_factory: Any, result_factory: Any
) -> None:
    """顺序铁律：产物先落盘，进度后落盘；进度里的 row 回填成产物行号。"""
    sink = RunSink(tmp_path / "data.jsonl", tmp_path / "progress.jsonl")
    backend = LLMBackend(
        ProviderChain([provider_factory(fake_engine(script=[result_factory()]), name="a")]),
        sink=sink,
    )
    backend.generate_many(
        [task_factory("t1")], assemble=lambda task, draft: {"id": task.task_id, "draft": draft.data}
    )
    lines = (tmp_path / "data.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0]) == {"id": "t1", "draft": {"answer": "x"}}
    entries = sink.progress.entries()
    assert len(entries) == 1
    assert entries[0].row == 1
    assert entries[0].provider == "a"
    assert sink.progress.completed()["t1"].status == STATUS_OK


def test_generate_many_without_assemble_uses_a_plain_sample(
    tmp_path: Any, fake_engine: Any, provider_factory: Any, task_factory: Any, result_factory: Any
) -> None:
    """没给 assemble 时就写「task_id + 草稿数据」，够调试用。"""
    sink = RunSink(tmp_path / "data.jsonl", tmp_path / "progress.jsonl")
    backend = LLMBackend(
        ProviderChain([provider_factory(fake_engine(script=[result_factory()]), name="a")]),
        sink=sink,
    )
    backend.generate_many([task_factory("t1")])
    line = json.loads((tmp_path / "data.jsonl").read_text(encoding="utf-8").strip())
    assert line == {"task_id": "t1", "draft": {"answer": "x"}}


def test_generate_many_keeps_progress_when_a_task_fails(
    tmp_path: Any, fake_engine: Any, provider_factory: Any, task_factory: Any, result_factory: Any
) -> None:
    """中途炸掉时已写下的进度必须保留，下次重启从断点接着跑。"""
    sink = RunSink(tmp_path / "data.jsonl", tmp_path / "progress.jsonl")
    engine = fake_engine(script=[result_factory(), DraftInvalidError("答非所问")])
    backend = LLMBackend(ProviderChain([provider_factory(engine, name="a")]), sink=sink)
    with pytest.raises(AllProvidersDown):
        backend.generate_many([task_factory("t1"), task_factory("t2")])
    assert list(sink.progress.completed()) == ["t1"]
    assert sink.rows == 1


def test_backend_without_sink_just_returns_drafts(
    fake_engine: Any, provider_factory: Any, task_factory: Any, result_factory: Any
) -> None:
    """不落盘时 generate_many 只负责逐条生成，不产生任何文件副作用。"""
    engine = fake_engine(script=[result_factory(), result_factory()])
    backend = LLMBackend(ProviderChain([provider_factory(engine, name="a")]))
    assert len(backend.generate_many([task_factory("t1"), task_factory("t2")])) == 2


# ---- 装配工厂 build_backend：配置 → 可直接用的后端 ------------------------------


def provider_config(name: str, model: str, keys_env: str | None = None) -> ProviderConfig:
    """输入：名字、模型、key 的环境变量名。输出：一个 provider 配置。

    预期行为：限速与重试全关，测试不真等、不被限速拖慢。
    """
    return ProviderConfig(
        name=name,
        base_url="https://example.test/v1",
        model=model,
        keys_env=keys_env or f"JC_{name.upper()}_KEYS",
        limits=Limits(requests_per_minute=0, min_interval_ms=0, max_attempts=1),
    )


class DispatchEngine:
    """按请求里的 model 分派的假引擎，用来在一个共享引擎下区分不同 provider。"""

    name = "dispatch"

    def __init__(self, by_model: dict[str, Any]) -> None:
        """输入：{模型名: 结果或异常实例}。输出：DispatchEngine 实例。"""
        self.by_model = by_model
        self.models: list[str] = []

    def extract(self, request: Any) -> Any:
        """输入：一次请求。输出：该模型对应的结果（或抛对应的异常）。"""
        self.models.append(request.model)
        item = self.by_model[request.model]
        if isinstance(item, BaseException):
            raise item
        return item


def test_build_backend_wires_config_to_backend() -> None:
    """配置里的每家 provider 都要变成链上的一环，顺序与 model 原样保留。"""
    config = BackendsConfig(
        providers=(provider_config("nim", "openai/gpt-oss-20b"), provider_config("spare", "m2"))
    )
    env = {"JC_NIM_KEYS": "k1,k2", "JC_SPARE_KEYS": "k3"}
    backend = build_backend(config, env=env, sleep=lambda _s: None)
    assert backend.providers == ["nim", "spare"]


def test_build_backend_rejects_a_provider_without_keys() -> None:
    """少一个环境变量就当场报人话点名，不静默变成「偷偷只用一半 provider」。"""
    config = BackendsConfig(
        providers=(provider_config("nim", "m1"), provider_config("spare", "m2"))
    )
    with pytest.raises(ValueError, match="JC_SPARE_KEYS"):
        build_backend(config, env={"JC_NIM_KEYS": "k1"}, sleep=lambda _s: None)


def test_build_backend_never_echoes_the_key_in_errors() -> None:
    """报错信息里绝不许出现 key 明文。"""
    config = BackendsConfig(providers=(provider_config("nim", "m1"),))
    with pytest.raises(ValueError) as info:
        build_backend(config, env={}, sleep=lambda _s: None)
    assert "nvapi-" not in str(info.value)


def test_build_backend_probe_only_shapes_the_starting_mode(
    monkeypatch: pytest.MonkeyPatch, result_factory: Any, task_factory: Any
) -> None:
    """探测结果只决定从哪一档起步：探出 tools 就从 tools 起，且只用第一个 key 探。"""
    calls: list[dict[str, Any]] = []

    def fake_probe(**kwargs: Any) -> BackendCapabilities:
        calls.append(kwargs)
        return BackendCapabilities(tools=True, notes=("tools: 可用",))

    seen: list[dict[str, Any]] = []
    real_provider = llm_module.Provider

    def spy_provider(config: Any, keys: Any, engine: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return real_provider(config, keys, engine, **kwargs)

    monkeypatch.setattr(llm_module, "probe_capabilities", fake_probe)
    monkeypatch.setattr(llm_module, "Provider", spy_provider)
    config = BackendsConfig(providers=(provider_config("nim", "m1"),))
    engine = DispatchEngine({"m1": result_factory(mode="tools")})
    backend = build_backend(
        config, engine=engine, probe=True, env={"JC_NIM_KEYS": "k1,k2"}, sleep=lambda _s: None
    )
    assert calls == [
        {"base_url": "https://example.test/v1", "api_key": "k1", "model": "m1", "timeout_s": 120.0}
    ]
    assert seen[0]["capabilities"].tools is True
    assert backend.capabilities.tools is True
    assert backend.generate(task_factory()).mode == "tools"


def test_build_backend_without_probe_starts_from_the_strictest_mode(
    result_factory: Any, task_factory: Any
) -> None:
    """不探测时从最严的 json_schema 起步，且不发任何探测请求。"""
    engine = DispatchEngine({"m1": result_factory(mode="json_schema")})
    config = BackendsConfig(providers=(provider_config("nim", "m1"),))
    backend = build_backend(config, engine=engine, env={"JC_NIM_KEYS": "k1"}, sleep=lambda _s: None)
    assert backend.generate(task_factory()).mode == "json_schema"
    assert len(engine.models) == 1


def test_build_backend_stops_at_the_first_provider_by_default(
    result_factory: Any, task_factory: Any
) -> None:
    """默认「一家到底」：各家 model 不同时不偷偷换模型，前一家挂了整批停。"""
    engine = DispatchEngine({"m1": TransportError("500"), "m2": result_factory()})
    config = BackendsConfig(
        providers=(provider_config("a", "m1"), provider_config("b", "m2"))
    )
    backend = build_backend(
        config, engine=engine, env={"JC_A_KEYS": "k1", "JC_B_KEYS": "k2"}, sleep=lambda _s: None
    )
    with pytest.raises(AllProvidersDown):
        backend.generate(task_factory())
    assert engine.models == ["m1"]  # 第二家根本没被打扰


def test_build_backend_can_opt_into_mixing_models(result_factory: Any, task_factory: Any) -> None:
    """显式打开混用后，前一家挂了才顺延到不同 model 的那家。"""
    engine = DispatchEngine({"m1": TransportError("500"), "m2": result_factory()})
    config = BackendsConfig(
        providers=(provider_config("a", "m1"), provider_config("b", "m2")),
        allow_mixed_providers=True,
    )
    backend = build_backend(
        config, engine=engine, env={"JC_A_KEYS": "k1", "JC_B_KEYS": "k2"}, sleep=lambda _s: None
    )
    assert backend.generate(task_factory()).provider == "b"


def test_build_backend_reads_a_real_toml_file(tmp_path: Any) -> None:
    """端到端：TOML 配置 → load_config → build_backend，key 只从环境变量来。"""
    toml_path = tmp_path / "backends.toml"
    toml_path.write_text(
        "\n".join(
            [
                "allow_mixed_providers = true",
                "",
                "[[provider]]",
                'name = "nim"',
                'base_url = "https://integrate.api.nvidia.com/v1"',
                'model = "openai/gpt-oss-20b"',
                'keys_env = "JC_NIM_KEYS"',
                "",
                "  [provider.limits]",
                "  requests_per_minute = 0",
                "  min_interval_ms = 0",
                "",
                "[[provider]]",
                'name = "spare"',
                'base_url = "https://example.test/v1"',
                'model = "m2"',
                'keys_env = "JC_SPARE_KEYS"',
                "",
                "  [provider.limits]",
                "  requests_per_minute = 0",
                "  min_interval_ms = 0",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(toml_path)
    backend = build_backend(
        config, env={"JC_NIM_KEYS": "k1", "JC_SPARE_KEYS": "k2"}, sleep=lambda _s: None
    )
    assert backend.providers == ["nim", "spare"]
    assert config.allow_mixed_providers is True
