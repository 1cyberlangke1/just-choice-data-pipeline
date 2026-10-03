"""LLM 后端的降级与断点续跑测试（PLAN 3.1.1 / 3.1.4 / 3.1.5）。

只测「决策顺序」与「记账」：档位降级、provider 降级、全挂才停、先写产物再写进度。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import json


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


def test_starting_mode_reports_the_configured_mode_not_the_probe(
    fake_engine: object, provider_factory: object
) -> None:
    """后端自报的起始档位必须是**真正**要试的第一档，而不是探测结论。

    输入：一个写死 `json_schema` 档位的 provider（没探测）。输出：无。
    预期行为：探测没开时 `capabilities` 是空的，`starting_mode()` 会答 `md_json`；
    拿它记账会把实际从 `json_schema` 起步的 run 记成 `md_json`（错账）。
    """
    provider = provider_factory(fake_engine(script=[]), name="a", mode="json_schema")
    backend = LLMBackend(ProviderChain([provider]))
    assert backend.capabilities.starting_mode() == "md_json"
    assert backend.starting_mode == "json_schema"


def test_generate_returns_draft_with_full_provenance(
    fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
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


def test_call_log_names_the_model_that_served_each_task(
    fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
) -> None:
    """输入：两次调用（第二次顺延到另一家）。输出：逐条流水。

    预期行为：每条都写清 task_id → provider / 模型 / 档位 / key 指纹 / 尝试次数 / 用量，
    并且按 task_id 排序——汇总计数不够用，脏数据要能**按来源切片**，落成 jsonl 才有意义。
    """
    first = provider_factory(fake_engine(script=[TransportError("500")] * 2), name="a", model="m1")
    second = provider_factory(fake_engine(script=[result_factory()] * 2), name="b", model="m2")
    backend = LLMBackend(ProviderChain([first, second]))
    backend.generate(task_factory("t1"))
    backend.generate(task_factory("t2"))
    log = backend.call_log()
    assert [row["task_id"] for row in log] == ["t1", "t2"]
    assert [(row["provider"], row["model"]) for row in log] == [("b", "m2"), ("b", "m2")]
    assert log[0]["mode"] == "json_schema"
    assert log[0]["key_fingerprint"] == fingerprint("k1")
    assert log[0]["usage"] == {"total_tokens": 3}
    # 流水是副本：外部改它不该动到后端内部记账
    log.clear()
    assert len(backend.call_log()) == 2


def test_call_stats_records_which_provider_model_and_key_served(
    fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
) -> None:
    """输入：两家 provider（前一家真挂了）+ 两次调用。输出：来源计数。

    预期行为：只统计**成功产出草稿**的那一家（挂掉的那家不该有记录）；key 指纹分开记，
    多 key 负载均衡才核得出来。一次都没调时两张表都是空的，不编造。
    """
    # 第一家的脚本给两份：两次调用都会先撞上它，再顺延到第二家
    first = provider_factory(fake_engine(script=[TransportError("500")] * 2), name="a", model="m1")
    second = provider_factory(fake_engine(script=[result_factory()] * 2), name="b", model="m2")
    backend = LLMBackend(ProviderChain([first, second]))
    assert backend.call_stats() == {"models": {}, "keys": {}}
    backend.generate(task_factory("t1"))
    backend.generate(task_factory("t2"))
    assert backend.call_stats() == {
        "models": {"b/m2": 2},
        "keys": {fingerprint("k1"): 2},
    }


def test_call_stats_splits_by_key_when_a_provider_has_several(
    fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
) -> None:
    """输入：一家 provider、两把 key、四次调用。输出：来源计数。

    预期行为：key 按轮转分摊（2 + 2），模型那一栏是 `provider/model`——`allow_mixed_providers`
    下同一批可能混多家模型，不记就没人知道这批数据混了来源。
    """
    provider = provider_factory(
        fake_engine(script=[result_factory()] * 4), name="a", keys=("k1", "k2")
    )
    backend = LLMBackend(ProviderChain([provider]))
    for index in range(4):
        backend.generate(task_factory(f"t{index}"))
    stats = backend.call_stats()
    assert stats["models"] == {"a/fake-model": 4}
    assert stats["keys"] == {fingerprint("k1"): 2, fingerprint("k2"): 2}


def test_mode_degrades_when_the_server_rejects_the_schema(
    fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
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
    fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
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
    fake_engine: object, provider_factory: object, task_factory: object
) -> None:
    """三档全被服务端拒掉 = 这家没戏；只有一家时就是「全部 provider 都不可用」。"""
    engine = fake_engine(script=[SchemaUnsupportedError("400")] * 3)
    backend = LLMBackend(ProviderChain([provider_factory(engine, name="a")]))
    with pytest.raises(AllProvidersDown):
        backend.generate(task_factory())
    assert len(engine.calls) == 3


def test_provider_degrades_to_the_next_one(
    fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
) -> None:
    """前一家真挂了就顺延下一家，并在草稿里记下究竟是谁出的。"""
    first = provider_factory(fake_engine(script=[TransportError("500")]), name="a", model="m1")
    second = provider_factory(fake_engine(script=[result_factory()]), name="b", model="m2")
    backend = LLMBackend(ProviderChain([first, second]))
    draft = backend.generate(task_factory())
    assert draft.provider == "b"
    assert draft.model == "m2"


def test_model_misbehaviour_moves_to_the_next_provider(
    fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
) -> None:
    """模型答非所问属于「草稿不合格」，不算 provider 故障，但这家这题救不回来了。"""
    first = provider_factory(fake_engine(script=[DraftInvalidError("答非所问")]), name="a")
    second = provider_factory(fake_engine(script=[result_factory()]), name="b")
    backend = LLMBackend(ProviderChain([first, second]))
    assert backend.generate(task_factory()).provider == "b"
    assert first.breaker_state == "closed"


def test_all_providers_down_when_no_key_is_left(
    fake_engine: object, provider_factory: object, task_factory: object
) -> None:
    """key 全废 → 这家不可用 → 整批停下的唯一条件。"""
    provider = provider_factory(fake_engine(script=[AuthError("401")]), name="a", keys=("k1",))
    with pytest.raises(AuthError):
        provider.extract(task_factory(), "json_schema")
    assert provider.available is False
    backend = LLMBackend(ProviderChain([provider]))
    with pytest.raises(AllProvidersDown, match="没有可用的 provider"):
        backend.generate(task_factory())


def test_every_provider_giving_junk_is_a_reject_not_a_batch_stop(
    fake_engine: object, provider_factory: object, task_factory: object
) -> None:
    """两家都活着、但都给不出合格草稿 → 这一条不合格，**不是**「全部 provider 都挂了」。

    实测踩坑：kimi 在 tools 档偶尔不回 tool_calls，instructor 抛 IndexError，
    被这里误判成 AllProvidersDown，把整批掐了（docs 黑盒API…调研 §9.11）。
    """
    first = provider_factory(fake_engine(script=[DraftInvalidError("答非所问")]), name="a")
    second = provider_factory(fake_engine(script=[DraftInvalidError("答非所问")]), name="b")
    backend = LLMBackend(ProviderChain([first, second]))
    with pytest.raises(DraftInvalidError, match="答非所问"):
        backend.generate(task_factory())
    assert first.available and second.available  # 两家都还在，谁都没被误判成挂了


def test_generate_many_skips_finished_tasks(
    tmp_path: object, fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
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
    tmp_path: object, fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
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
    tmp_path: object, fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
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
    tmp_path: object, fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
) -> None:
    """中途炸掉时已写下的进度必须保留，下次重启从断点接着跑。"""
    sink = RunSink(tmp_path / "data.jsonl", tmp_path / "progress.jsonl")
    engine = fake_engine(script=[result_factory(), DraftInvalidError("答非所问")])
    backend = LLMBackend(ProviderChain([provider_factory(engine, name="a")]), sink=sink)
    with pytest.raises(DraftInvalidError):
        backend.generate_many([task_factory("t1"), task_factory("t2")])
    assert list(sink.progress.completed()) == ["t1"]
    assert sink.rows == 1


def test_backend_without_sink_just_returns_drafts(
    fake_engine: object, provider_factory: object, task_factory: object, result_factory: object
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

    def __init__(self, by_model: dict[str, JsonValue]) -> None:
        """输入：{模型名: 结果或异常实例}。输出：DispatchEngine 实例。"""
        self.by_model = by_model
        self.models: list[str] = []

    def extract(self, request: object) -> object:
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


def test_build_backend_skips_a_provider_without_keys() -> None:
    """某家凭证撤下（环境变量不存在）就跳过它，链上不出现，并在能力备注里点名。"""
    config = BackendsConfig(
        providers=(provider_config("nim", "m1"), provider_config("spare", "m2"))
    )
    backend = build_backend(config, env={"JC_NIM_KEYS": "k1"}, sleep=lambda _s: None)
    assert backend.providers == ["nim"]
    assert any("spare" in note for note in backend.capabilities.notes)


def test_build_backend_reports_when_no_provider_has_keys() -> None:
    """一家都拿不到 key 才报人话，并点名是哪几家缺。"""
    config = BackendsConfig(providers=(provider_config("nim", "m1"),))
    with pytest.raises(ValueError, match="JC_NIM_KEYS"):
        build_backend(config, env={}, sleep=lambda _s: None)


def test_build_backend_never_echoes_the_key_in_errors() -> None:
    """报错信息里绝不许出现 key 明文。"""
    config = BackendsConfig(providers=(provider_config("nim", "m1"),))
    with pytest.raises(ValueError) as info:
        build_backend(config, env={}, sleep=lambda _s: None)
    assert "nvapi-" not in str(info.value)


def test_build_backend_probe_only_shapes_the_starting_mode(
    monkeypatch: pytest.MonkeyPatch, result_factory: object, task_factory: object
) -> None:
    """探测结果只决定从哪一档起步：探出 tools 就从 tools 起，且只用第一个 key 探。"""
    calls: list[dict[str, JsonValue]] = []

    def fake_probe(**kwargs: object) -> BackendCapabilities:
        calls.append(kwargs)
        return BackendCapabilities(tools=True, notes=("tools: 可用",))

    seen: list[dict[str, JsonValue]] = []
    real_provider = llm_module.Provider

    def spy_provider(config: object, keys: object, engine: object, **kwargs: object) -> object:
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
    result_factory: object, task_factory: object
) -> None:
    """不探测时从最严的 json_schema 起步，且不发任何探测请求。"""
    engine = DispatchEngine({"m1": result_factory(mode="json_schema")})
    config = BackendsConfig(providers=(provider_config("nim", "m1"),))
    backend = build_backend(config, engine=engine, env={"JC_NIM_KEYS": "k1"}, sleep=lambda _s: None)
    assert backend.generate(task_factory()).mode == "json_schema"
    assert len(engine.models) == 1


def test_build_backend_stops_at_the_first_provider_by_default(
    result_factory: object, task_factory: object
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


def test_build_backend_can_opt_into_mixing_models(result_factory: object, task_factory: object) -> None:
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


def test_build_backend_reads_a_real_toml_file(tmp_path: object) -> None:
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
