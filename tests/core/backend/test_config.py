"""后端配置测试：TOML 解析、默认值、缺项与非法取值的报错。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc.backend import config as cfg

GOOD = """
allow_mixed_providers = true

[[provider]]
name = "nim"
base_url = "https://integrate.api.nvidia.com/v1"
model = "openai/gpt-oss-20b"
keys_env = "JC_NIM_KEYS"

[[provider]]
name = "backup"
base_url = "https://example.com/v1"
model = "other/model"
mode = "tools"

[provider.limits]
requests_per_minute = 12
"""


def write(tmp_path: Path, text: str) -> Path:
    """输入：临时目录与 TOML 文本。输出：写好的文件路径。"""
    path = tmp_path / "backends.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_good_config(tmp_path: Path) -> None:
    loaded = cfg.load_config(write(tmp_path, GOOD))
    assert loaded.allow_mixed_providers is True
    assert [p.name for p in loaded.providers] == ["nim", "backup"]
    assert loaded.providers[0].mode == "auto"
    assert loaded.providers[1].mode == "tools"
    assert loaded.providers[1].limits.requests_per_minute == 12


def test_limits_defaults_match_plan(tmp_path: Path) -> None:
    """PLAN 3.1.6 定的初值：并发 4 / 单 key 1 / 40 RPM / 熔断 3 次 60 秒。"""
    limits = cfg.load_config(write(tmp_path, GOOD)).providers[0].limits
    assert limits.max_concurrency == 4
    assert limits.max_concurrency_per_key == 1
    assert limits.requests_per_minute == 40
    assert limits.fail_max == 3
    assert limits.cooldown_s == 60.0


def test_missing_file_reports_human_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="找不到后端配置文件"):
        cfg.load_config(tmp_path / "nope.toml")


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="未知的顶层键"):
        cfg.load_config(write(tmp_path, 'oops = 1\n[[provider]]\nname="a"\nbase_url="u"\nmodel="m"\n'))


def test_missing_provider_section_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="至少要有一个"):
        cfg.load_config(write(tmp_path, "allow_mixed_providers = true\n"))


def test_unknown_provider_key_rejected(tmp_path: Path) -> None:
    text = '[[provider]]\nname="a"\nbase_url="u"\nmodel="m"\nkeis_env="X"\n'
    with pytest.raises(ValueError, match="未知的键"):
        cfg.load_config(write(tmp_path, text))


def test_unknown_limits_key_rejected(tmp_path: Path) -> None:
    text = '[[provider]]\nname="a"\nbase_url="u"\nmodel="m"\n[provider.limits]\nrpm = 3\n'
    with pytest.raises(ValueError, match="未知的限额键"):
        cfg.load_config(write(tmp_path, text))


def test_duplicate_provider_names_rejected(tmp_path: Path) -> None:
    text = '[[provider]]\nname="a"\nbase_url="u"\nmodel="m"\n\n[[provider]]\nname="a"\nbase_url="u2"\nmodel="m2"\n'
    with pytest.raises(ValueError, match="名字重复"):
        cfg.load_config(write(tmp_path, text))


def test_bad_mode_rejected(tmp_path: Path) -> None:
    text = '[[provider]]\nname="a"\nbase_url="u"\nmodel="m"\nmode="yaml"\n'
    with pytest.raises(ValueError, match="mode 非法"):
        cfg.load_config(write(tmp_path, text))


def test_resolve_keys_splits_and_trims() -> None:
    provider = cfg.ProviderConfig(
        name="a", base_url="u", model="m", keys_env="JC_TEST_KEYS"
    )
    keys = cfg.resolve_keys(provider, {"JC_TEST_KEYS": " k1 ,, k2 ,"})
    assert keys == ("k1", "k2")


def test_resolve_keys_missing_env_names_the_variable_without_leaking() -> None:
    provider = cfg.ProviderConfig(name="a", base_url="u", model="m", keys_env="JC_ABSENT")
    with pytest.raises(ValueError) as exc:
        cfg.resolve_keys(provider, {})
    assert "JC_ABSENT" in str(exc.value)


def test_resolve_keys_requires_keys_env() -> None:
    provider = cfg.ProviderConfig(name="a", base_url="u", model="m")
    with pytest.raises(ValueError, match="没配 keys_env"):
        cfg.resolve_keys(provider, {"JC_X": "k"})


def test_resolve_all_keys_matches_provider_order(tmp_path: Path) -> None:
    text = GOOD.replace(
        'model = "other/model"', 'model = "other/model"\nkeys_env = "JC_BACKUP_KEYS"'
    )
    loaded = cfg.load_config(write(tmp_path, text))
    pairs = cfg.resolve_all_keys(loaded, {"JC_NIM_KEYS": "a,b", "JC_BACKUP_KEYS": "c"})
    assert pairs == (("a", "b"), ("c",))


def test_resolve_all_keys_fails_loudly_when_one_provider_is_unconfigured(tmp_path: Path) -> None:
    """第二个 provider 没配 keys_env 就该整份报错，不许悄悄跳过它。"""
    loaded = cfg.load_config(write(tmp_path, GOOD))
    with pytest.raises(ValueError, match="没配 keys_env"):
        cfg.resolve_all_keys(loaded, {"JC_NIM_KEYS": "a,b"})


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("max_concurrency", 0, "max_concurrency"),
        ("max_concurrency_per_key", 0, "max_concurrency_per_key"),
        ("requests_per_minute", -1, "requests_per_minute"),
        ("min_interval_ms", -1, "min_interval_ms"),
        ("cooldown_s", -1.0, "cooldown_s"),
        ("max_attempts", 0, "max_attempts"),
        ("fail_max", 0, "fail_max"),
    ],
)
def test_limits_reject_illegal_values(field: str, value: object, expected: str) -> None:
    """每个限额字段的非法取值都要在构造期报人话，不带着坏配置往下跑。"""
    with pytest.raises(ValueError, match=expected):
        cfg.Limits(**{field: value})


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"name": "  "}, "name 不能为空"),
        ({"base_url": " "}, "base_url 不能为空"),
        ({"model": ""}, "model 不能为空"),
        ({"timeout_s": 0}, "timeout_s 必须为正"),
        ({"max_retries": -1}, "max_retries 不能为负"),
    ],
)
def test_provider_config_rejects_illegal_fields(overrides: dict, expected: str) -> None:
    """名字 / 地址 / 模型 / 超时 / 重试次数的非法取值同样当场报错。"""
    base = {"name": "a", "base_url": "u", "model": "m"}
    with pytest.raises(ValueError, match=expected):
        cfg.ProviderConfig(**{**base, **overrides})


def test_backends_config_requires_at_least_one_provider() -> None:
    """空 provider 列表是配置错误，直接构造也不许放过。"""
    with pytest.raises(ValueError, match="至少要配置一个 provider"):
        cfg.BackendsConfig(providers=())


def test_provider_entry_must_be_a_table(tmp_path: Path) -> None:
    """`[[provider]]` 写成了别的形状要报人话，而不是抛一个 KeyError 出去。"""
    with pytest.raises(ValueError, match="不是一个表"):
        cfg.load_config(write(tmp_path, "provider = [1]\n"))


def test_allow_mixed_must_be_a_boolean(tmp_path: Path) -> None:
    """`allow_mixed_providers = "yes"` 这种写法必须被拒，不能按真值蒙混过去。"""
    text = 'allow_mixed_providers = "yes"\n[[provider]]\nname="a"\nbase_url="u"\nmodel="m"\n'
    with pytest.raises(ValueError, match="必须是布尔值"):
        cfg.load_config(write(tmp_path, text))
