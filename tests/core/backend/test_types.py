"""后端类型层测试：档位降级链、错误分类、任务派生 schema。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from jc.backend import types


class Answer(BaseModel):
    """测试用的最小草稿模型。"""

    answer: str


def test_modes_from_auto_starts_at_strictest() -> None:
    assert types.modes_from(types.MODE_AUTO) == (
        types.MODE_JSON_SCHEMA,
        types.MODE_TOOLS,
        types.MODE_MD_JSON,
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("json_schema", ("json_schema", "tools", "md_json")),
        ("tools", ("tools", "md_json")),
        ("md_json", ("md_json",)),
    ],
)
def test_modes_from_degrades_downward(mode: str, expected: tuple[str, ...]) -> None:
    assert types.modes_from(mode) == expected


def test_modes_from_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="未知的结构化输出档位"):
        types.modes_from("yaml")


@pytest.mark.parametrize("error_type", [types.TransportError, types.RateLimitError])
def test_transport_failures_count_as_faults(error_type: type) -> None:
    assert error_type.counts_as_fault is True


@pytest.mark.parametrize(
    "error_type",
    [
        types.AuthError,
        types.SchemaUnsupportedError,
        types.DraftInvalidError,
        types.ProviderUnavailable,
        types.AllProvidersDown,
    ],
)
def test_non_provider_failures_do_not_count(error_type: type) -> None:
    """「模型答错」「换 key 就好」这类都不能算 provider 故障，否则会误熔断。"""
    assert error_type.counts_as_fault is False


def test_rate_limit_error_carries_retry_after() -> None:
    exc = types.RateLimitError("慢点", retry_after_s=12.5)
    assert exc.retry_after_s == 12.5
    assert types.RateLimitError("慢点").retry_after_s is None


def test_task_schema_is_derived_from_response_model() -> None:
    task = types.GenerationTask(
        task_id="t-1", kind="choice", domain="demo", prompt="p", response_model=Answer
    )
    schema = task.schema
    assert schema["type"] == "object"
    assert "answer" in schema["properties"]


@pytest.mark.parametrize(
    ("caps", "expected"),
    [
        (types.BackendCapabilities(json_schema=True), types.MODE_JSON_SCHEMA),
        (types.BackendCapabilities(tools=True), types.MODE_TOOLS),
        (types.BackendCapabilities(json_object=True), types.MODE_MD_JSON),
        (types.BackendCapabilities(), types.MODE_MD_JSON),
    ],
)
def test_capabilities_pick_starting_mode(caps: types.BackendCapabilities, expected: str) -> None:
    assert caps.starting_mode() == expected
