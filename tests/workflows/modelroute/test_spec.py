"""真值规格：六条规则函数、档位表自洽、case 矩阵、值留出与指纹。"""

from __future__ import annotations

import pytest

from workflows.modelroute import spec as sp
from workflows.modelroute.spec import RouteCase, RouteConfig

# ---------- 选档与升级链 ----------


def test_tier_follows_the_power_need() -> None:
    """输入：四个任务。输出：无。预期行为：首选档就是能力需求对应的那个档。"""
    assert [sp.tier_of(task) for task in sp.TASKS] == ["nano", "mini", "core", "max"]


def test_tier_rejects_an_unknown_task() -> None:
    """输入：不存在的任务名。输出：无。预期行为：报人话并列出可用任务，不静默给默认档。"""
    with pytest.raises(KeyError, match="没有任务类型"):
        sp.tier_of("nope")


def test_power_need_matches_the_profile_power() -> None:
    """输入：四张表。输出：无。

    预期行为：`POWER_NEED` 的四个取值恰好是档位表里的四个能力分——`TIER_FOR_POWER`
    是按这条成立的，两边一旦对不上，「首选档」就会静默落到别的档上。
    """
    assert sorted(sp.POWER_NEED.values()) == sorted(
        profile["power"] for profile in sp.PROFILE.values()
    )
    assert set(sp.TIER_FOR_POWER) == set(sp.POWER_NEED.values())


def test_escalation_chain_stops_at_the_top() -> None:
    """输入：四个任务。输出：无。

    预期行为：只有首选档之上还有档时才有链——`reason` 的首选档已经是 max，
    所以 `has_next` 为假、`next_tier_of` 给 None（`next` 那一族的两阶段依赖）。
    """
    assert [sp.next_tier_of(task) for task in sp.TASKS] == ["mini", "core", "max", None]
    assert [sp.has_next(task) for task in sp.TASKS] == [True, True, True, False]


# ---------- 推理深度挪档 ----------


def test_effort_shift_table_is_three_equal_thirds() -> None:
    """输入：复杂度三档 × 延迟三档。输出：无。

    预期行为：三个深度档各出现 **3** 次——这是 `effort` 那一族答案均衡的唯一来源
    （挪档表恰好把九格三等分）。改了 `SHIFT` 或 `STEPS` 这条会先红。
    """
    table = [sp.effort_of(complexity, latency) for complexity in sp.COMPLEXITIES for latency in sp.LATENCIES]
    assert table.count("low") == 3
    assert table.count("mid") == 3
    assert table.count("high") == 3


@pytest.mark.parametrize(
    ("complexity", "latency", "expected"),
    [
        ("low", "relaxed", "mid"),
        ("low", "normal", "low"),
        ("low", "strict", "low"),  # 触底：不再往下挪
        ("mid", "strict", "low"),
        ("high", "relaxed", "high"),  # 到顶：不再往上挪
        ("high", "strict", "mid"),
    ],
)
def test_effort_shifts_and_clamps(complexity: str, latency: str, expected: str) -> None:
    """输入：复杂度、延迟、期望。输出：无。预期行为：按 `SHIFT` 挪一档并在端点钳位。"""
    assert sp.effort_of(complexity, latency) == expected


# ---------- 三个判断 ----------


def test_detection_rules_are_value_lookups() -> None:
    """输入：两个取值。输出：无。预期行为：首次请求为真、成本敏感允许降档、公开可缓存。"""
    assert sp.is_first_of("first") is True and sp.is_first_of("retry") is False
    assert sp.degrade_of("tight") is True and sp.degrade_of("normal") is False
    assert sp.cache_of("public") is True and sp.cache_of("private") is False


# ---------- case 与素材 ----------


def test_material_fingerprint_covers_every_dimension() -> None:
    """输入：七个只差一维的 case。输出：无。

    预期行为：指纹逐维不同——指纹是切分分桶的键，漏一维会让「只有该维不同」的 case
    被当成同一份素材，切分之间立刻撞车。
    """
    base = RouteCase("classify", "low", "relaxed", "normal", "first", "public", 0)
    variants = [
        RouteCase("extract", "low", "relaxed", "normal", "first", "public", 0),
        RouteCase("classify", "mid", "relaxed", "normal", "first", "public", 0),
        RouteCase("classify", "low", "normal", "normal", "first", "public", 0),
        RouteCase("classify", "low", "relaxed", "tight", "first", "public", 0),
        RouteCase("classify", "low", "relaxed", "normal", "retry", "public", 0),
        RouteCase("classify", "low", "relaxed", "normal", "first", "private", 0),
        RouteCase("classify", "low", "relaxed", "normal", "first", "public", 1),
    ]
    prints = {sp.material_of(base)} | {sp.material_of(case) for case in variants}
    assert len(prints) == 8


def test_unseen_dims_names_the_task_and_the_phrasing() -> None:
    """输入：各含一个留出取值的 case。输出：无。

    预期行为：只留出 `task` 与 `phrasing` 两处——另外五个维度留出一个取值都会让某一族在
    训练侧退化成常量或把分布压歪（`cost` / `data` / `retry` 各自是某个 noul 族的唯一判据，
    `complexity` / `latency` 一起决定 `effort`）。
    """
    assert RouteCase("reason", "low", "relaxed", "normal", "first", "public", 0).unseen_dims() == (
        "task",
    )
    assert RouteCase("codefix", "low", "relaxed", "normal", "first", "public", 1).unseen_dims() == (
        "phrasing",
    )
    assert RouteCase("reason", "low", "relaxed", "normal", "first", "public", 1).unseen_dims() == (
        "task",
        "phrasing",
    )
    assert RouteCase("codefix", "low", "relaxed", "normal", "first", "public", 0).unseen_dims() == ()


def test_case_describe_is_readable() -> None:
    """输入：一个 case。输出：无。预期行为：描述里七个维度都在。"""
    text = RouteCase("extract", "mid", "normal", "tight", "retry", "private", 1).describe()
    assert text == "extract/mid/normal/tight/retry/private/1"


def test_case_as_dict_flattens_dimensions() -> None:
    """输入：一个 case。输出：无。预期行为：七个维度都在，值是原始类型（phrasing 是整数）。"""
    data = RouteCase("reason", "high", "strict", "normal", "first", "public", 1).as_dict()
    assert data == {
        "task": "reason",
        "complexity": "high",
        "latency": "strict",
        "cost": "normal",
        "retry": "first",
        "data": "public",
        "phrasing": 1,
    }


# ---------- 采样参数 ----------


def test_build_cases_is_the_full_product() -> None:
    """输入：全开的采样参数。输出：无。预期行为：七张表的全组合，顺序稳定。"""
    cases = sp.build_cases(RouteConfig())
    expected = (
        len(sp.TASKS)
        * len(sp.COMPLEXITIES)
        * len(sp.LATENCIES)
        * len(sp.COSTS)
        * len(sp.RETRIES)
        * len(sp.DATAS)
        * len(sp.PHRASINGS)
    )
    assert len(cases) == expected == 576
    assert cases[0] == RouteCase("classify", "low", "relaxed", "normal", "first", "public", 0)
    assert cases == sp.build_cases(RouteConfig())


def test_build_cases_ood_only_keeps_cases_with_an_unseen_value() -> None:
    """输入：`ood_only=True`。输出：无。

    预期行为：留下的每条都至少含一个留出取值，且条数 = 全量 − 同分布量
    （同分布量就是「训练侧见过的取值」的全组合）。
    """
    ood = sp.build_cases(RouteConfig(ood_only=True))
    assert ood and all(case.unseen_dims() for case in ood)
    id_count = (
        len(sp.SEEN_TASKS)
        * len(sp.COMPLEXITIES)
        * len(sp.LATENCIES)
        * len(sp.COSTS)
        * len(sp.RETRIES)
        * len(sp.DATAS)
        * len(sp.SEEN_PHRASINGS)
    )
    assert id_count == 216
    assert len(ood) == len(sp.build_cases(RouteConfig())) - id_count == 360


@pytest.mark.parametrize(
    "field",
    ["tasks", "complexities", "latencies", "costs", "retries", "datas", "phrasings"],
)
def test_empty_pool_reports_the_field(field: str) -> None:
    """输入：逐张清空的取值表。输出：无。预期行为：报人话并点名是哪个字段。"""
    with pytest.raises(ValueError, match=field):
        RouteConfig(**{field: ()})


# ---------- 真值规格 ----------


def test_spec_computes_every_answer_from_the_case() -> None:
    """输入：一个 case。输出：无。预期行为：六个答案都由规则函数算出来、与 case 一致。"""
    spec = sp.sample_spec(RouteCase("extract", "mid", "relaxed", "normal", "retry", "public", 1), "val", 4, 1)
    assert spec.split == "val" and spec.seq == 4 and spec.phrasing == 1
    assert spec.tier == "mini"
    assert spec.next_tier == "core"
    assert spec.effort == "high"
    assert spec.is_first is False
    assert spec.degrade is False
    assert spec.cache is True


def test_request_id_is_stable_and_split_independent() -> None:
    """输入：同一 case 的两次规格、不同切分与序号。输出：无。

    预期行为：请求号只由素材指纹决定——带序号或切分名会让同一条请求在不同切分里拿到
    不同请求号，等于把切分信息写进 state。
    """
    case = RouteCase("codefix", "low", "strict", "tight", "retry", "private", 0)
    first = sp.sample_spec(case, "train", 0, 0)
    second = sp.sample_spec(case, "test_ood", 7, 1)
    assert first.request_id == second.request_id
    assert first.request_id.startswith("R-") and len(first.request_id) == 7
    assert first.tier == second.tier


def test_spec_to_dict_is_json_ready() -> None:
    """输入：一份规格。输出：无。预期行为：摘要里带上 case、留出维度与六个答案。"""
    data = sp.spec_to_dict(sp.sample_spec(RouteCase("reason", "high", "relaxed", "normal", "first", "public", 0), "test_ood", 1, 0))
    assert data["case"] == "reason/high/relaxed/normal/first/public/0"
    assert data["unseen_dims"] == ["task"]
    assert data["next_tier"] is None
    assert data["effort"] == "high"
    assert set(data) == {
        "case",
        "unseen_dims",
        "split",
        "seq",
        "request_id",
        "tier",
        "next_tier",
        "effort",
        "is_first",
        "degrade",
        "cache",
    }
