"""采样参数与真值规格：旋钮校验、请求号稳定、标签挂在规格上。"""

from __future__ import annotations

import random

import pytest

from tests.workflows.sentiment.conftest import _make_case
from workflows.sentiment import features as ft
from workflows.sentiment import spec as sp


def test_defaults_still_construct() -> None:
    """输入：全默认参数。输出：无。预期行为：默认值都合法，校验不会误伤正常配置。"""
    config = sp.SentimentConfig()
    assert config.split == "train"
    assert config.cover_limit == sp.COVER_LIMIT
    assert config.case_limit == 0
    assert config.phrasings == sp.PHRASINGS


def test_empty_phrasings_reports_the_field() -> None:
    """输入：问法池清空。输出：无。预期行为：构造期 ValueError，且报错点名字段。"""
    with pytest.raises(ValueError, match="phrasings"):
        sp.SentimentConfig(phrasings=())


@pytest.mark.parametrize(("field", "value"), [("cover_limit", 0), ("case_limit", -1)])
def test_bad_knobs_report_the_field(field: str, value: int) -> None:
    """输入：非法的上限旋钮。输出：无。预期行为：构造期报人话，不拖到采样阶段。"""
    with pytest.raises(ValueError, match=field):
        sp.SentimentConfig(**{field: value})


def test_cover_limit_bigger_than_the_space_still_enumerates() -> None:
    """输入：上限远大于全空间。输出：无。预期行为：全枚举，不重复、不越界。"""
    cases = ft.build_cases(sp.COVER_LIMIT, dims={"arc": ft.ID_ARCS, "topic": ft.ID_TOPICS})
    assert len(cases) == len(ft.ID_ARCS) * len(ft.ID_TOPICS)
    assert len(set(cases)) == len(cases)


def test_query_id_is_stable_and_distinguishes_cases() -> None:
    """输入：同一个 case 造两次、不同 case 各一个。输出：无。

    预期行为：请求号由素材指纹派生——同 case 同号（跨切分跨语言都是同一个），不同 case
    不同号（否则「同输入同标签」这条一致性校验认不出人）。
    """
    first = sp.query_id_of(_make_case(arc=4, topic="review"))
    assert first == sp.query_id_of(_make_case(arc=4, topic="review"))
    assert first.startswith("E-") and len(first) == 8
    others = {
        sp.query_id_of(ft.SentimentCase(arc=arc, topic=topic))
        for arc in ft.ID_ARCS
        for topic in ft.ID_TOPICS
    }
    assert len(others) == len(ft.ID_ARCS) * len(ft.ID_TOPICS)


def test_spec_exposes_the_four_labels() -> None:
    """输入：一条规格。输出：无。预期行为：四个答案属性就是 `label_for` 算出来的那一份。"""
    spec = sp.sample_spec(random.Random(0), _make_case(arc=3, topic="standup"), "train", 0, 0)
    labels = ft.label_for(spec.case)
    assert spec.labels == labels
    assert spec.tone == labels.tone
    assert spec.pivot_key == labels.pivot_key
    assert spec.drop == labels.drop
    assert spec.rebound == labels.rebound


def test_sample_spec_is_pure() -> None:
    """输入：同 case 用两个不同种子的 RNG。输出：无。预期行为：规格一模一样，RNG 不参与。"""
    case = _make_case(arc=6, topic="neighbors")
    assert sp.sample_spec(random.Random(1), case, "val", 7, 2) == sp.sample_spec(
        random.Random(999), case, "val", 7, 2
    )


