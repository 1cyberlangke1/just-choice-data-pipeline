"""真值规格：config 旋钮、四个答案与规则一致、请求号稳定、切分尺寸。"""

from __future__ import annotations

import random
from typing import Final

import pytest

from workflows.quoteread import features as ft
from workflows.quoteread import spec as sp
from workflows.quoteread.workflow import QuoteReadWorkflow

#: 全空间 case（同分布侧）。
FULL: Final[tuple[ft.QuoteCase, ...]] = tuple(ft.build_cases(512))


def test_default_config_is_valid() -> None:
    """输入：全默认参数。输出：无。预期行为：默认值都合法，校验不会误伤正常配置。"""
    config = sp.QuoteConfig()
    assert config.split == "train"
    assert config.case_limit == 0
    assert config.phrasings == sp.PHRASINGS


def test_empty_phrasings_reports_the_field() -> None:
    """输入：空问法池。输出：无。预期行为：构造期 ValueError 且报错点名字段。"""
    with pytest.raises(ValueError, match="phrasings"):
        sp.QuoteConfig(phrasings=())


def test_negative_case_limit_reports_the_field() -> None:
    """输入：负数 `case_limit`。输出：无。

    预期行为：报人话——以前负数会被 `cases[:case_limit]` 悄悄解释成「去掉尾部几条」。
    """
    with pytest.raises(ValueError, match="case_limit"):
        sp.QuoteConfig(case_limit=-1)


def test_zero_cover_limit_reports_the_field() -> None:
    """输入：`cover_limit=0`。输出：无。预期行为：报人话，不拖到枚举器里抛裸异常。"""
    with pytest.raises(ValueError, match="cover_limit"):
        sp.QuoteConfig(cover_limit=0)


def test_spec_answers_match_the_rules() -> None:
    """输入：全空间 case。输出：无。

    预期行为：四个答案与 `features` 那四条规则算出来的一致——这里是「生成器把答案算错了」
    那条路的第一道闸。
    """
    for case in FULL:
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        labels = ft.label_for(case)
        assert spec.tier == labels.tier, case.describe()
        assert spec.leader == case.name_ids[labels.leader], case.describe()
        assert spec.alert is labels.alert, case.describe()
        assert spec.divergent is labels.divergent, case.describe()


def test_query_id_is_stable_and_case_derived() -> None:
    """输入：同一 case 两次、不同 case 一次。输出：无。

    预期行为：请求号由素材指纹派生——同 case 必得同号（跨切分、跨语言都认得出是同一份素材），
    不同 case 不该撞号。
    """
    first = sp.sample_spec(random.Random(0), ft.QuoteCase(0, 0, 0), "train", 0, 0)
    again = sp.sample_spec(random.Random(9), ft.QuoteCase(0, 0, 0), "val", 7, 2)
    other = sp.sample_spec(random.Random(0), ft.QuoteCase(1, 0, 0), "train", 0, 0)
    assert first.query_id == again.query_id
    assert first.query_id != other.query_id
    assert first.query_id.startswith("Q-")




def test_split_sizes_follow_the_buckets() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：三档加起来正好是全空间、且每档都非空——切分按 case 哈希分桶，同一条不会
    落进两档（`config_for` 忘了替换 `split` 时这里会红）。
    """
    workflow = QuoteReadWorkflow()
    sizes = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in ("train", "val", "test_iid")
    }
    assert all(count > 0 for count in sizes.values()), sizes
    assert sum(sizes.values()) == len(FULL)


def test_ood_keeps_the_whole_value_and_unit_space() -> None:
    """输入：`test_ood` 的配置。输出：无。

    预期行为：留出的只有标的名集那一维，读数与单位两个维度在留出侧**全开**——把某一维
    一起冻掉会让 `tier` 的档位在留出侧取不全。
    """
    workflow = QuoteReadWorkflow()
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert {case.names for case in cases} == set(ft.OOD_DIMS["names"])
    assert {case.values for case in cases} == set(ft.DIMS["values"])
    assert {case.units for case in cases} == set(ft.DIMS["units"])
