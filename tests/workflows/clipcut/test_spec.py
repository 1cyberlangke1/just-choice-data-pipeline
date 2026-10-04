"""真值规格：config 旋钮、四个答案与规则一致、请求号稳定、切分尺寸。"""

from __future__ import annotations

import random
from typing import Final

import pytest

from workflows.clipcut import features as ft
from workflows.clipcut import spec as sp
from workflows.clipcut.workflow import ClipCutWorkflow

#: 全空间 case（同分布侧）。
FULL: Final[tuple[ft.ClipCase, ...]] = tuple(ft.build_cases(512))


def test_default_config_is_valid() -> None:
    """输入：全默认参数。输出：无。预期行为：默认值都合法，校验不会误伤正常配置。"""
    config = sp.ClipConfig()
    assert config.split == "train"
    assert config.case_limit == 0
    assert config.phrasings == sp.PHRASINGS


def test_empty_phrasings_reports_the_field() -> None:
    """输入：空问法池。输出：无。预期行为：构造期 ValueError 且报错点名字段。"""
    with pytest.raises(ValueError, match="phrasings"):
        sp.ClipConfig(phrasings=())


def test_negative_case_limit_reports_the_field() -> None:
    """输入：负数 `case_limit`。输出：无。

    预期行为：报人话——以前负数会被 `cases[:case_limit]` 悄悄解释成「去掉尾部几条」。
    """
    with pytest.raises(ValueError, match="case_limit"):
        sp.ClipConfig(case_limit=-1)


def test_zero_cover_limit_reports_the_field() -> None:
    """输入：`cover_limit=0`。输出：无。预期行为：报人话，不拖到枚举器里抛裸异常。"""
    with pytest.raises(ValueError, match="cover_limit"):
        sp.ClipConfig(cover_limit=0)


def test_spec_answers_match_the_rules() -> None:
    """输入：全空间 case。输出：无。

    预期行为：四个答案与 `features` 那两条规则算出来的一致——这里是「生成器把答案算错了」
    那条路的第一道闸。
    """
    for case in FULL:
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        labels = ft.label_for(case)
        assert spec.keep_count == labels.count, case.describe()
        assert spec.blocked_any is bool(labels.blocked), case.describe()
        assert spec.dup_any is labels.dup, case.describe()
        want_first = "none" if labels.first is None else f"s{labels.first + 1}"
        assert spec.first_kept == want_first, case.describe()


def test_first_kept_names_the_earliest_kept_segment() -> None:
    """输入：两条手写的 case。输出：无。

    预期行为：最早保留的那一段用 `s<段号>` 表示；一段都没保留时是 `none`——
    这两个极端各用一条 case 钉住，免得 `first_kept` 只在中间几档上被测过。
    """
    kept_some = sp.sample_spec(random.Random(0), ft.ClipCase(topics=2, flags=1), "train", 0, 0)
    assert kept_some.first_kept == "s2"
    kept_none = sp.sample_spec(random.Random(0), ft.ClipCase(topics=0, flags=4), "train", 0, 0)
    assert kept_none.first_kept == "none"


def test_sample_spec_ignores_the_rng() -> None:
    """输入：同一 case、两个不同 RNG。输出：无。预期行为：规格逐字段一致（不掷骰子）。"""
    case = ft.ClipCase(topics=1, flags=3)
    first = sp.sample_spec(random.Random(0), case, "train", 7, 1)
    second = sp.sample_spec(random.Random(999), case, "train", 7, 1)
    assert first == second


def test_query_id_is_stable_and_distinct_per_case() -> None:
    """输入：全空间 case。输出：无。

    预期行为：请求号由素材指纹派生——同一 case 必得同一串（跨进程也稳），不同 case 之间
    不撞车（撞了「同输入同标签」这条一致性校验就认错人）。
    """
    ids = [sp.query_id_of(case) for case in FULL]
    assert len(set(ids)) == len(ids)
    assert sp.query_id_of(FULL[0]) == sp.query_id_of(FULL[0])
    assert all(value.startswith("K-") for value in ids)




def test_split_sizes_add_up_to_the_id_matrix() -> None:
    """输入：四个切分的配置。输出：无。

    预期行为：三个同分布切分都非空（切片按**位置**而不是桶号过滤，否则哈希方差会让 val
    只剩个位数），且加起来正好等于同分布侧的全枚举：4 主题序列 × 8 高亮序列。
    """
    workflow = ClipCutWorkflow()
    sizes = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in workflow.split_names()
    }
    assert min(sizes.values()) > 0
    assert sum(sizes[split] for split in ("train", "val", "test_iid")) == 4 * 8
    assert sizes["test_ood"] == len(ft.OOD_TOPIC_PATTERNS) * 8


def test_ood_holds_out_exactly_the_topic_pattern() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：同分布侧的主题序列 = 训练侧那几条，`test_ood` 侧**只有**留出那一条——
    这是「值留出」在数据上的形态，只写注释不写断言的话，改一次取值表就能把它变回同分布。
    """
    workflow = ClipCutWorkflow()
    id_topics = {
        str(workflow.stats_dimensions(case)["topics"])
        for split in ("train", "val", "test_iid")
        for case in workflow.build_cases(workflow.config_for(split))
    }
    ood_topics = {
        str(workflow.stats_dimensions(case)["topics"])
        for case in workflow.build_cases(workflow.config_for("test_ood"))
    }
    assert id_topics == {str(index) for index in ft.ID_TOPIC_PATTERNS}
    assert ood_topics == {str(index) for index in ft.OOD_TOPIC_PATTERNS}
    assert ood_topics - id_topics
