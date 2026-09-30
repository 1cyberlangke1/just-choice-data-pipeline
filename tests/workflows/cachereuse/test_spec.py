"""真值规格的测试：档位映射、case 空间、两条独立路径、答案分布。"""

from __future__ import annotations

from collections import Counter
from typing import Final

import pytest

from workflows.cachereuse import spec as sp

#: 全空间的 case（默认配置 = 全部五个维）。
FULL: Final[tuple[sp.ReuseCase, ...]] = tuple(sp.build_cases(sp.ReuseConfig(split="train")))


def test_case_space_is_not_empty() -> None:
    """输入：无。输出：无。预期行为：全空间够大，否则下面的分布尺子量不到东西。"""
    assert len(FULL) >= 100, f"全空间只有 {len(FULL)} 个 case"


@pytest.mark.parametrize("band", sp.BANDS)
@pytest.mark.parametrize("length", sp.LENGTHS)
def test_band_prefix_covers_the_step_range(band: str, length: int) -> None:
    """输入：档名、步数。输出：无。

    预期行为：四档的可复用步数落在 0~length 里，且**四档互不相同**（`half` 用上取整，
    所以 length ≥ 4 时它既不等于 `few` 也不等于 `most`）。
    """
    prefix = sp.band_prefix(band, length)
    assert 0 <= prefix <= length
    others = {sp.band_prefix(name, length) for name in sp.BANDS}
    assert len(others) == len(sp.BANDS), f"length={length} 时四档的可复用步数有重合：{others}"


def test_unknown_band_is_rejected() -> None:
    """输入：表外的档名。输出：无。预期行为：报人话而不是悄悄给 0。"""
    with pytest.raises(ValueError, match="未知档位"):
        sp.band_prefix("almost", 5)


@pytest.mark.parametrize("band", sp.BANDS)
def test_feasible_keeps_the_tail_inside_the_trace(band: str) -> None:
    """输入：一个档。输出：无。

    预期行为：三种步数下，分叉之后都装得下这一档的 `tail`——装不下说明档位表与步数范围脱节，
    `build_cases` 会静默少收一批 case（切分尺寸测试会红，但那时已经很难查了）。
    """
    for length in sp.LENGTHS:
        case = sp.ReuseCase(length, band, sp.DIFF_KINDS[0], sp.VERBATIM, sp.FLAVORS[0])
        assert sp.feasible(case), case.describe()
        assert case.prefix + 1 + case.tail <= length, case.describe()


def test_every_band_has_a_divergence() -> None:
    """输入：无。输出：无。

    预期行为：**每一档都真的有分叉**——没有分叉的档会让「分叉原因」那一族出不了题，
    也会把布尔题族的分布拽偏（模块文档里记着那次 0.111 / 0.556）。
    """
    for case in FULL:
        assert case.first_diff <= case.length, case.describe()
        assert case.prefix < case.length, case.describe()


def test_material_ignores_the_text_only_dims() -> None:
    """输入：无。输出：无。

    预期行为：同一条请求的几种写法 / 来源**共用同一个素材指纹**——切分按素材分桶，
    指纹里带上它们就会把同一条请求拆到两个切分里去。
    """
    base = sp.ReuseCase(5, "half", "action", sp.VERBATIM, "web")
    for spelling in sp.SPELLINGS:
        for flavor in sp.FLAVORS:
            other = sp.ReuseCase(5, "half", "action", spelling, flavor)
            assert other.material() == base.material()


def test_traces_are_deterministic() -> None:
    """输入：无。输出：无。预期行为：同一 case 在任何进程里都得到同一对轨迹。"""
    for case in FULL[:20]:
        assert sp.traces_of(case) == sp.traces_of(case)
        assert sp.query_id_of(case) == sp.query_id_of(case)


def test_both_paths_agree_on_every_case() -> None:
    """输入：全空间 case。输出：无。

    预期行为：构造侧（`ReuseCase` 的属性）与审计侧（`observed_labels` 从两条轨迹重算）
    对四个答案**逐条一致**——两条独立路径是这一族的判据骨架，一条歪了整族都不可信。
    """
    for case in FULL:
        old, new = sp.traces_of(case)
        seen = sp.observed_labels(old, new)
        assert seen["band"] == case.band, case.describe()
        assert seen["diverge"] == case.diverge_band(), case.describe()
        assert seen["diff_kind"] == case.diff_kind, case.describe()
        assert seen["reuse_half"] == case.reuse_half, case.describe()
        assert seen["prefix"] == case.prefix, case.describe()
        assert seen["equivalent_total"] == case.equivalent_total, case.describe()


def test_observed_labels_rejects_mismatched_lengths() -> None:
    """输入：两条步数不同的轨迹。输出：无。预期行为：当场报错，不静默对齐到较短那条。"""
    with pytest.raises(ValueError, match="步数不同"):
        sp.observed_labels((("open", "search_box"),), ())


def test_answer_distributions_are_not_degenerate() -> None:
    """输入：全空间 case。输出：无。

    预期行为：**先验分布再定档位**（范式 §9.70 一）——四个题族的答案都不许出现「某一档吃掉
    一半以上」，也不许有档位一条都取不到。这一族最初多留了一档「整段可复用」，它没有分叉，
    把布尔题族拽到 0.111 / 0.556，当场被删掉。
    """
    total = len(FULL)
    families = {
        "reuse_band": Counter(case.band for case in FULL),
        "diverge_at": Counter(case.diverge_band() for case in FULL),
        "diff_kind": Counter(case.diff_kind for case in FULL),
        "reuse_half": Counter(case.reuse_half for case in FULL),
    }
    levels = {
        "reuse_band": sp.BANDS,
        "diverge_at": ("first", "second", "middle", "last"),
        "diff_kind": sp.DIFF_KINDS,
        "reuse_half": (True, False),
    }
    for family, counts in families.items():
        assert set(counts) == set(levels[family]), (family, counts)
        shares = [counts[level] / total for level in levels[family]]
        assert max(shares) <= 0.5, (family, shares)
        assert min(shares) >= 0.1, (family, shares)
