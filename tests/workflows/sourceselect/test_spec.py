"""来源选择的真值规格测试：计划派生、任务 id 与序列化。"""

from __future__ import annotations

from workflows.sourceselect import spec as spec_mod
from workflows.sourceselect.features import SourceCase

CASE = SourceCase(need="coverage", size=4, shape="unique", near_miss=1)


def make(phrasing: int = 0, split: str = "train", seq: int = 3) -> spec_mod.SourceSpec:
    """输入：问法、切分、序号。输出：一条规格。"""
    return spec_mod.SourceSpec(case=CASE, phrasing=phrasing, split=split, seq=seq)
