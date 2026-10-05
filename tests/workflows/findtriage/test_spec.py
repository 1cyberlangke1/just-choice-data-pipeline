"""改动引入性判定的真值规格测试：计划派生、任务 id 与序列化。"""

from __future__ import annotations


from workflows.findtriage import features, spec as spec_mod
from workflows.findtriage.features import RecordCase

CASE = RecordCase(cite="touched", evidence="elsewhere", fix="concrete", size=4, decoy=1)


def make(phrasing: int = 0, split: str = "train", seq: int = 3) -> spec_mod.FindTriageSpec:
    """输入：问法、切分、序号。输出：一条规格。"""
    return spec_mod.FindTriageSpec(case=CASE, phrasing=phrasing, split=split, seq=seq)


def test_plan_is_derived_not_stored() -> None:
    """输入：同一条规格取两次。输出：无。预期行为：计划与标签每次都从 case 重算、完全一致。"""
    one = make()
    assert one.plan == features.plan_for(CASE, 0)
    assert one.plan == make().plan
    assert one.labels == features.label_for(one.plan)
