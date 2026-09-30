"""改动引入性判定的真值规格测试：计划派生、任务 id 与序列化。"""

from __future__ import annotations

import json

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


def test_task_ids_are_unique_and_recomputable() -> None:
    """输入：两条不同规格。输出：无。预期行为：渲染与反解的 id 各自稳定、互不相同。"""
    one, two = make(seq=3), make(seq=4)
    assert one.render_task_id == "findtriage-train-0000003-render"
    assert one.trace_task_id == "findtriage-train-0000003-trace"
    assert one.render_task_id != two.render_task_id
    assert one.render_task_id != one.trace_task_id


def test_spec_to_dict_is_serialisable() -> None:
    """输入：一条规格。输出：无。预期行为：字段齐全、可直接 JSON 序列化。"""
    payload = spec_mod.spec_to_dict(make())
    assert payload["split"] == "train"
    assert payload["seq"] == 3
    assert payload["case"] == CASE.as_dict()
    json.dumps(payload, ensure_ascii=False)
