"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations


from workflows.memorygate import features, locales, questions
from workflows.memorygate.features import MemoryGateCase

#: 一个普通 case。
CASE = MemoryGateCase("M1+M3", features.FORM_SENTENCE)


def test_gate_answers_follow_the_thresholds() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：`gate` 的答案是阈值算出来的那一档。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "gate")
        assert item.answer == features.handling_level(len(plan.hits))
        assert item.question["criteria"][item.answer] == questions.gate_text(loc, len(plan.hits))
