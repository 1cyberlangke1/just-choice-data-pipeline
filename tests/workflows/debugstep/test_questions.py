"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations


from workflows.debugstep import features, locales, questions
from workflows.debugstep.features import DebugStepCase

#: 一个普通 case：命中两条，会话来自 Python 调试器。
CASE = DebugStepCase("D1+D3", features.RUNTIME_PYTHON)


def test_flag_answers_are_fifty_fifty() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：每条判据的 True / False 各占一半。"""
    loc = locales.load("zh")
    totals = dict.fromkeys(features.TAG_IDS, 0)
    for case in features.build_cases(512):
        out = questions.build(features.plan_for(case, 0), loc, 0)
        for tag, item in zip(features.TAG_IDS, out, strict=False):
            if item.answer:
                totals[tag] += 1
    assert len(set(totals.values())) == 1, totals
    assert totals["D1"] == 24
