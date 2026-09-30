"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations


from workflows.turntake import features, locales, questions
from workflows.turntake.features import TurnTakeCase

#: 一个普通 case：命中两条，通道是聊天消息。
CASE = TurnTakeCase("T1+T3", features.CHANNEL_IM)


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
    assert totals["T1"] == 24
