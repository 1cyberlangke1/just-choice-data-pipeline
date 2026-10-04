"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations


from workflows.logtriage import features, locales, questions
from workflows.logtriage.features import LogTriageCase

#: 一个普通 case：命中两条，通道是应用日志。
CASE = LogTriageCase("N1+N3", features.SOURCE_APP)


def test_route_answers_follow_the_thresholds() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：`route` 的答案是阈值算出来的那一档。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "route")
        assert item.answer == features.handling_level(len(plan.hits))
        assert item.question["criteria"][item.answer] == questions.route_text(loc, len(plan.hits))
