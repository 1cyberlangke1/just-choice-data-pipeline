"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations

import pytest

from workflows.tracescore import features, locales, questions
from workflows.tracescore.features import TraceScoreCase

#: 一个普通 case：命中两条，入口是命令行代理。
CASE = TraceScoreCase("S1+S3", features.SOURCE_CLI)


def test_route_answers_follow_the_score_thresholds() -> None:
    """输入：全空间 48 组 case。输出：无。

    预期行为：`route` 的答案由**推进分**算出来（不是命中条数）——`S1+S3` 是 5 分走继续推进，
    `S2+S4` 是 1 分走停下问人，两条命中数一样。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "route")
        assert item.answer == features.route_for(plan.score)
        assert item.question["criteria"][item.answer] == questions.route_text(loc, plan.score)
    assert questions.route_text(loc, 5) != questions.route_text(loc, 1)
