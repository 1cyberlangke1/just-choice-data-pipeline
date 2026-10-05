"""特征层测试：判据只认字面、命中组合按构造均衡、两条标签路径互不打架。"""

from __future__ import annotations

import pytest

from tests.workflows.debugstep.conftest import session_of
from workflows.debugstep import features


def test_hit_set_is_purely_literal() -> None:
    """输入：一屏会话里的一行。输出：无。预期行为：命中组合按 `TAG_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ("Stopped: uncaught TypeError raised while loading config (app/config.py:17)", ("D1",)),
        ("Stopped: Fatal error while opening the socket (net.go:88, in dial)", ("D1",)),
        ("Stopped: panic: runtime error (main.go:12, in main)", ("D1",)),
        ("Stopped: traceback on the worker thread (jobs.py:5, in run)", ("D1",)),
        ("Locals: user=None, retries=3", ("D2",)),
        ("Locals: payload=null, size=0", ("D2",)),
        ("Locals: head=nil, tail=node", ("D2",)),
        ("Locals: node=undefined, depth=2", ("D2",)),
        ("Stack: load <- run <- main", ("D3",)),
        ("Breakpoints: app.py:42, app.py:42", ("D4",)),
        ("Breakpoints: src/net.go:88, src/net.go:88", ("D4",)),
        # 这几条是判据「只认字面、各占一行、看词边界」的证据：正常的一屏一条都不该踩。
        ("Stopped: pause requested at a user breakpoint (app/main.py:42, in run)", ()),
        ("Stopped: reached the end of the loop (app/main.py:9, in run)", ()),
        ("Stack: run (app/main.py:42)", ()),
        ("Locals: count=3, name=alpha, items=[1, 2]", ()),
        ("Locals: limit=undefinedness, name=alpha", ()),
        ("Breakpoints: app/main.py:42, app/config.py:17", ()),
        ("Note: 手动暂停，先看看当前这一帧里的取值喵", ()),
    ]:
        try:
            assert features.hit_set(text) == expect
            assert features.hits_of(text) == "+".join(expect)
            assert features.parse_hits(features.hits_of(text)) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_key_lines_are_the_five_structure_keys() -> None:
    """输入：无。输出：无。

    预期行为：五个结构键是这一族 state 格式的一部分（键名用英文标识符），`missing_keys` 认的是
    「单独占一行、后面跟冒号」——少一行就定位不到钉在行上的那几条判据。
    """
    assert features.KEY_LINES == ("stopped", "stack", "locals", "breakpoints", "note")
    session = session_of(())
    assert features.missing_keys(session) == ()
    assert features.missing_keys(session.replace("Note:", "备忘：")) == ("note",)


def test_runtimes_are_three_distinct_shapes() -> None:
    """输入：无。输出：无。

    预期行为：三个运行时键两两不同，且**全空间里每种运行时都被枚举到**——留出维只砍取值，
    不该把别的取值一起砍掉（`build_cases` 默认维度表覆盖 `RUNTIMES`）。
    """
    assert features.RUNTIMES == ("python", "node", "go")
    assert len(set(features.RUNTIMES)) == len(features.RUNTIMES)
    assert features.DIMS["runtime"] == features.RUNTIMES
    cases = features.build_cases(512)
    assert {case.runtime for case in cases} == set(features.RUNTIMES)


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且与真值只差一两处
    （差得越少越难分辨，题目才有区分度）。
    """
    case = features.DebugStepCase("D1+D3", features.RUNTIME_PYTHON)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.hits == ("D1", "D3")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.hits for combo in plan.distractors)
    assert all(len(set(combo) ^ set(plan.hits)) <= 2 for combo in plan.distractors)
    assert plan.distractors == tuple(
        sorted(plan.distractors, key=lambda combo: (len(combo), combo))
    )


@pytest.mark.parametrize(
    ("count", "expect"),
    [(0, "continue"), (1, "inspect"), (2, "step_over"), (3, "stop"), (4, "stop")],
)
def test_act_level_thresholds(count: int, expect: str) -> None:
    """输入：命中条数。输出：无。预期行为：0/1/2/3 条以上 分别对应四个档位。"""
    assert features.act_level(count) == expect


def test_two_label_paths_agree_on_every_faithful_session() -> None:
    """输入：全空间的 48 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——
    会话兑现了计划，两条路就不该打架。
    """
    cases = features.build_cases(512)
    assert len(cases) == 48  # 16 组命中 × 3 种运行时
    for case in cases:
        session = session_of(case.tags())
        assert features.hits_of(session) == case.hits, case.describe()
        assert features.observed_labels(session).hits == features.label_for(
            features.plan_for(case, 0)
        ).hits


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.DebugStepCase("", features.RUNTIME_PYTHON)
    assert case.describe() == "无命中-python"
    assert case.fingerprint() == "hits=|runtime=python"
    assert case.tags() == ()
    assert features.parse_hits("") == ()
