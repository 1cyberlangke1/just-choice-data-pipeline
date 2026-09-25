"""通用层测试：判别器聚合（`jc.judge`）。

这里只验「异常怎么归类」这一件事：普通异常收成「不通过」，而标了 `fatal_for_run`
的异常必须冒泡——否则「全部 provider 都不可用」会被写成一条 `E_JUDGE_*` 拒绝，
整批照跑，最后产出一个全被拒的假数据集（实测就是这么翻车的）。
"""

from __future__ import annotations

from typing import Any

import pytest

from jc import judge as judge_mod


class Boom(Exception):
    """普通异常：判别器自己崩了。"""


class Fatal(Exception):
    """致命异常：像 `AllProvidersDown` 那样声明「整批该停下」。"""

    fatal_for_run = True


class Judge:
    """假判别器：按 `exc` 决定是给结论还是抛异常。"""

    def __init__(self, name: str, *, ok: bool = True, exc: BaseException | None = None) -> None:
        self.name = name
        self._ok = ok
        self._exc = exc

    def judge(self, sample: dict[str, Any], spec: Any = None) -> judge_mod.Verdict:
        """输入：样本与真值。输出：Verdict 或抛异常。"""
        if self._exc is not None:
            raise self._exc
        return judge_mod.Verdict(self._ok, reason="ok" if self._ok else "不行")


def test_ordinary_judge_crash_becomes_a_failed_verdict() -> None:
    """输入：一个抛普通异常的判别器。

    输出：[(名字, 不通过的 Verdict)]。
    预期行为：判别器崩了不能把整批带走，要如实记成「这条没过」并留下原因。
    """
    out = judge_mod.run_judges({}, None, (Judge("boom", exc=Boom("炸了")),))
    assert [name for name, _ in out] == ["boom"]
    assert out[0][1].ok is False
    assert "炸了" in out[0][1].reason


def test_fatal_judge_error_propagates() -> None:
    """输入：一个抛致命异常的判别器。

    输出：异常原样冒泡。
    预期行为：致命错误不能被吞成 Verdict——那会让整批继续跑下去（PLAN 3.1.4）。
    """
    with pytest.raises(Fatal):
        judge_mod.run_judges({}, None, (Judge("fatal", exc=Fatal("全部 provider 都不可用")),))


def test_verdicts_keep_the_declared_order() -> None:
    """输入：三个判别器（通过 / 不通过 / 通过）。

    输出：三份结论。
    预期行为：顺序与入参一致，统计里每个判别器都能对上号。
    """
    out = judge_mod.run_judges({}, None, (Judge("a"), Judge("b", ok=False), Judge("c")))
    assert [(name, verdict.ok) for name, verdict in out] == [("a", True), ("b", False), ("c", True)]


def test_partition_names_splits_the_llm_prefix() -> None:
    """输入：混着规则判别器与 LLM 视角的名字。输出：(规则名, 视角名)。

    预期行为：只按前缀切分，顺序保持声明顺序——通用层据此决定哪些自己装配、
    哪些交给 workflow。
    """
    rules, llm = judge_mod.partition_names(("link", "llm:solvable", "trace", "llm:wording"))
    assert rules == ("link", "trace")
    assert llm == ("solvable", "wording")
    assert judge_mod.partition_names(()) == ((), ())


def test_partition_names_keeps_an_empty_suffix() -> None:
    """输入：`llm:` 后面什么都没写。输出：空名字照实交出去。

    预期行为：不在这里静默丢掉——空视角由装配方报错（`未知质检视角 ''`），
    丢掉的话就成了「写了名字却没跑」。
    """
    rules, llm = judge_mod.partition_names(("llm:",))
    assert rules == ()
    assert llm == ("",)


def test_verdict_is_not_skipped_by_default() -> None:
    """输入：普通 Verdict。输出：无（断言）。预期行为：默认不是「没判成」。"""
    assert judge_mod.Verdict(True).skipped is False
    assert judge_mod.Verdict(False, reason="不行", skipped=True).skipped is True
