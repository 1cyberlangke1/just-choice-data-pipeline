"""通用层测试：workflow 注册表与组装器（不依赖任何具体 workflow）。"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from jc import assemble
from jc.jsontypes import JsonValue
from jc.workflow import (
    LabeledQuestion,
    answer_value,
    compare_answers,
    compare_expected,
    discover,
    family_of,
    load,
    register,
    registered,
    show_answers,
)

# 仓库根：CLI 子进程要以它为工作目录，才能 import 到 workflows/ 包。
ROOT = Path(__file__).resolve().parents[2]


def test_family_of_strips_suffix() -> None:
    assert family_of("can_move") == "can_move"
    assert family_of("can_move_2") == "can_move"
    assert family_of("off_by_one_11") == "off_by_one"


def test_register_rejects_duplicate() -> None:
    class Dummy:
        name = "dummy-for-test"

    register("dummy-for-test")(lambda: Dummy())
    with pytest.raises(ValueError):
        register("dummy-for-test")(lambda: Dummy())
    assert "dummy-for-test" in registered()


def test_load_unknown_workflow_lists_known() -> None:
    with pytest.raises(KeyError):
        load("no-such-workflow")


def test_load_maze_by_dynamic_import() -> None:
    """通用层不静态依赖 maze：靠动态导入触发自注册。"""
    workflow = load("maze")
    assert workflow.name == "maze"
    assert workflow.needs == frozenset()  # 纯构造式


def test_discover_finds_maze_package() -> None:
    """输入：无。输出：已注册的 workflow 名。

    预期行为：扫描 `workflows/` 包并逐个导入以触发自注册——否则注册表只在
    「有人恰好 import 过」时才有内容。
    """
    assert "maze" in discover()


def test_cli_workflows_lists_maze_in_fresh_process() -> None:
    """输入：独立子进程执行 `python -m jc workflows`。输出：进程输出与退出码。

    预期行为：新进程里注册表是空的，`jc workflows` 仍必须自己发现并列出 maze
    （否则这条命令恒为空，而 `--workflow` 的帮助文本正指着它）。
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    proc = subprocess.run(
        [sys.executable, "-m", "jc", "workflows"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "maze" in proc.stdout.split()


def test_assemble_shuffles_and_maps_answer_to_index() -> None:
    labeled = [
        LabeledQuestion(
            family="pick",
            question={
                "type": "choice",
                "instructions": "which?",
                "criteria": {"a": None, "b": None, "c": None},
            },
            answer="b",
        )
    ]
    sample = assemble.assemble_sample("maze-train-0000000", "s", labeled, random.Random(1))
    criteria = list(sample["questions"]["pick"]["criteria"])
    assert sample["answers"]["pick"] == {"type": "choice", "choice": criteria.index("b")}


def test_assemble_suffixes_repeated_families() -> None:
    labeled = [
        LabeledQuestion("ask", {"type": "noul", "instructions": "x"}, True, shuffle=False),
        LabeledQuestion("ask", {"type": "noul", "instructions": "y"}, False, shuffle=False),
    ]
    sample = assemble.assemble_sample("maze-train-0000000", "s", labeled, random.Random(1))
    assert list(sample["questions"]) == ["ask", "ask_2"]
    assert list(sample["answers"]) == ["ask", "ask_2"]


def _sample(
    question: dict[str, JsonValue], answer: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """输入：一道题、它的答案。输出：一条最小样本（够 `compare_answers` 用）。"""
    return {"questions": {"pick": question}, "answers": {"pick": answer}}


def test_show_answers_joins_candidates() -> None:
    """输入：可接受答案元组。输出：给人看的写法。预期行为：多个候选用 ` / ` 连、逐个转字符串。"""
    assert show_answers(("甲",)) == "甲"
    assert show_answers(("甲", "乙")) == "甲 / 乙"
    assert show_answers((True, 3)) == "True / 3"


def test_compare_answers_accepts_any_candidate() -> None:
    """输入：一条 choice 样本 + 两个可接受答案。输出：空列表。

    预期行为：比的是**选项描述原文**（不是选项名），任一候选对上就算对——这正是
    `choice` 存编号、判别器却要拿描述比的理由（范式：选项名是各语言自己起的）。
    """
    question = {"type": "choice", "instructions": "x", "criteria": {"a": "甲", "b": "乙"}}
    assert compare_answers(_sample(question, {"type": "choice", "choice": 1}), lambda _: ("乙", "丙")) == []


def test_compare_answers_reports_every_kind_of_mismatch() -> None:
    """输入：四种对不上的样本。输出：逐条问题。

    预期行为：答案不对、编号越界、选项形状不对、按 state 重算不出来，各记一笔，
    谁也不静默吞掉——判别器崩了不能把整批带走，所以形状不对是「记一笔」而不是抛异常。
    """
    question = {"type": "choice", "instructions": "x", "criteria": {"a": "甲", "b": "乙"}}
    wrong = _sample(question, {"type": "choice", "choice": 0})
    assert compare_answers(wrong, lambda _: ("乙",)) == ["pick: 答案指向 甲，按 state 重算是 乙"]
    out_of_range = _sample(question, {"type": "choice", "choice": 9})
    assert compare_answers(out_of_range, lambda _: ("乙",)) == ["pick: 答案编号越界（9）"]
    bad_shape = _sample(
        {"type": "choice", "instructions": "x", "criteria": ["甲"]}, {"type": "choice", "choice": 0}
    )
    assert compare_answers(bad_shape, lambda _: ("甲",)) == ["pick: 选项形状不对"]

    def boom(family: str) -> tuple[str | bool | int, ...]:
        """输入：题族。输出：永不返回。预期行为：模拟「按 state 重算不出来」。"""
        raise KeyError(family)

    assert compare_answers(wrong, boom) == ["pick: 重算不出答案（'pick'）"]


def test_compare_answers_covers_noul_score_and_unknown_type() -> None:
    """输入：noul / score / 未知题型的样本。输出：逐条问题。

    预期行为：noul 比布尔、score 比档位；未知题型当场记一笔而不是崩。
    """
    noul = _sample({"type": "noul", "instructions": "x"}, {"type": "noul", "noul": True})
    assert compare_answers(noul, lambda _: (False,)) == ["pick: 答案是 True，按 state 重算是 False"]
    score = _sample({"type": "score", "instructions": "x"}, {"type": "score", "score": 2})
    assert compare_answers(score, lambda _: (1,)) == ["pick: 答案是 2，按 state 重算是 1"]
    weird = _sample({"type": "pairwise", "instructions": "x"}, {"type": "pairwise"})
    assert compare_answers(weird, lambda _: ()) == ["pick: 未知题型 'pairwise'"]


def test_answer_value_reads_each_type_and_rejects_bad_shapes() -> None:
    """输入：三种题型的答案 + 各种坏形态。输出：无。

    预期行为：choice 给**选项名**、score 给**档位下标**、noul 只认真布尔；编号越界、编号写成
    布尔 / 字符串、档位表不是列表、题型认不出，一律给 None——那是对不上，不是崩。
    """
    choice = {"type": "choice", "criteria": {"a": "甲", "b": "乙"}}
    assert answer_value(choice, {"choice": 1}) == "b"
    assert answer_value(choice, {"choice": 2}) is None
    assert answer_value(choice, {"choice": True}) is None
    assert answer_value(choice, {"choice": "1"}) is None
    assert answer_value({"type": "choice", "criteria": []}, {"choice": 0}) is None
    score = {"type": "score", "criteria": ["甲", "乙", "丙"]}
    assert answer_value(score, {"score": 2}) == 2
    assert answer_value(score, {"score": 3}) is None
    assert answer_value(score, {"score": False}) is None
    assert answer_value({"type": "score", "criteria": {}}, {"score": 0}) is None
    assert answer_value({"type": "noul"}, {"noul": True}) is True
    assert answer_value({"type": "noul"}, {"noul": 1}) is None
    assert answer_value({"type": "noul"}, {}) is None
    assert answer_value({"type": "bogus"}, {}) is None


def test_compare_expected_reports_every_kind_of_mismatch() -> None:
    """输入：一条好样本 + 五种改坏法。输出：无。

    预期行为：对得上给空列表；未知题族、多出来的题、答案不对、少了题、样本形状不对各记一笔。
    「少了题」那一条走的是**期望表**——遍历产物时整族缺席会一声不响，漏报比误报危险。
    """
    question = {"type": "choice", "criteria": {"a": "甲", "b": "乙"}}
    sample: dict[str, JsonValue] = {
        "questions": {"pick": question},
        "answers": {"pick": {"choice": 1}},
    }
    assert compare_expected(sample, {"pick": ["b"]}, ("pick",)) == []

    unknown = {"questions": {"mystery": question}, "answers": {"mystery": {"choice": 1}}}
    assert compare_expected(unknown, {}, ("pick",)) == ["未知题族 'mystery'（题号 mystery）"]

    extra = {
        "questions": {"pick": question, "pick_2": question},
        "answers": {"pick": {"choice": 1}, "pick_2": {"choice": 1}},
    }
    assert compare_expected(extra, {"pick": ["b"]}, ("pick",)) == [
        "pick_2 多出来了：文本里只算得出 1 道 pick 题"
    ]

    wrong = {"questions": {"pick": question}, "answers": {"pick": {"choice": 0}}}
    assert compare_expected(wrong, {"pick": ["b"]}, ("pick",)) == [
        "pick：答案是 'a'，按文本重算应为 'b'"
    ]

    assert compare_expected(sample, {"pick": ["b", "a"]}, ("pick",)) == ["pick 少了 1 道题"]
    assert compare_expected({"questions": 1, "answers": 1}, {}, ("pick",)) == [
        "样本的 questions / answers 不是对象"
    ]
