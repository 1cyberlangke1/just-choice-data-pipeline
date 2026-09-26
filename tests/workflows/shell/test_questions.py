"""题族的单元测试：题型、金答案、选项、以及「打乱之后位置不泄露标签」。"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from jc import assemble
from workflows.shell import locales, questions
from workflows.shell import spec as sp


def build(case: sp.ShellCase, seed: int = 1) -> tuple[sp.CommandSpec, list]:
    """输入：case、种子。输出：(规格, 题目列表)。预期行为：纯代码推导，不碰模型。"""
    spec = sp.sample_spec(random.Random(seed), case, "train", 0, 0)
    return spec, questions.build(spec, locales.load("zh"), 0, random.Random(seed))


def test_all_families_are_produced_with_the_right_types() -> None:
    """输入：一条普通命令。输出：无。预期行为：8 个题族齐全，题型与声明一致。"""
    _, labeled = build(sp.ShellCase("bash", "remove_tree", "build_dir", "build_dir", "plain", "imperative", 2))
    assert [item.family for item in labeled] == list(questions.FAMILIES)
    types = {item.family: item.question["type"] for item in labeled}
    assert types["verdict"] == "choice" and types["target_owner"] == "choice"
    assert types["blast_radius"] == "score"
    assert all(types[name] == "noul" for name in questions.NOUL_FAMILIES)


def test_answers_match_the_command_facts() -> None:
    """输入：几条代表性命令。输出：无。

    预期行为：金答案就是 `spec.flags` 推出来的那几项——答案与判据同源。
    """
    spec, labeled = build(sp.ShellCase("bash", "read_file", "ssh_key", "ssh_key", "elevated", "imperative", 1))
    answers = {item.family: item.answer for item in labeled}
    flags = spec.flags
    assert answers["verdict"] == "ask"  # 只读、与目标一致，但提权
    assert answers["target_owner"] == "home"
    assert answers["matched_goal"] is True
    assert answers["elevated"] is True
    assert answers["network"] is False
    assert answers["destructive"] is False
    assert answers["touches_secrets"] is True
    assert answers["blast_radius"] == sp.blast_of(flags)


def test_misaligned_write_is_denied() -> None:
    """输入：一条「会写数据且与目标不符」的命令。输出：无。预期行为：答案是 deny。"""
    _, labeled = build(sp.ShellCase("bash", "remove_tree", "build_dir", "dist_dir", "plain", "imperative", 1))
    answers = {item.family: item.answer for item in labeled}
    assert answers["verdict"] == "deny"
    assert answers["matched_goal"] is False
    assert answers["destructive"] is True


def test_option_descriptions_must_be_unique() -> None:
    """输入：两个描述一模一样的选项。输出：异常。

    预期行为：直接报错——描述重复意味着两个选项都说得通，是构造 bug，
    必须在生成时炸掉而不是产出一条坏数据。
    """
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("verdict", "题面", {"a": "同样的话", "b": "同样的话"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("verdict", "题面", {"a": "甲", "b": "乙"}, "c")


def test_verdict_options_are_shuffled_so_position_does_not_leak() -> None:
    """输入：同一道 verdict 题组装很多次。输出：答案位置的分布。

    预期行为：三个位置都出现过——`allow` 只占命令表的一小部分，若不打乱选项，
    「答案位置」就等于标签分布，等于把捷径白送给考生（范式 §5.8）。
    """
    case = sp.ShellCase("bash", "remove_tree", "build_dir", "build_dir", "plain", "imperative", 1)
    spec, labeled = build(case)
    positions: Counter[int] = Counter()
    for seed in range(60):
        row = assemble.assemble_sample("shell-train-0000000", "state", labeled, random.Random(seed))
        positions[row["answers"]["verdict"]["choice"]] += 1
    assert set(positions) == {0, 1, 2}


def test_score_answer_is_an_index_into_the_levels() -> None:
    """输入：几条命令。输出：无。预期行为：score 答案在等级表范围内，等级表就是题面。"""
    loc = locales.load("zh")
    for op_name in ("list_dir", "chmod_world", "drop_table"):
        spec, labeled = build(sp.ShellCase("bash", op_name, "build_dir" if op_name != "drop_table" else "orders_db", "build_dir" if op_name != "drop_table" else "orders_db", "plain", "imperative", 1))
        item = next(one for one in labeled if one.family == "blast_radius")
        assert item.question["criteria"] == loc["fam"]["blast_radius"]["levels"]
        assert 0 <= item.answer < len(item.question["criteria"])


def test_noul_families_are_not_shuffled() -> None:
    """输入：全部题目。输出：无。预期行为：noul 与 score 都不打乱（判据在题面里，顺序有意义）。"""
    _, labeled = build(sp.ShellCase("bash", "list_dir", "build_dir", "build_dir", "plain", "imperative", 1))
    for item in labeled:
        if item.question["type"] in ("noul", "score"):
            assert item.shuffle is False


def test_unknown_noul_family_reports_the_name() -> None:
    """输入：一个不存在的 noul 题族。输出：异常。预期行为：报错点名题族。"""
    spec = sp.sample_spec(
        random.Random(1),
        sp.ShellCase("bash", "list_dir", "build_dir", "build_dir", "plain", "imperative", 1),
        "train",
        0,
        0,
    )
    with pytest.raises(KeyError, match="made_up"):
        questions._noul_answer("made_up", spec)
