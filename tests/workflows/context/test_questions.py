"""题族：出题规则、两阶段依赖、选项与答案，以及「均衡靠构造」的验收。"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from workflows.context import locales, questions
from workflows.context import spec as sp
from workflows.context.workflow import ContextWorkflow

WF = ContextWorkflow()
LOC = locales.load("zh")
SEED = 20260927


def _spec(**overrides: object) -> sp.SessionSpec:
    """输入：case 覆盖项。输出：一条真值规格（固定种子）。"""
    fields = {
        "keep_form": "err_small",
        "drop_form": "off_small",
        "kind_set": "read_run",
        "extra": "keep",
        "order": "p0",
        "phrasing": 0,
    }
    fields.update(overrides)
    return sp.sample_spec(sp.SessionCase(**fields), "train", 0, 0, random.Random(3))  # type: ignore[arg-type]


def _all_id_specs() -> list[sp.SessionSpec]:
    """输入：无。输出：训练侧整张 case 矩阵的规格（固定种子，可复现）。"""
    rng = random.Random(SEED)
    return [
        sp.sample_spec(case, "train", index, case.phrasing, rng)
        for index, case in enumerate(sp.build_cases(WF.config_for("train")))
    ]


def test_build_emits_one_action_question_per_block() -> None:
    """输入：一条规格。输出：无。预期行为：五道 `action`，答案与各块的动作一致。"""
    spec = _spec()
    items = questions.build(spec.blocks, spec.budget, LOC, 0)
    actions = [item for item in items if item.family == "action"]
    assert len(actions) == len(spec.blocks) == 5
    assert [item.answer for item in actions] == [block.action() for block in spec.blocks]
    for item in actions:
        assert item.question["type"] == "choice"
        assert set(item.question["criteria"]) == set(sp.ACTIONS)
        assert item.answer in item.question["criteria"]


def test_build_emits_the_gate_and_the_two_position_families() -> None:
    """输入：一条 `spot` / `top` 都该出的规格。输出：无。

    预期行为：门控是 noul（带 true / false 判据），定位与极值都是 choice，答案就是位置键。
    """
    spec = _spec()
    items = questions.build(spec.blocks, spec.budget, LOC, 0)
    by_family = {item.family: item for item in items if item.family != "action"}
    assert set(by_family) == {"compact", "spot", "top"}
    gate = by_family["compact"]
    assert gate.question["type"] == "noul"
    assert set(gate.question["criteria"]) == {"true", "false"}
    assert gate.answer is spec.compact
    assert by_family["spot"].answer == f"p{spec.drop_position}"
    assert by_family["top"].answer == f"p{spec.top_position}"
    assert list(by_family["spot"].question["criteria"]) == [f"p{i}" for i in range(1, 6)]


def test_families_for_matches_what_build_emits() -> None:
    """输入：整张 case 矩阵。输出：无。

    预期行为：`families_for` 说的题族集合与 `build` 真的出的一致——审计靠它算「应有题族」，
    对不上就会漏报（或误报）缺题。
    """
    for case in sp.build_cases(sp.SessionConfig()):
        spec = sp.sample_spec(case, "train", 0, 0, random.Random(1))
        emitted = {item.family for item in questions.build(spec.blocks, spec.budget, LOC, 0)}
        assert emitted == set(questions.families_for(spec.blocks)), case.describe()


def test_spot_and_top_are_two_stage() -> None:
    """输入：两种「没有唯一解」的会话。输出：无。

    预期行为：重复角色是丢弃块时不出 `spot`；重复角色是截断且基本保留块是小块时不出 `top`。
    """
    dropped = _spec(extra="drop")
    assert dropped.drop_position is None
    families = {item.family for item in questions.build(dropped.blocks, dropped.budget, LOC, 0)}
    assert "spot" not in families and "top" in families
    tied = _spec(extra="trim")
    assert tied.top_position is None
    families = {item.family for item in questions.build(tied.blocks, tied.budget, LOC, 0)}
    assert "top" not in families and "spot" in families


def test_position_options_describe_every_block() -> None:
    """输入：整张 case 矩阵。输出：无。

    预期行为：位置选项的描述里带那一块的抬头，所以选项之间两两不同（选项描述撞车是构造 bug）。
    """
    for case in sp.build_cases(sp.SessionConfig()):
        spec = sp.sample_spec(case, "train", 0, 0, random.Random(2))
        options = questions._position_options(spec.blocks, LOC)
        assert len(options) == 5
        assert len(set(options.values())) == 5, case.describe()
        for index, block in enumerate(spec.blocks, start=1):
            assert LOC["kind"][block.kind] in options[f"p{index}"]


def test_fill_rejects_a_leftover_placeholder() -> None:
    """输入：模板里有一个没填的槽。输出：无。

    预期行为：直接报错——题面少填一个槽会变成一道读不懂的题，而答案照样发出去。
    """
    with pytest.raises(ValueError, match="还剩占位符"):
        questions._fill("第 {n} 块 {extra} 怎么办？", n="1")


def test_choice_rejects_duplicate_options_and_unknown_answers() -> None:
    """输入：描述重复的选项 / 不在选项里的答案。输出：无。预期行为：都当场报错。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("action", "问", {"a": "一样", "b": "一样"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("action", "问", {"a": "甲", "b": "乙"}, "c")


def test_build_rejects_an_empty_session() -> None:
    """输入：一个块都没有。输出：无。预期行为：报人话。"""
    with pytest.raises(ValueError, match="一个块都没有"):
        questions.build((), 1024, LOC, 0)


def test_action_answers_are_exactly_balanced_by_construction() -> None:
    """输入：训练侧整张矩阵。输出：无。

    预期行为：四个动作各占 **1/4**（每段会话四个基本动作各一个 + 重复角色均匀）——
    均衡是构造出来的，不靠事后过滤。
    """
    counter: Counter[str] = Counter()
    for spec in _all_id_specs():
        for item in questions.build(spec.blocks, spec.budget, LOC, 0):
            if item.family == "action":
                counter[item.answer] += 1
    total = sum(counter.values())
    assert total == 5 * len(_all_id_specs())
    for action in sp.ACTIONS:
        assert counter[action] / total == pytest.approx(0.25)


def test_position_answers_are_uniform_by_construction() -> None:
    """输入：训练侧整张矩阵。输出：无。

    预期行为：`spot` 与 `top` 的答案位置在五个位置上各占 1/5——五种排布是拉丁方。
    """
    counters: dict[str, Counter[str]] = {"spot": Counter(), "top": Counter()}
    for spec in _all_id_specs():
        for item in questions.build(spec.blocks, spec.budget, LOC, 0):
            if item.family in counters:
                counters[item.family][item.answer] += 1
    for family, counter in counters.items():
        total = sum(counter.values())
        assert total > 0
        for position in range(1, 6):
            assert counter[f"p{position}"] / total == pytest.approx(0.2), family


def test_compact_answers_stay_within_the_noise_floor() -> None:
    """输入：训练侧整张矩阵。输出：无。

    预期行为：`compact` 的目标标签是均匀抽的，所以 True 占比落在「3 倍标准误」之内——
    这是构造（先定标签再挑预算），不是事后过滤。
    """
    answers = [
        item.answer
        for spec in _all_id_specs()
        for item in questions.build(spec.blocks, spec.budget, LOC, 0)
        if item.family == "compact"
    ]
    assert answers
    share = sum(1 for value in answers if value) / len(answers)
    noise = 3 * math.sqrt(0.25 / len(answers))
    assert abs(share - 0.5) <= noise, share


def test_question_texts_carry_the_policy() -> None:
    """输入：整张 case 矩阵。输出：无。

    预期行为：每道题的题面里都带判据原文（范式 §9.8）——只读 state 与题面的考生要能自己
    把答案算出来，所以判据不能只留在生成器代码里。
    """
    policy = LOC["policy"]["text"]
    for case in sp.build_cases(sp.SessionConfig()):
        spec = sp.sample_spec(case, "train", 0, 0, random.Random(4))
        for item in questions.build(spec.blocks, spec.budget, LOC, 0):
            assert item.question["instructions"].endswith(policy)


def test_phrasing_picks_different_questions_but_the_same_answers() -> None:
    """输入：同一批块的两套句面。输出：无。

    预期行为：问法不同、答案完全相同——句面留出因此不会把哪一族的分布压歪。
    """
    blocks = sp.blocks_of(sp.SessionCase("err_small", "off_small", "read_run", "keep", "p1", 0))
    first = questions.build(blocks, 1024, LOC, 0)
    second = questions.build(blocks, 1024, LOC, 1)
    assert [item.answer for item in first] == [item.answer for item in second]
    assert [item.question["instructions"] for item in first] != [
        item.question["instructions"] for item in second
    ]
