"""题族：四道题、题型、答案、判据写进题面、选项表与判据表。"""

from __future__ import annotations

import random

import pytest

from tests.workflows.cachereuse.conftest import _make_spec
from workflows.cachereuse import locales, questions
from workflows.cachereuse import spec as sp

LOC = locales.load("zh")


def test_families_are_the_four_known_ones() -> None:
    """输入：无。输出：题族清单。

    预期行为：四个题族、顺序稳定（先档位、再两个 choice、最后那个布尔），三张分类表正好
    把四族分完——多一族少一族都会让下面的「答案在选项里」漏查。
    """
    assert questions.FAMILIES == ("reuse_band", "diverge_at", "diff_kind", "reuse_half")
    assert questions.SCORE_FAMILIES == ("reuse_band",)
    assert questions.CHOICE_FAMILIES == ("diverge_at", "diff_kind")
    assert questions.NOUL_FAMILIES == ("reuse_half",)
    assert set(questions.FAMILIES) == (
        set(questions.SCORE_FAMILIES)
        | set(questions.CHOICE_FAMILIES)
        | set(questions.NOUL_FAMILIES)
    )


def test_build_returns_one_question_per_family() -> None:
    """输入：一条规格。输出：无。预期行为：四道题、族名一一对应、题型正确。"""
    built = questions.build(_make_spec(1), LOC, 0, random.Random(0))
    assert [item.family for item in built] == list(questions.FAMILIES)
    types = {item.family: item.question["type"] for item in built}
    for family in questions.CHOICE_FAMILIES:
        assert types[family] == "choice"
    assert types["reuse_band"] == "score"
    assert types["reuse_half"] == "noul"


def test_only_choice_options_may_be_shuffled() -> None:
    """输入：一条规格。输出：无。

    预期行为：只有 choice 允许打乱——`reuse_band` 的四档有自然顺序（从「只差最后一步」
    到「一步都不能用」），打乱它会把「档位」读成「随便一个等级」。
    """
    for item in questions.build(_make_spec(2), LOC, 0, random.Random(0)):
        assert item.shuffle is (item.question["type"] == "choice"), item.family


def test_answers_match_the_spec_for_every_case() -> None:
    """输入：全空间 case。输出：无。

    预期行为：四道题的答案都与真值规格算出来的一致——choice 是选项名（组装器再换成编号）、
    noul 是布尔、score 是档位下标。这里是「生成器把答案算错了」那条路的第一道闸。
    """
    for case in sp.build_cases(sp.ReuseConfig(split="train")):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        answers = {
            item.family: item.answer
            for item in questions.build(spec, LOC, 0, random.Random(0))
        }
        assert answers["reuse_band"] == sp.BANDS.index(case.band), case.describe()
        assert answers["diverge_at"] == case.diverge_band(), case.describe()
        assert answers["diff_kind"] == case.diff_kind, case.describe()
        assert answers["reuse_half"] is case.reuse_half, case.describe()


def test_choice_answers_are_option_names_inside_the_criteria() -> None:
    """输入：全空间 case。输出：无。

    预期行为：choice 的答案必须是 `criteria` 里的**键**——组装器靠 `list(criteria).index`
    取编号，答案不在选项里会在组装期直接报错（那是构造 bug，不能进产物）。
    """
    for case in sp.build_cases(sp.ReuseConfig(split="train")):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        for item in questions.build(spec, LOC, 0, random.Random(0)):
            if item.question["type"] != "choice":
                continue
            assert item.answer in item.question["criteria"], item.family


def test_options_of_diverge_at_has_the_four_position_bands() -> None:
    """输入：一条规格。输出：无。

    预期行为：四个位置档固定、描述互不相同——档位与**步号**挂钩（第 1 / 第 2 / 中间 / 最后），
    所以同一张选项表在三种步数下都成立。
    """
    options = questions.options_of("diverge_at", _make_spec(3), LOC)
    assert set(options) == {"first", "second", "middle", "last"}
    assert len(set(options.values())) == 4


def test_options_of_diff_kind_has_the_two_causes() -> None:
    """输入：一条规格。输出：无。预期行为：两个选项（动作 / 元素）、描述互不相同。"""
    options = questions.options_of("diff_kind", _make_spec(4), LOC)
    assert set(options) == {"action", "target"}
    assert len(set(options.values())) == 2


def test_options_of_rejects_unknown_family() -> None:
    """输入：不存在的题族。输出：无。预期行为：报错（不然会静默返回另一族的选项）。"""
    with pytest.raises(ValueError, match="未知题族"):
        questions.options_of("reuse_half", _make_spec(5), LOC)


def test_choice_rejects_bad_answer_and_duplicate_options() -> None:
    """输入：答案不在选项里 / 选项描述重复。输出：无。预期行为：构造期报错。"""
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("x", "题面", {"a": "1", "b": "2"}, "c")
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("x", "题面", {"a": "1", "b": "1"}, "a")


def test_rules_block_is_the_joined_lines() -> None:
    """输入：语言字典。输出：无。

    预期行为：判据原文就是 `[rules].lines` 原样连起来——判据不能藏在判别器或生成器里
    （范式 §9.8），只读 state 与题面的读者必须能自己算。
    """
    assert questions.rules_block(LOC) == "\n".join(LOC["rules"]["lines"])


def test_every_instruction_carries_the_rules_text() -> None:
    """输入：一条规格。输出：无。预期行为：四道题的题面都以判据原文结尾。"""
    rules = questions.rules_block(LOC)
    for item in questions.build(_make_spec(6), LOC, 0, random.Random(0)):
        assert item.question["instructions"].endswith(rules), item.family


def test_diff_kind_instruction_fills_the_step_number() -> None:
    """输入：一条规格（5 步、约一半 → 第一个不同在第 4 步）。输出：无。

    预期行为：题面里的 `{step}` 换成**第一个不同的步号**——占位符没填的话题面会问「第 {step} 步」，
    考生没法答，而契约校验又看不出来。
    """
    spec = _make_spec(7)
    assert spec.first_diff == 4, spec.first_diff
    item = next(
        item for item in questions.build(spec, LOC, 0, random.Random(0)) if item.family == "diff_kind"
    )
    assert "第 4 步" in item.question["instructions"]


def test_score_criteria_is_the_four_level_table() -> None:
    """输入：一条规格。输出：无。

    预期行为：`criteria` 是四档描述（原序，不打乱），答案就是这个数组里的下标——下标与
    `spec.BANDS` 的顺序一一对应。
    """
    spec = _make_spec(8)
    item = next(
        item for item in questions.build(spec, LOC, 0, random.Random(0)) if item.family == "reuse_band"
    )
    assert item.question["criteria"] == list(LOC["fam"]["reuse_band"]["levels"])
    assert len(set(item.question["criteria"])) == len(sp.BANDS)
    assert item.answer == sp.BANDS.index(spec.case.band)
    assert 0 <= item.answer < len(item.question["criteria"])


def test_noul_criteria_has_true_and_false() -> None:
    """输入：一条规格。输出：无。

    预期行为：noul 带 `criteria`（官方支持描述，那我们就带上）——真 / 假两行都在，
    答案必须是布尔而不是字符串 `"true"`。
    """
    item = next(
        item for item in questions.build(_make_spec(9), LOC, 0, random.Random(0)) if item.family == "reuse_half"
    )
    assert set(item.question["criteria"]) == {"true", "false"}
    assert isinstance(item.answer, bool)


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_no_question_text_keeps_a_placeholder(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：题面、选项描述、等级描述里都不许留下 `{...}` 占位符——留一个就是一道
    问不出来的题，而契约校验只管形状，管不到这个。
    """
    loc = locales.load(lang)
    for item in questions.build(_make_spec(10), loc, 0, random.Random(0)):
        texts = [item.question["instructions"]]
        criteria = item.question.get("criteria")
        if isinstance(criteria, dict):
            texts.extend(str(value) for value in criteria.values())
        elif isinstance(criteria, list):
            texts.extend(str(value) for value in criteria)
        for text in texts:
            assert "{" not in text and "}" not in text, f"{lang}/{item.family}: {text}"


def test_build_is_a_pure_function() -> None:
    """输入：同一规格、两个不同 RNG。输出：无。预期行为：题目逐字段一致（不掷骰子）。"""
    spec = _make_spec(11)
    first = questions.build(spec, LOC, 0, random.Random(0))
    second = questions.build(spec, LOC, 0, random.Random(999))
    assert first == second


def test_phrasing_index_cycles_through_the_pool() -> None:
    """输入：三种问法编号。输出：无。

    预期行为：问法按编号取，取遍池子里的每一条——同一 case 的多条样本靠它换说法，
    三种编号给同一句话等于这一维白设。
    """
    pool = LOC["fam"]["reuse_band"]["q"]
    seen = {
        questions.build(_make_spec(12), LOC, index, random.Random(0))[0].question["instructions"]
        for index in range(len(pool))
    }
    assert len(seen) == len(pool)
