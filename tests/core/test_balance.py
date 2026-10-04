"""通用层测试：均衡闸（`jc.pipeline.balance_report`）的口径。

为什么要单独测：这把尺子决定「一批数据要不要被拒」，口径错一次就是全库误判。实测踩过
一次——`invoice` 的 `line_field` 选项是「单据上的每一行」，选项数随样本在 2~4 之间变；
拿「出现过的最大选项数」当参考，它被打成 0.157 的失衡，而按**每条样本自己的选项数**算
期望只有 0.009。反过来也不能放宽：样本量低于 `BALANCE_MIN_TOTAL` 的族判不出东西，
要标成「没判」而不是「通过」。
"""

from __future__ import annotations

import pytest

from jc import pipeline
from jc.jsontypes import JsonValue


def _stats(
    answers: dict[str, dict[str, int]],
    space: dict[str, int],
    sizes: dict[str, dict[str, int]] | None = None,
) -> dict[str, JsonValue]:
    """输入：答案计数、选项/等级空间、可选的选项数直方图。输出：`balance_report` 要的统计字典。"""
    return {
        "answers": {key: dict(bucket) for key, bucket in answers.items()},
        "answer_space": dict(space),
        "choice_sizes": {} if sizes is None else {key: dict(v) for key, v in sizes.items()},
    }


def _one(stats: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    """输入：统计字典、题族键。输出：报告里该族那一项（不存在直接报错）。"""
    for item in pipeline.balance_report(stats, 0.03):
        if item["key"] == key:
            return item
    raise AssertionError(f"报告里没有 {key}")


def test_choice_reference_follows_each_samples_option_count() -> None:
    """输入：选项数一半是 2、一半是 4 的 choice 族，答案按期望铺满。输出：无。

    预期行为：期望占比是 `Σ_{k>i} n_k/k/n`（编号 0/1 各 0.375，2/3 各 0.125），
    按它铺的分布偏差为 0；按「最大选项数 4」的均匀参考则会算出 0.125 的假偏差。
    """
    stats = _stats(
        {"choice:x": {"0": 75, "1": 75, "2": 25, "3": 25}},
        {"choice:x": 4},
        {"choice:x": {"2": 100, "4": 100}},
    )
    item = _one(stats, "choice:x")
    assert item["max_dev"] == 0.0
    assert item["ok"] is True
    assert item["judged"] is True


def test_small_option_samples_are_not_mistaken_for_a_dead_level() -> None:
    """输入：编号 2/3 只可能出现在 4 选项的样本里，而它们确实按期望出现。输出：无。

    预期行为：编号 2/3 的占比低（0.125）**不是**失衡——只有 4 选项的样本取得到它们，
    这正是「每条样本在它自己的选项里均匀乱答」的占比。
    """
    stats = _stats(
        {"choice:x": {"0": 70, "1": 80, "2": 30, "3": 20}},
        {"choice:x": 4},
        {"choice:x": {"2": 100, "4": 100}},
    )
    assert _one(stats, "choice:x")["ok"] is True


def test_constant_option_count_keeps_the_uniform_reference() -> None:
    """输入：选项数恒为 3 的族，一份均匀、一份偏斜。输出：无。

    预期行为：均匀那份偏差 0；偏斜那份（50/5/5）偏差 0.5，判成失衡——换参考不能把
    「真的偏」也一起放过。
    """
    sizes = {"choice:even": {"3": 60}, "choice:skew": {"3": 60}}
    space = {"choice:even": 3, "choice:skew": 3}
    even = _one(_stats({"choice:even": {"0": 20, "1": 20, "2": 20}}, space, sizes), "choice:even")
    skew = _one(_stats({"choice:skew": {"0": 50, "1": 5, "2": 5}}, space, sizes), "choice:skew")
    assert even["ok"] is True
    assert skew["ok"] is False
    assert skew["max_dev"] == pytest.approx(0.5, abs=1e-4)


def test_a_two_option_family_that_always_answers_the_same_is_caught() -> None:
    """输入：选项数恒为 2、答案全是编号 0 的族。输出：无。

    预期行为：偏差 0.5，判成失衡——「选项数随样本变」这条新参考不会把常量答案放过。
    """
    stats = _stats({"choice:x": {"0": 60}}, {"choice:x": 2}, {"choice:x": {"2": 60}})
    item = _one(stats, "choice:x")
    assert item["ok"] is False
    assert item["max_dev"] == pytest.approx(0.5, abs=1e-4)


def test_score_and_old_artifacts_fall_back_to_a_uniform_reference() -> None:
    """输入：没有选项数记录的族（score 与旧产物）。输出：无。

    预期行为：参考一律 1/space；四档各 25 份判均衡，100/0/0/0 判失衡。
    """
    even = _one(_stats({"score:z": {"0": 25, "1": 25, "2": 25, "3": 25}}, {"score:z": 4}), "score:z")
    skew = _one(_stats({"score:z": {"0": 100}}, {"score:z": 4}), "score:z")
    assert even["ok"] is True
    assert skew["ok"] is False
    assert skew["max_dev"] == pytest.approx(0.75, abs=1e-4)


def test_noul_reference_is_the_true_share() -> None:
    """输入：noul 族 True 占 70%。输出：无。预期行为：偏差 = |0.7 − 0.5| = 0.2。"""
    item = _one(_stats({"noul:w": {"True": 70, "False": 30}}, {"noul:w": 2}), "noul:w")
    assert item["max_dev"] == pytest.approx(0.2, abs=1e-4)
    assert item["space"] == 2


def test_a_family_below_the_minimum_sample_size_is_not_judged() -> None:
    """输入：样本量低于 `BALANCE_MIN_TOTAL` 的族，答案还全是一个值。输出：无。

    预期行为：`judged=False` 且 `ok=True`——那个规模上判不出东西，报告要写「没判」，
    而不是假装它通过了（也不能拿它当失衡去拦整批）。
    """
    small = pipeline.BALANCE_MIN_TOTAL - 1
    stats = _stats({"choice:x": {"0": small}}, {"choice:x": 2}, {"choice:x": {"2": small}})
    item = _one(stats, "choice:x")
    assert item["judged"] is False
    assert item["ok"] is True


def test_a_family_exactly_at_the_minimum_sample_size_is_judged() -> None:
    """输入：样本量**恰好等于** `BALANCE_MIN_TOTAL` 的族。输出：无。

    预期行为：`judged=True`——判据是 `total >= min_total`，「刚好够」算可判；改成 `>`
    会把刚好够的族降级成「没判」，于是它既不算通过也不算失衡，从报告里静默消失。
    上面那条只钉了「差一条」的那一侧，这一条钉住等号本身。
    """
    exact = pipeline.BALANCE_MIN_TOTAL
    stats = _stats({"choice:x": {"0": exact}}, {"choice:x": 2}, {"choice:x": {"2": exact}})
    item = _one(stats, "choice:x")
    assert item["judged"] is True


def test_the_report_is_sorted_by_deviation_descending() -> None:
    """输入：两个偏差不同的族。输出：无。预期行为：偏差大的排在前面（报告第一眼最刺眼）。"""
    stats = _stats(
        {"choice:small": {"0": 20, "1": 20, "2": 20}, "choice:big": {"0": 40, "1": 10, "2": 10}},
        {"choice:small": 3, "choice:big": 3},
        {"choice:small": {"3": 60}, "choice:big": {"3": 60}},
    )
    keys = [item["key"] for item in pipeline.balance_report(stats, 0.03)]
    assert keys == ["choice:big", "choice:small"]
