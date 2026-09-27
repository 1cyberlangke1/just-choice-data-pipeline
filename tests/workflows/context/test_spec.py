"""真值规格：处置表、开销表、排布、块构造、case 矩阵、值留出与指纹。"""

from __future__ import annotations

import random
from collections import Counter
from itertools import product

import pytest

from workflows.context import spec as sp
from workflows.context.spec import BlockCase, SessionCase, SessionConfig

# ---------- 处置表 ----------


def test_action_table_over_the_whole_grid() -> None:
    """输入：类型 × 大小 × 年龄 × 主题全组合。输出：无。

    预期行为：三条规则按顺序判——报错一律保留（不管年龄 / 大小 / 主题）；不同主题一律丢弃；
    其余按年龄与大小（最近小 → 保留、最近大 → 截断、过期小 → 摘要、过期大 → 丢弃）。
    """
    for kind, size, age, topic in product(sp.KINDS, sp.SIZES, sp.AGES, sp.TOPICS):
        action = sp.action_of(kind, size, age, topic)
        if kind == "err":
            assert action == sp.KEEP, (kind, size, age, topic)
        elif topic == "other":
            assert action == sp.DROP, (kind, size, age, topic)
        elif age == "recent":
            assert action == (sp.KEEP if size == "small" else sp.TRIM)
        else:
            assert action == (sp.DIGEST if size == "small" else sp.DROP)


@pytest.mark.parametrize(
    ("field", "value"),
    [("kind", "nope"), ("size", "nope"), ("age", "nope"), ("topic", "nope")],
)
def test_action_rejects_unknown_keys(field: str, value: str) -> None:
    """输入：某一维的非法取值。输出：无。预期行为：报人话并点名那一维，不静默给默认动作。"""
    args = {"kind": "ask", "size": "small", "age": "recent", "topic": "same"}
    args[field] = value
    with pytest.raises(KeyError, match=f"没有 {field}"):
        sp.action_of(**args)


def test_error_blocks_beat_the_topic_rule() -> None:
    """输入：一个「报错 + 大 + 过期 + 无关主题」的块。输出：无。

    预期行为：规则一的优先级高于规则二——报错块哪怕在别的主题上也要原样保留。
    这是 `keep_form=err_big` 那一格留出时最容易被考到的一条。
    """
    assert sp.action_of("err", "big", "old", "other") == sp.KEEP


def test_cost_table() -> None:
    """输入：四种动作 × 两种大小。输出：无。预期行为：保留按原大小、截断 800、摘要 60、丢弃 0。"""
    assert sp.cost_of(sp.KEEP, "small") == sp.TOKENS["small"] == 180
    assert sp.cost_of(sp.KEEP, "big") == sp.TOKENS["big"] == 1600
    assert sp.cost_of(sp.TRIM, "big") == 800
    assert sp.cost_of(sp.DIGEST, "small") == 60
    assert sp.cost_of(sp.DROP, "big") == 0


def test_cost_rejects_unknown_action_and_size() -> None:
    """输入：非法动作 / 非法大小。输出：无。预期行为：都报人话。"""
    with pytest.raises(KeyError, match="没有动作"):
        sp.cost_of("nope", "small")
    with pytest.raises(KeyError, match="没有 size"):
        sp.cost_of(sp.KEEP, "nope")


def test_compact_is_strictly_over_the_budget() -> None:
    """输入：总账与预算。输出：无。预期行为：**严格超过**才算需要压缩（等于不算）。"""
    assert sp.compact_of(1025, 1024) is True
    assert sp.compact_of(1024, 1024) is False
    assert sp.compact_of(1, 1024) is False


def test_budgets_for_realizes_both_labels() -> None:
    """输入：几个总账。输出：无。

    预期行为：每个总账都能落两种标签——这正是「先定标签再挑预算」的前提；
    池子里全都在同一侧时报错而不是静默退回。
    """
    for total in (1040, 1100, 1840, 2460, 2520, 3260):
        over = sp.budgets_for(total, True)
        fits = sp.budgets_for(total, False)
        assert over and fits
        assert all(budget < total for budget in over)
        assert all(budget >= total for budget in fits)
    with pytest.raises(ValueError, match="落不下标签"):
        sp.budgets_for(max(sp.BUDGETS) + 1, False)


# ---------- 排布与块构造 ----------


def test_order_is_a_latin_square() -> None:
    """输入：五种排布。输出：无。

    预期行为：每个槽在五个位置上都恰好出现一次——`spot` / `top` 的答案位置分布均匀的
    唯一来源。改了 `ORDERS` 这条会先红。
    """
    for slot in sp.SLOTS:
        positions = [sp.order_of(order).index(slot) for order in sp.ORDERS]
        assert sorted(positions) == [0, 1, 2, 3, 4]
    for order in sp.ORDERS:
        assert sorted(sp.order_of(order)) == sorted(sp.SLOTS)


def test_order_rejects_an_unknown_name() -> None:
    """输入：不存在的排布名。输出：无。预期行为：报人话并列出可用排布。"""
    with pytest.raises(KeyError, match="没有排布"):
        sp.order_of("p9")


def test_other_kind_always_differs() -> None:
    """输入：三个非报错类型。输出：无。预期行为：给回来的一定是**另一个**非报错类型。"""
    for kind in ("ask", "read", "run"):
        other = sp._other_kind(kind)
        assert other in ("ask", "read", "run") and other != kind
    with pytest.raises(KeyError, match="没有非报错类型"):
        sp._other_kind("err")


def test_every_session_has_five_distinct_blocks() -> None:
    """输入：整张 case 矩阵。输出：无。

    预期行为：五个块的 `类型/大小/年龄/主题` 四元组两两不同——撞了就会渲染出两行
    一模一样的正文（正文模板正是按这个四元组取的）。
    """
    cases = sp.build_cases(SessionConfig())
    assert cases
    for case in cases:
        blocks = sp.blocks_of(case)
        assert len(blocks) == len(sp.SLOTS) == 5
        tuples = [block.describe() for block in blocks]
        assert len(set(tuples)) == 5, (case.describe(), tuples)


def test_every_session_carries_one_of_each_action_plus_the_repeat() -> None:
    """输入：整张 case 矩阵。输出：无。

    预期行为：动作的多重集恰好是「四个基本动作各一个 + 重复角色的动作」——`action` 那一族
    四档各 1/4 就是从这里来的。
    """
    for case in sp.build_cases(SessionConfig()):
        actions = Counter(block.action() for block in sp.blocks_of(case))
        assert actions == Counter((*sp.BASE_ROLES, case.extra)), case.describe()


def test_block_helpers_describe_and_dict() -> None:
    """输入：一个块。输出：无。预期行为：`describe` / `as_dict` 与字段一致。"""
    block = BlockCase("read", "big", "recent", "same")
    assert block.describe() == "read/big/recent/same"
    assert block.as_dict() == {
        "kind": "read",
        "size": "big",
        "age": "recent",
        "topic": "same",
    }
    assert block.action() == sp.TRIM and block.cost() == 800


def test_unknown_keep_form_and_drop_form_and_extra_report() -> None:
    """输入：三个非法的构造取值。输出：无。预期行为：都报人话（不静默落到某个分支）。"""
    with pytest.raises(KeyError):
        sp.blocks_of(SessionCase("nope", "off_small", "read_run", "keep", "p0", 0))
    with pytest.raises(KeyError):
        sp.blocks_of(SessionCase("err_small", "nope", "read_run", "keep", "p0", 0))
    with pytest.raises(KeyError, match="没有重复动作"):
        sp.blocks_of(SessionCase("err_small", "off_small", "read_run", "nope", "p0", 0))


# ---------- 定位与极值 ----------


def test_drop_position_is_unique_unless_the_repeat_is_a_drop() -> None:
    """输入：`extra=drop` 与其它三种重复角色。输出：无。

    预期行为：重复角色也是丢弃块时有两块被丢 → 没有唯一解 → None（`spot` 的两阶段依赖）。
    """
    assert sp.drop_position(sp.blocks_of(SessionCase("err_small", "off_small", "read_run", "drop", "p0", 0))) is None
    for extra in ("keep", "trim", "digest"):
        blocks = sp.blocks_of(SessionCase("err_small", "off_small", "read_run", extra, "p0", 0))
        assert sp.drop_position(blocks) == 4  # p0 的第 4 个位置就是 drop 槽


def test_top_position_is_none_when_the_two_trims_tie() -> None:
    """输入：重复角色是截断、且基本保留块是小块。输出：无。

    预期行为：两块都记 800 → 并列 → None（`top` 的两阶段依赖）；基本保留块是大块时
    最大值是那块 1600，唯一。
    """
    small = SessionCase("err_small", "off_small", "read_run", "trim", "p0", 0)
    assert sp.top_position(sp.blocks_of(small)) is None
    big = SessionCase("err_big", "off_small", "read_run", "trim", "p0", 0)
    assert sp.top_position(sp.blocks_of(big)) == 1


def test_top_position_points_at_the_big_keep() -> None:
    """输入：重复角色是保留（两块保留块大小故意不同）。输出：无。

    预期行为：最大值一定在较大的那一块保留块上，且唯一。
    """
    blocks = sp.blocks_of(SessionCase("err_small", "off_small", "read_run", "keep", "p0", 0))
    costs = [block.cost() for block in blocks]
    assert sp.top_position(blocks) == costs.index(max(costs)) + 1
    assert sp.total_of(blocks) == sum(costs) == 2640


# ---------- case、指纹与值留出 ----------


def test_material_fingerprint_covers_every_dimension() -> None:
    """输入：六个只差一维的 case。输出：无。

    预期行为：指纹逐维不同——指纹是切分分桶的键，漏一维会让「只有该维不同」的 case
    被当成同一份素材，切分之间立刻撞车。
    """
    base = SessionCase("err_small", "off_small", "read_run", "keep", "p0", 0)
    variants = [
        SessionCase("fresh", "off_small", "read_run", "keep", "p0", 0),
        SessionCase("err_small", "stale", "read_run", "keep", "p0", 0),
        SessionCase("err_small", "off_small", "run_ask", "keep", "p0", 0),
        SessionCase("err_small", "off_small", "read_run", "drop", "p0", 0),
        SessionCase("err_small", "off_small", "read_run", "keep", "p1", 0),
        SessionCase("err_small", "off_small", "read_run", "keep", "p0", 1),
    ]
    prints = {sp.material_of(base)} | {sp.material_of(case) for case in variants}
    assert len(prints) == 7


def test_unseen_dims_names_the_keep_form_and_the_phrasing() -> None:
    """输入：各含一个留出取值的 case。输出：无。

    预期行为：只留出 `keep_form` 与 `phrasing` 两处——另外几维留出一个取值都会让某一族
    在训练侧退化成常量或把位置分布压歪。
    """
    assert SessionCase("err_big", "off_small", "read_run", "keep", "p0", 0).unseen_dims() == ("keep_form",)
    assert SessionCase("err_small", "off_small", "read_run", "keep", "p0", 1).unseen_dims() == ("phrasing",)
    assert SessionCase("err_big", "off_small", "read_run", "keep", "p0", 1).unseen_dims() == (
        "keep_form",
        "phrasing",
    )
    assert SessionCase("err_small", "off_small", "read_run", "keep", "p0", 0).unseen_dims() == ()


def test_case_as_dict_and_describe() -> None:
    """输入：一个 case。输出：无。预期行为：六个维度全摊平、描述与维度一致。"""
    case = SessionCase("fresh", "stale", "ask_read", "digest", "p2", 1)
    assert case.as_dict() == {
        "keep_form": "fresh",
        "drop_form": "stale",
        "kind_set": "ask_read",
        "extra": "digest",
        "order": "p2",
        "phrasing": 1,
    }
    assert case.describe() == "fresh/stale/ask_read/digest/p2/1"


def test_build_cases_ood_only_keeps_unseen_combinations() -> None:
    """输入：训练侧配置与 OOD 配置。输出：无。

    预期行为：训练侧一条留出取值都没有，OOD 侧每条都至少含一个——两侧按构造不重叠。
    """
    seen = sp.build_cases(SessionConfig(keep_forms=("err_small", "fresh"), phrasings=(0,)))
    assert seen and not any(case.unseen_dims() for case in seen)
    ood = sp.build_cases(SessionConfig(ood_only=True))
    assert ood and all(case.unseen_dims() for case in ood)
    assert len(seen) + len(ood) == len(sp.build_cases(SessionConfig()))


@pytest.mark.parametrize(
    "field",
    ["keep_forms", "drop_forms", "kind_sets", "extras", "orders", "phrasings"],
)
def test_config_rejects_empty_pools(field: str) -> None:
    """输入：某一维取空表。输出：无。预期行为：报人话并点名那个字段。"""
    with pytest.raises(ValueError, match=field):
        SessionConfig(**{field: ()})


# ---------- 规格 ----------


def test_session_id_is_deterministic_and_split_free() -> None:
    """输入：同一 case 的两次取样、不同切分。输出：无。

    预期行为：会话号由素材指纹派生——同一 case 必得同一号，且**不随切分变化**
    （否则切分信息就被写进了 state）。句面不同则号码不同（两套问法要是两条素材）。
    """
    case = SessionCase("err_small", "off_small", "read_run", "keep", "p0", 0)
    first = sp.session_id_of(case)
    assert first.startswith(sp.SESSION_PREFIX) and first == sp.session_id_of(case)
    assert first != sp.session_id_of(SessionCase("err_small", "off_small", "read_run", "keep", "p0", 1))
    assert sp.session_id_of(case) != sp.session_id_of(SessionCase("err_small", "off_small", "read_run", "keep", "p1", 0))


def test_sample_spec_is_deterministic_for_a_seed() -> None:
    """输入：同 case、同种子。输出：无。预期行为：块与判断完全一致（只有预算随种子变）。"""
    case = SessionCase("err_small", "off_small", "read_run", "keep", "p2", 0)
    first = sp.sample_spec(case, "train", 0, 0, random.Random(11))
    second = sp.sample_spec(case, "train", 0, 0, random.Random(11))
    assert first == second
    assert first.budget in sp.BUDGETS
    assert first.total == sp.total_of(first.blocks)
    assert first.compact is sp.compact_of(first.total, first.budget)
    assert first.split == "train" and first.seq == 0 and first.phrasing == 0


def test_spec_to_dict_is_json_ready() -> None:
    """输入：一条规格。输出：无。预期行为：摘要把块、预算与四个判断都摊平。"""
    spec = sp.sample_spec(
        SessionCase("err_small", "off_small", "read_run", "keep", "p0", 0),
        "train",
        3,
        0,
        random.Random(5),
    )
    payload = sp.spec_to_dict(spec)
    assert payload["case"] == "err_small/off_small/read_run/keep/p0/0"
    assert payload["seq"] == 3
    assert len(payload["blocks"]) == 5
    assert payload["unseen_dims"] == []
    assert set(payload) == {
        "case",
        "unseen_dims",
        "split",
        "seq",
        "session_id",
        "blocks",
        "budget",
        "total",
        "compact",
        "drop_position",
        "top_position",
    }
