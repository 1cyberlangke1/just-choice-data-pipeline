"""边界与错误路径：坏切分、坏判别器名、坏产物、越界取值、构造式约束。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from pathlib import Path

import pytest

from jc import assemble, jsonio
from jc.state_parse import read_after_label
from tests.workflows.cachereuse.conftest import tiny_config
from workflows.cachereuse import locales, parse, render
from workflows.cachereuse import spec as sp
from workflows.cachereuse import workflow as cachereuse_workflow

LOC = locales.load("zh")
WORKFLOW = cachereuse_workflow.CacheReuseWorkflow()


def build_row(config: sp.ReuseConfig, case: sp.ReuseCase, seq: int) -> dict[str, JsonValue]:
    """输入：配置、case、序号。输出：契约样本。

    预期行为：走与管线同一条路（计划 → 规格 → 渲染 → 出题 → 组装），不手写样本。
    """
    plan = WORKFLOW.plan_sample(random.Random(seq), config, case, seq)
    spec = WORKFLOW.sample(random.Random(seq), plan)
    state = WORKFLOW.render(spec, random.Random(seq), "zh")
    labeled = WORKFLOW.ask(spec, random.Random(seq), "zh")
    return assemble.assemble_sample(f"c-{seq:07d}", state, labeled, random.Random(seq))


def test_workflow_protocol_surface() -> None:
    """输入：无。输出：无。预期行为：协议方法齐备、声明正确。"""
    assert WORKFLOW.name == "cachereuse"
    assert WORKFLOW.version
    assert WORKFLOW.needs == frozenset()
    assert WORKFLOW.judges == ("roundtrip", "label")
    assert WORKFLOW.kinds == frozenset({"choice", "noul", "score"})
    assert WORKFLOW.split_names() == ("train", "val", "test_iid", "test_ood")
    assert WORKFLOW.available_locales() == ["en", "zh"]
    assert WORKFLOW.dimension_titles() == dict(cachereuse_workflow.DIMENSION_TITLES)


def test_config_for_swaps_the_split_name() -> None:
    """输入：三个同分布切分。输出：无。

    预期行为：每次返回的配置里 `split` 都是调用方要的那一档——`ID_CONFIG` 是单例，
    直接返回它会让三档切出一模一样的数据。
    """
    for split in ("train", "val", "test_iid"):
        assert WORKFLOW.config_for(split).split == split
    assert WORKFLOW.config_for("test_ood").ood_only is True


def test_language_texts_is_empty() -> None:
    """输入：一份产物。输出：无。

    预期行为：报**空表**——state 全由代码按语言模板拼，报正文会把「以代码为主的 state」
    喂进语言判据，把好数据当坏数据（范式 §9.32）。
    """
    config = tiny_config("train")
    row = build_row(config, WORKFLOW.build_cases(config)[0], 0)
    assert WORKFLOW.language_texts(row) == {}


def test_case_helpers_expose_the_dimensions() -> None:
    """输入：一个 case。输出：无。预期行为：描述与维度表都摊平了五个维度。"""
    case = sp.ReuseCase(5, "half", "target", sp.SYNONYM, "kiosk")
    assert WORKFLOW.case_label(case) == case.describe()
    assert WORKFLOW.stats_dimensions(case) == case.as_dict()
    assert set(case.as_dict()) == {"length", "band", "diff_kind", "spelling", "flavor"}


@pytest.mark.parametrize("length", sp.LENGTHS)
def test_band_prefix_orders_the_four_bands(length: int) -> None:
    """输入：三种步数。输出：无。

    预期行为：四档的可复用步数**严格递减**（most > half > few > none）——档位名就是按这个
    顺序排的，顺序错了整族的答案都会跟着错位。
    """
    prefixes = [sp.band_prefix(band, length) for band in sp.BANDS]
    assert prefixes == sorted(prefixes, reverse=True)
    assert len(set(prefixes)) == len(prefixes)


def test_case_derived_quantities_are_consistent() -> None:
    """输入：全空间 case。输出：无。

    预期行为：第一个不同的步号 = 可复用步数 + 1；等价总步数 = 可复用步数 + 分叉后对上的步数。
    **两者不是一回事**——这就是这一族要训的「前缀 ≠ 计数」。
    """
    for case in sp.build_cases(sp.ReuseConfig(split="train")):
        assert case.first_diff == case.prefix + 1, case.describe()
        assert case.equivalent_total == case.prefix + case.tail, case.describe()
        assert case.reuse_half is (case.prefix * 2 >= case.length), case.describe()
        if case.tail:
            assert case.equivalent_total > case.prefix, case.describe()


def test_traces_have_the_planned_shape() -> None:
    """输入：全空间 case。输出：无。

    预期行为：前 `prefix` 步逐字等价、紧接着 `tail` 步又等价、分叉那一步**只差一处**
    （动作或元素），其余步全不同——这是「前缀可复用」这条规则在数据上的唯一形态。
    """
    for case in sp.build_cases(sp.ReuseConfig(split="train")):
        old, new = sp.traces_of(case)
        assert len(old) == len(new) == case.length, case.describe()
        for index, (left, right) in enumerate(zip(old, new, strict=True)):
            equivalent = index < case.prefix or case.prefix < index <= case.prefix + case.tail
            assert (left == right) is equivalent, f"{case.describe()} 第 {index + 1} 步"
        left, right = old[case.prefix], new[case.prefix]
        if case.diff_kind == "action":
            assert left[1] == right[1] and left[0] != right[0], case.describe()
        else:
            assert left[0] == right[0] and left[1] != right[1], case.describe()


def test_sample_spec_ignores_the_rng() -> None:
    """输入：同一 case、两个不同 RNG。输出：无。

    预期行为：规格逐字段一致——「同素材同题必须同答」靠的就是这一步不掷骰子。
    """
    case = sp.ReuseCase(5, "few", "action", sp.VERBATIM, "web")
    first = sp.sample_spec(random.Random(0), case, "train", 0, 0)
    second = sp.sample_spec(random.Random(999), case, "train", 0, 0)
    assert first == second


def test_spelling_index_rejects_unknown_spelling() -> None:
    """输入：表外的写法名。输出：无。

    预期行为：报错——悄悄退回 0 会把「同义写法」这一类样本渲染成逐字相同，整条维白设。
    """
    with pytest.raises(ValueError, match="未知写法"):
        render.spelling_index("homophone")


def test_header_prefers_the_longest_label() -> None:
    """输入：互为前缀的两个抬头标签。输出：无。

    预期行为：最长前缀优先——短的先命中会把「请求号」读成「请求」，值里平白多出「号」。
    """
    tables = parse.Tables(request=("请求", "请求号"), sources=("来源",), sections={})
    assert read_after_label("请求号 C-1", tables.request, "请求号") == "C-1"
    with pytest.raises(ValueError, match="抬头认不出来"):
        read_after_label("别的抬头 C-1", tables.request, "请求号")


def test_parse_rejects_a_state_that_is_too_short() -> None:
    """输入：只有两行的 state。输出：无。预期行为：报「太短」，不抛 IndexError。"""
    with pytest.raises(ValueError, match="太短"):
        parse.parse_state("缓存复用请求 C-1\n缓存来源：网页版\n", parse.merged_tables())


def test_parse_rejects_a_step_before_any_section() -> None:
    """输入：区块抬头之前就出现步骤行。输出：无。预期行为：报出来（渲染版式写歪了）。"""
    tables = parse.merged_tables()
    state = "缓存复用请求 C-1\n缓存来源：网页版\n  1. 打开「搜索框」\n[缓存轨迹]\n  1. 打开「搜索框」\n[当前轨迹]\n  1. 打开「搜索框」\n"
    with pytest.raises(ValueError, match="区块抬头之前"):
        parse.parse_state(state, tables)


def test_parse_rejects_an_unrecognized_source_line() -> None:
    """输入：来源那一行认不出来 / 没写值。输出：无。预期行为：两种都当场报错。"""
    tables = parse.merged_tables()
    head = "缓存复用请求 C-1\n"
    tail = "[缓存轨迹]\n  1. 打开「搜索框」\n[当前轨迹]\n  1. 打开「搜索框」\n"
    with pytest.raises(ValueError, match="来源那一行认不出来"):
        parse.parse_state(head + "出处：网页版\n" + tail, tables)
    with pytest.raises(ValueError, match="没写值"):
        parse.parse_state(head + "缓存来源：\n" + tail, tables)


def test_unquote_normalizes_both_quote_styles() -> None:
    """输入：两种引号的同一行。输出：无。预期行为：都统一成 `"`，正则只需要一种写法。"""
    assert parse._unquote("点击「搜索框」") == '点击"搜索框"'
    assert parse._unquote('Click "Search box"') == 'Click "Search box"'


def test_parse_tolerates_blank_lines_and_indented_step_lines() -> None:
    """输入：区块之间掺了空行、区块内的行加了首尾空格的 state。输出：无。

    预期行为：空行跳过、行首行尾空格忽略——state 是人读的文本，排版噪声不该影响反解。
    只对**区块内的行**加噪声：抬头那两行是版式的一部分，渲染器永远顶格写。
    """
    config = tiny_config("train")
    case = WORKFLOW.build_cases(config)[0]
    plan = WORKFLOW.plan_sample(random.Random(0), config, case, 0)
    spec = WORKFLOW.sample(random.Random(0), plan)
    lines = WORKFLOW.render(spec, random.Random(0), "zh").splitlines()
    noisy = list(lines[:2])
    for line in lines[2:]:
        noisy.append("")
        noisy.append(f"   {line}   ")
    parsed = parse.parse_state("\n".join(noisy) + "\n", parse.tables_of(LOC))
    assert parsed.request_id == spec.query_id
    assert len(parsed.cached) == len(parsed.current) == case.length


def test_reparse_through_the_protocol_is_language_agnostic() -> None:
    """输入：两种语言渲染的 state。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，靠合并标签表也能反解出两条轨迹——它是
    「产物审计不依赖语言」的那条路。
    """
    config = tiny_config("train")
    case = WORKFLOW.build_cases(config)[0]
    plan = WORKFLOW.plan_sample(random.Random(0), config, case, 0)
    spec = WORKFLOW.sample(random.Random(0), plan)
    for lang in ("zh", "en"):
        parsed = WORKFLOW.reparse(WORKFLOW.render(spec, random.Random(0), lang))
        assert parsed.request_id == spec.query_id
        assert len(parsed.cached) == len(parsed.current) == case.length


def test_audit_flags_tampered_answers_and_unknown_families(tmp_path: Path) -> None:
    """输入：被改坏的产物。输出：无。

    预期行为：审计逐条报出来——答案与 state 重算不一致、编号越界、未知题族、题族一条没出。
    """
    config = tiny_config("train")
    rows = [
        build_row(config, case, index)
        for index, case in enumerate(WORKFLOW.build_cases(config))
    ]
    rows[0]["answers"]["reuse_band"]["score"] = (
        rows[0]["answers"]["reuse_band"]["score"] + 1
    ) % len(sp.BANDS)
    rows[1]["answers"]["diverge_at"]["choice"] = 99
    rows[2]["questions"]["mystery"] = dict(rows[2]["questions"]["reuse_band"])
    rows[2]["answers"]["mystery"] = dict(rows[2]["answers"]["reuse_band"])
    for row in rows:
        row["questions"].pop("reuse_half")
        row["answers"].pop("reuse_half")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)
    report, missing = WORKFLOW.audit(run_dir, ("train",), "zh")
    assert report["train"]["samples"] == len(rows)
    assert report["train"]["label_mismatch"] >= 1
    assert report["train"]["index_bad"] >= 1
    assert report["train"]["unknown_families"]["mystery"] >= 1
    assert any("不一致" in item for item in missing)
    assert any("编号异常" in item for item in missing)
    assert any("未知题族" in item for item in missing)
    assert any("reuse_half 一条都没出" in item for item in missing)


def test_audit_counts_the_four_answer_shapes(tmp_path: Path) -> None:
    """输入：一份干净产物。输出：无。

    预期行为：审计顺手统计档位 / 位置 / 分叉原因 / 是否过半四张分布表（报告要用），
    且题族齐全、零缺失。
    """
    config = tiny_config("train")
    rows = [
        build_row(config, case, index)
        for index, case in enumerate(WORKFLOW.build_cases(config))
    ]
    run_dir = tmp_path / "run3"
    run_dir.mkdir()
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)
    report, missing = WORKFLOW.audit(run_dir, ("train",), "zh")
    assert missing == []
    counts = report["train"]
    for key in ("band", "diverge", "diff_kind", "reuse_half"):
        assert sum(counts[key].values()) == len(rows), key
    assert set(counts["types"]) == {"choice", "noul", "score"}
    assert counts["index_bad"] == 0
