"""边界与接线：声明项、切片、审计的坏产物路径、以及「不碰后端」这条纪律。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from pathlib import Path


import pytest

from jc import assemble, jsonio
from jc import workflow as workflow_mod
from tests.workflows.functions.conftest import tiny_config
from workflows.functions import locales, parse, questions, render, spec as sp
from workflows.functions import workflow as fn_workflow
from workflows.functions.workflow import FunctionsWorkflow

WF = FunctionsWorkflow()


def test_declaration_matches_the_contract() -> None:
    """输入：workflow 的声明项。输出：无。

    预期行为：题型是 choice / noul，`needs` 为空（纯构造式，管线不初始化任何后端），
    默认判别器是 roundtrip / label。
    """
    assert WF.name == "functions"
    assert WF.version == "0.1.0"
    assert WF.kinds == frozenset({"choice", "noul"})
    assert WF.needs == frozenset()
    assert WF.judges == ("roundtrip", "label")


def test_bind_backend_refuses_to_be_wired() -> None:
    """输入：一个生成后端。输出：无。

    预期行为：纯构造式 workflow 根本不会被注入后端；收到就说明接线错了，直接报错。
    """
    with pytest.raises(AssertionError, match="纯构造式"):
        WF.bind_backend(object())


def test_workflow_is_registered_under_its_name() -> None:
    """输入：注册表。输出：无。预期行为：`load("functions")` 拿得到同一个类。"""
    assert "functions" in workflow_mod.registered()
    assert isinstance(workflow_mod.load("functions"), FunctionsWorkflow)


def test_split_names_and_unknown_split() -> None:
    """输入：切分名。输出：无。预期行为：四档齐全；未知切分报 KeyError 并列出可用档。"""
    assert WF.split_names() == ("train", "val", "test_iid", "test_ood")
    with pytest.raises(KeyError, match="未知切分"):
        WF.config_for("nope")


def test_config_for_swaps_the_split_on_the_shared_id_config() -> None:
    """输入：三个同分布切分。输出：无。

    预期行为：`ID_CONFIG` 是单例，返回前必须把 `split` 换成调用方要的那一档——
    否则 train / val / test_iid 会按同一档切片，切出三份一模一样的数据。
    """
    splits = {split: WF.config_for(split) for split in ("train", "val", "test_iid")}
    assert {config.split for config in splits.values()} == {"train", "val", "test_iid"}
    assert len({config for config in splits.values()}) == 3


def test_bucket_of_is_deterministic_and_in_range() -> None:
    """输入：一批素材指纹。输出：无。

    预期行为：同一指纹必得同一桶、桶号落在 0~99；不用内置 `hash()`（它对字符串每进程随机化）。
    """
    for material in ("a|b|c|d|0", "x|y|z|w|1", ""):
        first = fn_workflow.bucket_of(material)
        assert first == fn_workflow.bucket_of(material)
        assert 0 <= first < 100


def test_slice_keeps_the_matrix_whole() -> None:
    """输入：训练侧的 case 矩阵。输出：无。

    预期行为：三档切片两两不相交、并集就是全部 case、每档都非空。
    """
    cases = sp.build_cases(sp.CallConfig())
    parts = {split: set(fn_workflow._slice(cases, split)) for split in fn_workflow.SPLIT_BUCKETS}
    union = set().union(*parts.values())
    assert union == set(cases)
    assert all(part for part in parts.values())
    assert not parts["train"] & parts["val"]
    assert not parts["train"] & parts["test_iid"]
    assert not parts["val"] & parts["test_iid"]


def test_plan_sample_carries_the_case_and_the_split() -> None:
    """输入：一次 `plan_sample`。输出：无。预期行为：计划里带上切分名与 case，不掷骰子。"""
    case = sp.CallCase("plot_price", "terse", "1d", "ticker", 0)
    plan = WF.plan_sample(random.Random(1), tiny_config("val"), case, 3)
    assert plan.name == "val"
    assert plan.params == {"case": case, "seq": 3}


def test_sample_and_render_ignore_the_rng() -> None:
    """输入：同一计划、两个不同 RNG。输出：无。

    预期行为：规格与文本都不随 RNG 变（计划由素材指纹派生）——「两次生成逐字节一致」
    这条就是靠它成立的。
    """
    case = sp.CallCase("plot_price", "chatty", "1w", "ticker", 0)
    split = WF.plan_sample(random.Random(1), tiny_config("train"), case, 0)
    first = WF.sample(random.Random(1), split)
    second = WF.sample(random.Random(99), split)
    assert first == second
    assert WF.render(first, random.Random(1), "zh") == WF.render(second, random.Random(99), "zh")


def test_ask_ignores_the_rng() -> None:
    """输入：同一规格、两个不同 RNG。输出：无。预期行为：题目与答案完全一致。"""
    spec = sp.sample_spec(sp.CallCase("top_movers", "chatty", "1w", "ticker", 0), "train", 0)
    first = WF.ask(spec, random.Random(1), "zh")
    second = WF.ask(spec, random.Random(99), "zh")
    assert [(item.family, item.answer) for item in first] == [
        (item.family, item.answer) for item in second
    ]


def test_reparse_rejects_a_non_string_state() -> None:
    """输入：不是字符串的 state。输出：无。预期行为：TypeError（点明收到什么类型）。"""
    with pytest.raises(TypeError, match="字符串"):
        WF.reparse({"state": "x"})
    with pytest.raises(TypeError, match="字符串"):
        WF.reparse(None)


def test_reparse_accepts_an_ignored_case_argument() -> None:
    """输入：带上 `case=` 的反解。输出：无。预期行为：`case` 只影响措辞，读法不用它。"""
    spec = sp.sample_spec(sp.CallCase("plot_price", "terse", "1d", "ticker", 0), "train", 0)
    state = WF.render(spec, random.Random(0), "zh")
    assert WF.reparse(state, case=spec.case).tool == "plot_price"


def test_case_label_and_dimensions() -> None:
    """输入：一个 case。输出：无。预期行为：描述可读、五个维度标题都在。"""
    case = sp.CallCase("compare_returns", "chatty", "3mo", "name", 1)
    assert WF.case_label(case) == case.describe()
    assert set(WF.stats_dimensions(case)) == {"tool", "voice", "window", "alias", "phrasing"}
    assert set(WF.dimension_titles()) == {"tool", "voice", "window", "alias", "phrasing"}


def test_config_for_ood_is_the_full_table() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：取值表全开、`ood_only` 打开、切分名对。"""
    config = WF.config_for("test_ood")
    assert config.split == "test_ood"
    assert config.ood_only is True
    assert set(config.tools) == set(sp.TOOL_NAMES)
    assert set(config.windows) == set(sp.WINDOWS)


def _sample_row() -> dict[str, JsonValue]:
    """输入：无。输出：一条真样本（中文）。预期行为：判别器与审计都判它合格。"""
    case = sp.CallCase("plot_price", "chatty", "1w", "ticker", 0)
    loc = locales.load("zh")
    spec = sp.sample_spec(case, "train", 0)
    state = render.render_state(spec, loc)
    items = questions.build(spec.plan, loc, spec.plan.phrasing, random.Random(0))
    return assemble.assemble_sample("s1", state, items, random.Random(1))


def test_audit_flags_an_out_of_range_choice_index(tmp_path: Path) -> None:
    """输入：编号越界的产物。输出：无。预期行为：审计记一笔「编号越界」。"""
    row = _sample_row()
    qid = next(q for q in row["questions"] if row["questions"][q]["type"] == "choice")
    row["answers"][qid]["choice"] = 99
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


def test_audit_flags_an_unknown_family(tmp_path: Path) -> None:
    """输入：带未知题族的产物。输出：无。预期行为：审计点名那个题族。"""
    row = _sample_row()
    row["questions"]["bogus"] = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    row["answers"]["bogus"] = {"type": "noul", "noul": True}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["unknown_families"]["bogus"] == 1
    assert any("未知题族" in item for item in missing)


def test_audit_flags_a_state_it_cannot_read(tmp_path: Path) -> None:
    """输入：state 读不出来的产物。输出：无。预期行为：记一条「反解不出来」，不静默跳过。"""
    row = _sample_row()
    row["state"] = "今天天气不错。"
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("反解不出来" in item for item in missing)


def test_audit_only_requires_families_of_the_tools_it_saw(tmp_path: Path) -> None:
    """输入：只含 plot_price 的产物。输出：无。

    预期行为：审计**不**把 `top_movers` 那几族报成缺题——它们的缺席由 `tools` 计数如实记下，
    而「题族齐全」只对出现过的工具点名。
    """
    row = _sample_row()
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert dict(report["train"]["tools"]) == {"plot_price": 1}
    assert not any("top_movers" in item or "direction" in item for item in missing)
    assert not any("一条都没出" in item for item in missing), missing


def test_audit_flags_a_family_that_never_shows_up(tmp_path: Path) -> None:
    """输入：把某一族的题全删掉。输出：无。预期行为：审计记一笔「一条都没出」。"""
    row = _sample_row()
    del row["questions"]["stated_style"]
    del row["answers"]["stated_style"]
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["families"]["stated_style"] == 0
    assert any("stated_style" in item and "一条都没出" in item for item in missing)


def test_parse_read_request_defaults_alias_and_phrasing() -> None:
    """输入：一段请求。输出：无。

    预期行为：读法给 `alias="ticker"` / `phrasing=0`——它们只影响措辞，不影响答案，
    所以读不出来也不该乱猜。
    """
    loc = locales.load("zh")
    state = render.render_state(
        sp.sample_spec(sp.CallCase("plot_price", "chatty", "1w", "name", 1), "train", 0), loc
    )
    plan = parse.read_request(state, loc)
    assert plan.alias == "ticker"
    assert plan.phrasing == 0
