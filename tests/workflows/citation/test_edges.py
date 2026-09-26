"""边界与错误路径：坏旋钮、坏切分、坏产物都要**报人话**，不许静默。

来由：这些分支在正常生成路径上永远走不到，但它们是「不崩、不泄密、报错人话」这条纪律的
落点——真出问题时，报人话的那行代码就是唯一的线索。另有一类是**回归**：写这些测试之前
实测踩到过「编造 × 截断」这一格渲染层拿到空来源句直接炸。
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from jc import assemble, jsonio
from jc import workflow as workflow_mod
from workflows.citation import judges, locales, parse, questions, render
from workflows.citation import spec as sp
from workflows.citation import workflow as citation_workflow


def _case(**overrides: Any) -> sp.CitationCase:
    """输入：维度覆盖项。输出：一个 case（默认「鉴权 / 通过 / 照抄 / 三节 / 中间」）。"""
    fields: dict[str, Any] = {
        "topic": "auth",
        "status": "verified",
        "quote_form": "verbatim",
        "sections": 3,
        "noise": "plain",
        "claim_style": 1,
        "where": "middle",
        "doc_style": "bracketed",
    }
    fields.update(overrides)
    return sp.CitationCase(**fields)


def _row(case: sp.CitationCase, seed: int = 1, locale: str = "zh") -> tuple[dict[str, Any], sp.CitationSpec]:
    """输入：case、种子、语言。输出：(契约样本行, 真值规格)。预期行为：走真的组装器。"""
    loc = locales.load(locale)
    rng = random.Random(seed)
    spec = sp.sample_spec(rng, case, "train", 0, 0)
    labeled = questions.build(spec, loc, spec.phrasing, rng)
    row = assemble.assemble_sample(
        "citation-train-0000000", render.build(spec, loc), labeled, rng
    )
    return row, spec


# ---------- 回归：曾经炸过的那一格 ----------


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_the_fabricated_truncated_cell_renders_and_is_really_absent(lang: str) -> None:
    """输入：全部「编造 × 截断」的 case。输出：无。

    预期行为：渲染得出来（来源句不再是 None——之前这一格直接抛「要渲染的句子是空的」），
    而且引文是**文档里某一句的前半句**、本身不等于任何整句，所以结论仍是编造。
    """
    loc = locales.load(lang)
    sentences = judges.tables()[0]
    checked = 0
    for case in sp.build_cases(
        sp.CitationConfig(
            topics=sp.TOPIC_NAMES,
            statuses=("fabricated",),
            quote_forms=("truncated",),
            section_counts=(2,),
            noises=("plain",),
            claim_styles=(1,),
            wheres=("first",),
            doc_styles=("bracketed",),
        )
    ):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        assert spec.case.quote_form == "truncated"
        text = render.build(spec, loc)
        quote = parse.parse_citation(text, parse.label_index(loc)).quote
        assert quote not in sentences, case.describe()
        assert any(item.startswith(quote) for item in sentences), case.describe()
        checked += 1
    assert checked == len(sp.TOPIC_NAMES)


# ---------- 旋钮与切分 ----------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("topics", ("auth", "made_up")),
        ("statuses", ("made_up",)),
        ("quote_forms", ("made_up",)),
        ("noises", ("made_up",)),
        ("wheres", ("made_up",)),
        ("doc_styles", ("made_up",)),
    ],
)
def test_config_rejects_an_unknown_value(field: str, value: tuple[str, ...]) -> None:
    """输入：每个维度各塞一个不存在的取值。输出：ValueError。预期行为：报人话并列出可用值。"""
    with pytest.raises(ValueError, match="不存在的取值"):
        sp.CitationConfig(**{field: value})


def test_config_rejects_an_empty_pool() -> None:
    """输入：空取值表。输出：ValueError。预期行为：空表在构造期就被拦下。"""
    with pytest.raises(ValueError, match="phrasings"):
        sp.CitationConfig(phrasings=())


def test_workflow_rejects_unknown_split_names() -> None:
    """输入：不存在的切分名。输出：KeyError。预期行为：三处入口都报人话。"""
    workflow = citation_workflow.CitationWorkflow()
    assert workflow.split_names() == ("train", "val", "test_iid", "test_ood")
    for call in (workflow.config_for, workflow.expected_values):
        with pytest.raises(KeyError, match="未知切分"):
            call("nope")


def test_workflow_declares_its_surface() -> None:
    """输入：无。输出：无。

    预期行为：语言表、维度标题、case 描述、维度取值都对外报得出来；`reparse` 用合并标签表
    做语言无关反解；未知判别器名当场报错。
    """
    workflow = citation_workflow.CitationWorkflow()
    assert workflow.available_locales() == ["en", "zh"]
    assert set(workflow.dimension_titles()) == set(_case().as_dict())
    case = _case(status="mislocated")
    assert workflow.case_label(case) == case.describe()
    assert workflow.stats_dimensions(case) == case.as_dict()
    loc = locales.load("zh")
    spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
    parsed = workflow.reparse(render.build(spec, loc))
    assert parsed == parse.parse_citation(render.build(spec, loc), parse.label_index(loc))
    assert [item.name for item in workflow.make_judges(("roundtrip", "label"), "zh")] == [
        "roundtrip",
        "label",
    ]
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")


def test_an_unknown_dimension_value_counts_as_unseen() -> None:
    """输入：一个取值不在词表里的 case。输出：无。

    预期行为：`unseen_dims` 把它当「没见过」——宁可把它划进 OOD，也不要漏掉。
    """
    stranger = sp.CitationCase("made_up", "verified", "verbatim", 3, "plain", 1, "first", "bracketed")
    assert stranger.unseen_dims() == ("topic",)


# ---------- 版式细节 ----------


def test_two_section_document_uses_the_right_marker_and_suffix() -> None:
    """输入：两节的规格。输出：无。预期行为：抬头写「共 2 个小节」、小节行是 `[1]` / `[2]`。"""
    loc = locales.load("zh")
    spec = sp.sample_spec(random.Random(0), _case(sections=2, where="last"), "train", 0, 0)
    lines = render.build(spec, loc).splitlines()
    assert lines[1].endswith(loc["meta"]["doc_suffix"].format(n=2))
    assert render.head_marker(1, "bracketed") == "[1]"
    assert render.head_marker(2, "dotted") == "2."


# ---------- 反解差异要按字段点名 ----------


@pytest.mark.parametrize(
    ("field", "needle"),
    [
        ("doc", "来源文档"),
        ("claim", "主张"),
        ("quote", "引文"),
        ("source", "声称出处"),
    ],
)
def test_roundtrip_diff_names_a_broken_field(field: str, needle: str) -> None:
    """输入：反解结果里改掉一个字段。输出：差异列表。预期行为：点名是哪个字段。"""
    loc = locales.load("zh")
    spec = sp.sample_spec(random.Random(0), _case(), "train", 0, 0)
    parsed = parse.parse_citation(render.build(spec, loc), parse.label_index(loc))
    broken = replace(parsed, **{field: "改坏的一行"})
    diff = judges.roundtrip_diff(spec, broken, loc)
    assert diff and any(needle in item for item in diff)


def test_roundtrip_diff_names_a_broken_section() -> None:
    """输入：反解结果里改掉标题、句子，或整节丢掉。输出：差异列表。

    预期行为：标题与句子分别点名；整节丢掉时既报「缺少小节」也报「小节数」。
    """
    loc = locales.load("zh")
    spec = sp.sample_spec(random.Random(0), _case(), "train", 0, 0)
    parsed = parse.parse_citation(render.build(spec, loc), parse.label_index(loc))
    first = parsed.sections[0]
    retitled = replace(parsed, sections=(replace(first, heading="改坏的标题"), *parsed.sections[1:]))
    assert any("标题" in item for item in judges.roundtrip_diff(spec, retitled, loc))
    sentence = replace(first, sentences=("改坏的句子",))
    rewritten = replace(parsed, sections=(sentence, *parsed.sections[1:]))
    assert any("句子" in item for item in judges.roundtrip_diff(spec, rewritten, loc))
    dropped = replace(parsed, sections=parsed.sections[1:])
    diff = judges.roundtrip_diff(spec, dropped, loc)
    assert any("小节数" in item for item in diff)
    assert any("缺少小节" in item for item in diff)


# ---------- 坏产物 ----------


def test_audit_reports_a_row_that_cannot_be_parsed(tmp_path: Path) -> None:
    """输入：一段不是这份契约的 state。输出：缺失项。预期行为：说「反解不出来」，不静默算过。"""
    jsonio.write_jsonl(
        tmp_path / "train.jsonl",
        [{"id": "citation-train-0000000", "state": "随便一段话", "questions": {}, "answers": {}}],
    )
    report, missing = workflow_mod.load("citation").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 1
    assert any("反解不出来" in item for item in missing)


def test_audit_reports_a_claimed_section_outside_the_document(tmp_path: Path) -> None:
    """输入：声称出处指向不存在的小节。输出：缺失项。预期行为：反解报错被审计接住。"""
    row, spec = _row(_case())
    lines = row["state"].splitlines()
    lines[-1] = lines[-1].replace(str(spec.claimed), "99")
    row["state"] = "\n".join(lines)
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    _, missing = workflow_mod.load("citation").audit(tmp_path, ("train",), "zh")
    assert any("反解不出来" in item for item in missing)


def test_recompute_rejects_a_missing_claimed_section() -> None:
    """输入：声称出处指向不存在的小节。输出：异常。预期行为：报人话，不拿别的小节顶上。"""
    loc = locales.load("zh")
    spec = sp.sample_spec(random.Random(0), _case(), "train", 0, 0)
    parsed = parse.parse_citation(render.build(spec, loc), parse.label_index(loc))
    with pytest.raises(ValueError, match="不存在"):
        judges.recompute(replace(parsed, source="小节 99"))


def test_audit_counts_an_unknown_family_and_a_dead_option(tmp_path: Path) -> None:
    """输入：一条多出未知题族、且选项表里塞了死选项的行。输出：报告。

    预期行为：未知题族被点名；死选项不影响重算（答案仍由文本算），但报告里能看出题数变多。
    """
    row, _spec = _row(_case(status="contradicted"))
    row["questions"]["made_up"] = {"type": "noul", "instructions": "x", "criteria": {"true": "1", "false": "2"}}
    row["answers"]["made_up"] = {"type": "noul", "noul": True}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = workflow_mod.load("citation").audit(tmp_path, ("train",), "zh")
    assert report["train"]["unknown_families"] == {"made_up": 1}
    assert any("未知题族" in item for item in missing)
    assert not [item for item in missing if "重算" in item]
