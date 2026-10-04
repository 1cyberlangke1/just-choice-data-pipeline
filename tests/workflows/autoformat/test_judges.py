"""结构恢复的判别器测试：三条不同的路（卫生、往返、反解）。"""

from __future__ import annotations

import random

import pytest

from jc import assemble
from jc.backend.types import AllProvidersDown, Draft, TransportError
from jc.jsontypes import JsonValue
from tests.workflows.autoformat import conftest as cf
from tests.workflows.autoformat.conftest import FakeBackend
from workflows.autoformat import features, judges, locales, parse, questions, render
from workflows.autoformat.features import KIND_CODE, KIND_HEADING, KIND_QUOTE
from workflows.autoformat.spec import DocSpec

ALL_CASES = features.build_cases(features.DocConfig())
LOC = locales.load("zh")
PLAIN = features.DocCase("build", 64, "plain", "na", "na")
CODE = features.DocCase("build", 64, "code", "na", "vcs")
MIXED = features.DocCase("build", 64, "mixed", "tip", "vcs")


def test_trace_reader_schema_pins_block_kinds() -> None:
    """输入：无。输出：无。

    预期行为：反解草稿的 `kinds` 只能取 `BLOCK_KINDS` 那六个名字——裸 `list[str]` 时 20B 读者
    交回过一串空串（`['', '', '']`），那一条就等于没判；写进 JSON schema 的 `enum` 才能让支持
    结构化输出的端点去约束生成。
    """
    items = judges.TraceDraft.model_json_schema()["properties"]["kinds"]["items"]
    assert items["enum"] == list(features.BLOCK_KINDS)
QUOTE = features.DocCase("build", 64, "quote", "na", "na")


def state(*bodies: str) -> str:
    """输入：各行正文（空串表示空行）。输出：契约形态的 state。预期行为：行号三位零填充。"""
    return "\n".join(
        f"{parse.line_id(index)}|" + (f" {body}" if body else "")
        for index, body in enumerate(bodies, start=1)
    )


def kinds_of(state_text: str, loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：state、语言字典。输出：块类型表。预期行为：只读文本，不碰构造侧的任何东西。"""
    return parse.parse_state(state_text, loc).kinds()


def sample_of(
    case: features.DocCase, lang: str = "zh", seq: int = 1
) -> tuple[DocSpec, dict[str, JsonValue]]:
    """输入：case、语言、序号。输出：(真值规格, 组装好的样本)。预期行为：忠实渲染，零 token。"""
    spec = cf.filled(case, lang)
    loc = locales.load(lang)
    state_text = render.render_state(spec.plan, spec.texts, loc)
    labeled = questions.build(spec, loc, spec.phrasing)
    sample = assemble.assemble_sample(
        f"autoformat-train-{seq:07d}", state_text, labeled, random.Random(seq)
    )
    return spec, sample


def first_question(sample: dict[str, JsonValue], family: str) -> str:
    """输入：样本、题族名。输出：该族第一道题的题号。"""
    for qid in sample["questions"]:
        if qid == family or str(qid).startswith(f"{family}_"):
            return str(qid)
    raise AssertionError(f"样本里没有 {family} 那一族")


def test_marker_leftover_finds_prefixes_and_infixes() -> None:
    """输入：几种带标记残留的 state。输出：无。

    预期行为：行首的井号 / 列表符号 / 引用符号、行内的加粗与代码围栏都算残留；空行不算。
    """
    assert judges._marker_leftover(state("", "正文", "")) == ""
    assert judges._marker_leftover(state("# 标题")) == "行首标记 '#'"
    assert judges._marker_leftover(state("- 列表项")) == "行首标记 '- '"
    assert judges._marker_leftover(state("> 引用")) == "行首标记 '> '"
    assert judges._marker_leftover(state("正文", "**加粗**")) == "行内标记 '**'"
    assert judges._marker_leftover(state("正文", "```")) == "行内标记 '```'"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_hygiene_accepts_every_faithful_sample(lang: str) -> None:
    """输入：两种语言 × 全空间 345 组。输出：无。

    预期行为：没有标记残留，各块的标记词都落在该落的地方。
    """
    judge = judges.HygieneJudge(lang)
    for case in ALL_CASES:
        _, sample = sample_of(case, lang)
        verdict = judge.judge(sample)
        assert verdict.ok, f"{lang} {case.describe()}: {verdict.reason}"


def test_hygiene_rejects_a_marker_leak() -> None:
    """输入：标题行前面多了一个井号。输出：无。预期行为：报残留，并点名是哪个标记。

    这条正是 `check_draft` **不**管的那一类：反解读出来的块类型仍然对得上，只有卫生判据看得见。
    """
    _, sample = sample_of(MIXED)
    leaked = dict(sample)
    leaked["state"] = str(sample["state"]).replace("L001| ", "L001| # ", 1)
    verdict = judges.HygieneJudge("zh").judge(leaked)
    assert not verdict.ok and not verdict.skipped
    assert "排版标记残留" in verdict.reason


@pytest.mark.parametrize(
    ("broken", "match"),
    [
        ({"state": 42}, "不是字符串"),
        ({"state": "随便一行没有行号的文本"}, "反解失败"),
        ({"state": state("platform and release leads", "monitoring-- and alerts")}, "署名前缀"),
        ({"state": state("platform and release leads", "monitoring Note: and alerts")}, "提示框关键词"),
        (
            {"state": state("Long prose that is long enough to be read as a paragraph. Note: here.")},
            "没以任何关键词开头",
        ),
    ],
)
def test_hygiene_rejects_broken_states(broken: dict[str, JsonValue], match: str) -> None:
    """输入：几种「标记词跑错块」的 state。输出：无。预期行为：都报出来，并点名是哪一块。

    这些都是**从文本本身**读出来的问题——正是「模型把标记词写进别的块」的落地形态。
    """
    verdict = judges.HygieneJudge("en").judge(broken)
    assert not verdict.ok and not verdict.skipped
    assert match in verdict.reason


def test_block_problems_flags_a_heading_that_ends_with_punctuation() -> None:
    """输入：手工拼一份「标题以标点结尾」的反解结果。输出：无。

    预期行为：报出来。解析器当前**不会**产出这种块（短行判据本身就排除了句末标点），但判别器是
    独立的一层：判据本身要立得住，不能只靠构造侧的自觉。
    """
    doc = parse.ParsedDoc(
        lines=(parse.ParsedLine("L001", "标题。"),),
        blocks=(parse.ParsedBlock(KIND_HEADING, ("L001",), "标题。"),),
    )
    problems = judges._block_problems(doc, LOC)
    assert any("以标点结尾" in item for item in problems)


def test_hygiene_allows_the_markers_inside_a_code_block() -> None:
    """输入：命令行块里带长选项与提示框关键词。输出：无。

    预期行为：通过——读法里「整段命令行 → 代码块」排在标记词判据之前，所以英文署名前缀 `--`
    与长选项同形、命令里出现 `Note:` 都不算残留（不豁免的话每个英文代码块都会被假判）。
    """
    sample = {"state": state("git commit -m 'Note: fix'", "git log --oneline -5")}
    verdict = judges.HygieneJudge("en").judge(sample)
    assert verdict.ok, verdict.reason
    assert kinds_of(str(sample["state"]), locales.load("en")) == (KIND_CODE,)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_roundtrip_accepts_every_faithful_sample(lang: str) -> None:
    """输入：两种语言 × 全空间 345 组。输出：无。预期行为：从文本反解后四个题族全部对上。"""
    judge = judges.RoundtripJudge(lang)
    for case in ALL_CASES:
        _, sample = sample_of(case, lang)
        verdict = judge.judge(sample)
        assert verdict.ok, f"{lang} {case.describe()}: {verdict.reason}"


def test_roundtrip_rejects_a_flipped_answer() -> None:
    """输入：把一道 `join` 题的答案说反。输出：无。预期行为：逐题比对时抓出来，并报出题号。"""
    _, sample = sample_of(MIXED)
    qid = first_question(sample, "join")
    tampered = dict(sample)
    answers = dict(sample["answers"])
    answers[qid] = {"type": "noul", "noul": not sample["answers"][qid]["noul"]}
    tampered["answers"] = answers
    verdict = judges.RoundtripJudge("zh").judge(tampered)
    assert not verdict.ok
    assert qid in verdict.reason


def test_roundtrip_rejects_a_state_whose_blocks_moved() -> None:
    """输入：把命令行块的一行改成不是命令。输出：无。

    预期行为：反解读出来的块类型与答案对不上——这条路走的是**独立实现**，所以渲染写歪也能抓到。
    """
    spec, sample = sample_of(CODE)
    tampered = dict(sample)
    tampered["state"] = str(sample["state"]).replace("hg --help", "whatever --help", 1)
    verdict = judges.RoundtripJudge("zh").judge(tampered)
    assert not verdict.ok
    assert "答案" in verdict.reason
    assert spec.plan.kinds() != kinds_of(str(tampered["state"]), LOC)


def test_recompute_matches_the_plan_labels() -> None:
    """输入：全空间 345 组。输出：无。预期行为：从文本重算的四族答案与计划的金标签逐项相同。"""
    for case in ALL_CASES:
        spec, sample = sample_of(case)
        expected = judges.recompute(render.parse_state(str(sample["state"])), LOC)
        labels = spec.labels
        assert expected["block_type"] == list(labels.kinds), case.describe()
        assert expected["callout_kind"] == list(labels.callout_kinds), case.describe()
        # 列表有序性那一族问的是**整份文档里那个列表**（一道题），金标签却是逐块的标记；
        # 计划的保证是「同一个列表的各项有序性相同」（见 test_features）。
        want = [labels.list_flags[0]] if labels.list_flags else []
        assert expected["list_ordered"] == want, case.describe()
        assert judges.compare(sample, expected) == [], case.describe()


def test_compare_reports_extra_missing_and_unknown_questions() -> None:
    """输入：三种被改坏的样本。输出：无。

    预期行为：多出来的题、少掉的题、认不出的题族各记一笔——不静默取第一个。
    """
    _, sample = sample_of(PLAIN)
    expected = judges.recompute(render.parse_state(str(sample["state"])), LOC)
    extra = dict(sample)
    extra["questions"] = {**sample["questions"], "join_99": {"type": "noul", "criteria": {}}}
    extra["answers"] = {**sample["answers"], "join_99": {"type": "noul", "noul": True}}
    assert any("多出来了" in item for item in judges.compare(extra, expected))
    missing = dict(sample)
    missing["questions"] = dict(list(sample["questions"].items())[:-1])
    assert any("少了" in item for item in judges.compare(missing, expected))
    unknown = dict(sample)
    unknown["questions"] = {**sample["questions"], "nope": {"type": "noul", "criteria": {}}}
    assert any("未知题族" in item for item in judges.compare(unknown, expected))


def test_compare_reports_a_broken_sample_shape() -> None:
    """输入：`questions` 不是对象的样本。输出：无。预期行为：报一行人话，不抛异常。"""
    assert judges.compare({"questions": 1, "answers": 1}, {}) == [
        "样本的 questions / answers 不是对象"
    ]


def test_option_names_and_answer_value_read_the_recorded_order() -> None:
    """输入：几道题与答案。输出：无。

    预期行为：choice 的答案编号是**产物里那个顺序**的下标（不是构造时的顺序）；越界或类型不对
    给 None（那是对不上，不是崩）；noul 只认布尔。
    """
    question: dict[str, JsonValue] = {"type": "choice", "criteria": {"b": "x", "a": "y"}}
    assert judges.answer_value(question, {"type": "choice", "choice": 0}) == "b"
    assert judges.answer_value(question, {"type": "choice", "choice": 5}) is None
    assert judges.answer_value(question, {"type": "choice", "choice": True}) is None
    assert judges.answer_value({"type": "noul"}, {"type": "noul", "noul": True}) is True
    assert judges.answer_value({"type": "noul"}, {"type": "noul", "noul": "yes"}) is None


def test_trace_accepts_the_truth_and_uses_the_trace_task_id() -> None:
    """输入：从不撒谎的假后端。输出：无。预期行为：通过，而且真的走的是反解那个任务 id。"""
    spec, sample = sample_of(MIXED)
    backend = FakeBackend()
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert backend.tasks == [spec.trace_task_id]


def test_trace_rejects_a_lying_reader() -> None:
    """输入：少报一块的反解。输出：无。预期行为：不通过，理由里列出两边。"""
    spec, sample = sample_of(MIXED)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "读出的块类型" in verdict.reason


def test_trace_marks_transport_errors_as_skipped() -> None:
    """输入：反解时端点故障。输出：无。预期行为：记成「没判成」，不算数据不合格。"""
    spec, sample = sample_of(MIXED)
    judge = judges.TraceJudge(FakeBackend(trace_raises=TransportError("boom")), "zh")
    verdict = judge.judge(sample, spec)
    assert verdict.skipped
    assert "端点故障" in verdict.reason


def test_trace_reraises_when_every_provider_is_down() -> None:
    """输入：全部 provider 都挂了。输出：无。预期行为：往上抛，让整批停下（不是这一条的错）。"""
    spec, sample = sample_of(MIXED)
    judge = judges.TraceJudge(FakeBackend(trace_raises=AllProvidersDown("all down")), "zh")
    with pytest.raises(AllProvidersDown):
        judge.judge(sample, spec)


def test_trace_skips_a_bad_spec_or_an_unreadable_draft() -> None:
    """输入：规格类型不对、以及一个答了不存在块类型的反解。输出：无。

    预期行为：两种都算「没判成」——那是反解格式没对齐，不是数据本身错。
    """
    _, sample = sample_of(PLAIN)
    assert judges.TraceJudge(FakeBackend(), "zh").judge(sample, object()).skipped
    verdict = judges.TraceJudge(_UnknownKindBackend(), "zh").judge(sample, cf.spec_for(PLAIN))
    assert verdict.skipped
    assert "认不出的块类型" in verdict.reason


def test_trace_prompt_carries_the_whole_state() -> None:
    """输入：一段 state。输出：无。预期行为：反解提示词把整段材料原样带上，且没有剩花括号。"""
    _, sample = sample_of(MIXED)
    prompt = judges.build_trace_prompt(str(sample["state"]), LOC)
    assert str(sample["state"]) in prompt
    # 题面里的 {steps} 会被顺序词表填掉，所以查前缀 + 每个顺序词都到齐（「列表有没有序」那条判据
    # 必须点名顺序词，读者才不用去猜「这几项看起来像不像有先后顺序」）。
    assert prompt.startswith(str(LOC["verify"]["ask"]).split("{")[0])
    for word in parse.seq_words(LOC):
        assert word in prompt
    assert "{" not in prompt and "}" not in prompt


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_trace_prompt_carries_the_splitting_rules(lang: str) -> None:
    """输入：一段 state、两种语言。输出：无。

    预期行为：反解提示词必须带上**切块规则**，而且与考生题面是**同一份**
    （`render.split_rules_block`）。渲染规则里有两处「不空行也算两块」——标题紧贴着它后面那段
    正文、同一个列表里相邻的列表项；只写「块与块之间用空行隔开」会让读者把标题并进正文：
    2026-10-04 真机 36 条被拒 17 条（47%），拒稿理由全是「读出来的块类型少一块」，而确定性
    读法 `parse.parse_state` 读同一份文本是对的——错的是读者手里的规则少了两条。
    """
    loc = locales.load(lang)
    _, sample = sample_of(MIXED)
    prompt = judges.build_trace_prompt(str(sample["state"]), loc)
    rules = render.split_rules_block(loc)
    assert rules in prompt
    for rule in loc["doc"]["split_rules"]:
        assert str(rule) in prompt
    asked = [
        item
        for item in questions.build(cf.filled(MIXED, lang), loc, 0)
        if item.family == "block_type"
    ]
    assert asked
    assert rules in str(asked[0].question["instructions"])
@pytest.mark.parametrize("lang", ("zh", "en"))
def test_trace_prompt_carries_the_same_criteria_as_the_examinee(lang: str) -> None:
    """输入：一段 state、两种语言。输出：无。

    预期行为：反解提示词里那六条块类型判据，与考生题面（`block_type` 的 criteria）**逐字同源**
    ——两边都来自语言包的 `type_desc`。这是范式 §9.8 的反解侧版本：考生看得见的判据，读者也得
    看得见，否则「答对了」与「标签对了」不是一回事。

    2026-10-01 真机小批量：不带判据时读者按自己的习惯认，20 条里 10 条读出别的结构（把标题紧贴
    的那段正文并进标题、把没带署名的引用读成正文）。
    """
    loc = locales.load(lang)
    _, sample = sample_of(MIXED)
    prompt = judges.build_trace_prompt(str(sample["state"]), loc)
    for kind in features.BLOCK_KINDS:
        assert str(loc["type_desc"][kind]) in prompt
        assert str(loc["type"][kind]) in prompt
    # 考生题面里的同一条判据也在（同一个 `type_desc`，两处不能各写一份）。
    asked = [
        item for item in questions.build(cf.filled(MIXED, lang), loc, 0) if item.family == "block_type"
    ]
    assert asked
    criteria = asked[0].question["criteria"]
    assert isinstance(criteria, dict)
    for name, desc in criteria.items():
        # 两处同源：题面的判据是 `{kind}: {判据}`，反解提示词里是 `{kind}（{类型名}）：{判据}`。
        assert str(loc["type_desc"][name]) in desc
        assert str(loc["type_desc"][name]) in prompt


def test_all_judges_needs_a_backend_only_for_trace() -> None:
    """输入：有没有后端两种情形。输出：无。预期行为：零 token 的两个总是在；`trace` 要后端才给。"""
    assert [judge.name for judge in judges.all_judges("zh", None)] == ["hygiene", "roundtrip"]
    names = [judge.name for judge in judges.all_judges("zh", FakeBackend())]
    assert names == ["hygiene", "roundtrip", "trace"]


def test_trace_draft_declares_the_three_questions() -> None:
    """输入：反解草稿的字段定义。输出：无。预期行为：三个字段（块类型序列、提示框种类、列表有序性）。"""
    fields = judges.TraceDraft.model_fields
    assert sorted(fields) == ["callout_kind", "kinds", "list_ordered"]
    assert fields["callout_kind"].default == ""
    assert fields["list_ordered"].default is False


def test_quote_blocks_are_the_only_place_a_mark_is_allowed() -> None:
    """输入：一份带引用块的文档。输出：无。

    预期行为：有引用块时署名前缀**照样**列进「不许出现」（英文的 `--` 与中文的 `——` 都是散文里
    常见的破折号，写进别的块会被读成引用）；而引用块本身照样能通过卫生判据。
    """
    plan = features.build_plan(QUOTE)
    assert any(slot.kind == KIND_QUOTE for slot in plan.slots)
    assert set(parse.attribution_marks(LOC)) <= set(render.forbidden_words(plan, LOC))
    _, sample = sample_of(QUOTE)
    assert judges.HygieneJudge("zh").judge(sample).ok


class _UnknownKindBackend:
    """只会答一个不存在的块类型的假后端。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：草稿。预期行为：固定答一个认不出的块类型。"""
        del task
        return Draft(
            data={"kinds": ["paragraph", "nonsense"], "callout_kind": "", "list_ordered": False},
            raw_text="kinds=paragraph,nonsense",
            backend="fake",
            provider="fake",
            model="fake",
            mode="fake",
        )
