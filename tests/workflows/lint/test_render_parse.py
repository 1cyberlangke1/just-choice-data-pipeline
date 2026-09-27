"""渲染与读法：state 的格式、往返一致、以及读不出来时的报错。"""

from __future__ import annotations

import pytest

from workflows.lint import locales, parse, render
from workflows.lint.spec import LintCase

ZH = locales.load("zh")
EN = locales.load("en")


def state_of(rule: str, genre: str, hits: int, phrasing: int = 0) -> str:
    """输入：四个维度。输出：渲染出来的 state。预期行为：走生产路径（sample → render）。"""
    from workflows.lint import spec as sp

    return render.render_state(sp.sample_spec(LintCase(rule, genre, hits, phrasing), "train", 0), ZH)


def test_state_shape() -> None:
    """输入：一条样本。输出：无。

    预期行为：第一行是查询句（含规则短语）、第二行是抬头（含体裁与行数）、第三行起是
    `n. 行原文` 的编号片段，共 8 行。
    """
    state = state_of("todo", "log", 2)
    rows = state.split("\n")
    assert len(rows) == 2 + 8
    assert ZH["rule"]["todo"] in rows[0]
    assert "服务日志" in rows[1] and "8 行" in rows[1]
    for index, row in enumerate(rows[2:], start=1):
        assert row.startswith(f"{index}. ")


def test_render_is_a_pure_function() -> None:
    """输入：同一条样本渲染两次。输出：无。预期行为：逐字节一致（不用 RNG、不碰后端）。"""
    assert state_of("debug", "doc", 5) == state_of("debug", "doc", 5)


def test_phrasing_changes_the_sentence() -> None:
    """输入：同 case 的两套句面。输出：无。

    预期行为：问法是 case 的一个维度，两套句面必须渲染出**不同**的文本——否则「只有问法
    不同」的两个 case 会渲染出同一段 state，跨切分立刻撞车。
    """
    assert state_of("secret", "code", 1, 0) != state_of("secret", "code", 1, 1)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_roundtrip_recovers_rule_lines_and_marks(lang: str) -> None:
    """输入：两种语言、各条规则各档命中数。输出：无。

    预期行为：从 state 读回来的规则、逐行原文、逐行命中标记与真值规格完全一致——
    这条是判别器能独立重算答案的前提。
    """
    loc = locales.load(lang)
    for rule in ZH["rule"]:
        for genre in ("code", "log", "doc"):
            for hits in (0, 1, 3, 8):
                from workflows.lint import spec as sp

                spec = sp.sample_spec(LintCase(rule, genre, hits, 0), "train", 0)
                state = render.render_state(spec, loc)
                reading = parse.read_state(state, loc)
                assert reading.rule == rule
                assert reading.lines == render.render_lines(spec.plan, loc)
                assert reading.matched == spec.plan.matched()
                assert reading.hits == hits


def test_reading_exposes_the_matched_lines() -> None:
    """输入：命中 3 行的一条样本。输出：无。预期行为：`matched_lines` 就是那 3 行原文。"""
    from workflows.lint import spec as sp

    spec = sp.sample_spec(LintCase("path", "code", 3, 0), "train", 0)
    reading = parse.read_state(render.render_state(spec, ZH), ZH)
    from workflows.lint.spec import matches

    assert len(reading.matched_lines()) == 3
    assert all(matches("path", line) for line in reading.matched_lines())


def test_reparse_uses_the_merged_table() -> None:
    """输入：两种语言的 state（`reparse` 没有语言参数）。输出：无。

    预期行为：走 `Workflow.reparse` → `locales.merged()` 也能认出规则——合并表把两种语言的
    规则短语取并集，所以中英 state 都读得动。
    """
    from workflows.lint import spec as sp
    from workflows.lint.workflow import LintWorkflow

    workflow = LintWorkflow()
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        spec = sp.sample_spec(LintCase("todo", "code", 2, 0), "train", 0)
        reading = workflow.reparse(render.render_state(spec, loc))
        assert reading.rule == "todo" and reading.hits == 2


def test_rule_of_requires_exactly_one_hit() -> None:
    """输入：一个规则短语都不含 / 含两个规则短语的文本。输出：无。

    预期行为：两种都报错——命中 0 个说明文本没说清按哪条规则查，命中 2 个说明规则短语
    不互斥（构造 bug），都不许猜一个。
    """
    with pytest.raises(ValueError, match="命中 0 条规则"):
        parse.rule_of("随便一段没有规则短语的文本", ZH)
    both = f"{ZH['rule']['secret']} 和 {ZH['rule']['todo']}"
    with pytest.raises(ValueError, match="命中 2 条规则"):
        parse.rule_of(both, ZH)


def test_read_state_rejects_non_string() -> None:
    """输入：非字符串 state。输出：无。预期行为：报类型错（不静默当成空片段）。"""
    with pytest.raises(TypeError, match="state 必须是字符串"):
        parse.read_state({"state": "x"}, ZH)  # type: ignore[arg-type]


def test_read_state_rejects_a_short_state() -> None:
    """输入：只有两行的 state。输出：无。预期行为：报人话（至少三行）。"""
    with pytest.raises(ValueError, match="至少要有三行"):
        parse.read_state("规则行\n抬头行", ZH)


def test_read_state_rejects_a_line_without_a_number() -> None:
    """输入：片段里混进一行没有行号的。输出：无。

    预期行为：报错——静默跳过会让片段悄悄少一行，`line` 那一族的题数与答案跟着错。
    """
    state = state_of("secret", "code", 1)
    rows = state.split("\n")
    rows[3] = rows[3].replace("2. ", "  ")
    with pytest.raises(ValueError, match="没有行号前缀"):
        parse.read_state("\n".join(rows), ZH)


def test_read_state_rejects_a_broken_line_number() -> None:
    """输入：行号跳号的片段。输出：无。预期行为：报错并说明应为几号。"""
    state = state_of("secret", "code", 1)
    rows = state.split("\n")
    rows[3] = rows[3].replace("2. ", "5. ")
    with pytest.raises(ValueError, match="应为 2"):
        parse.read_state("\n".join(rows), ZH)


def test_hit_line_reports_a_missing_slot() -> None:
    """输入：模板要一个槽、槽表里没有的规则。输出：无。预期行为：报错（不静默留占位符）。"""
    loc = {"hit": {"secret": {"code": {"tpl": "{nope}"}, "slots": {"s0": {}}}}, "ts": ["t"]}
    with pytest.raises(ValueError, match="占位符"):
        render.hit_line(loc, "secret", "code", 0)


def test_render_state_rejects_an_unknown_rule() -> None:
    """输入：规则名不在语言文件里。输出：无。预期行为：报 KeyError（点名缺的是哪条）。"""
    from workflows.lint import spec as sp

    spec = sp.sample_spec(LintCase("secret", "code", 1, 0), "train", 0)
    broken = {"rule": {}, "q": ["x"], "lead": {"code": "y"}, "hit": {}, "bg": {}, "ts": ["t"]}
    with pytest.raises(KeyError):
        render.render_state(spec, broken)


def test_as_phrases_handles_both_shapes() -> None:
    """输入：单语言的一格（字符串）与合并表的一格（列表）。输出：无。预期行为：都吃。"""
    assert render.as_phrases("硬编码的凭据") == ("硬编码的凭据",)
    assert render.as_phrases(["a", "b"]) == ("a", "b")
