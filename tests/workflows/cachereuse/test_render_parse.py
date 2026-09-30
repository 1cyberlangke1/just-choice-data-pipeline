"""渲染与反解的测试：逐步一致、写法维只改文本、结构坏了要报人话。"""

from __future__ import annotations

import pytest

from workflows.cachereuse import judges, locales, parse, render, spec as sp


def _parsed(case: sp.ReuseCase, lang: str) -> parse.ParsedReuse:
    """输入：case、语言。输出：把这一条渲染再反解的结果（测试里反复用）。"""
    loc = locales.load(lang)
    spec = sp.sample_spec(__import__("random").Random(0), case, "train", 0, 0)
    return parse.parse_state(render.build(spec, loc), parse.tables_of(loc))


@pytest.mark.parametrize("lang", ("zh", "en"))
@pytest.mark.parametrize("band", sp.BANDS)
def test_roundtrip_keeps_the_steps(band: str, lang: str) -> None:
    """输入：四档 × 两种语言。输出：无。

    预期行为：反解出来的步数、请求号、来源都与真值对得上——渲染的每一行都能被认回来。
    """
    case = sp.ReuseCase(5, band, "action", sp.SYNONYM, "desktop")
    spec = sp.sample_spec(__import__("random").Random(0), case, "train", 0, 0)
    parsed = _parsed(case, lang)
    assert parsed.request_id == spec.query_id
    assert parsed.flavor == locales.load(lang)["flavor"]["desktop"]
    assert len(parsed.cached) == len(parsed.current) == case.length


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_verbatim_steps_are_byte_identical(lang: str) -> None:
    """输入：逐字相同的写法。输出：无。

    预期行为：等价步两侧**逐字相同**；不同步两侧的键至少差一处。这一条把「写法维只改文本」
    钉住——逐字相同时归一化那一步其实什么都没做。
    """
    case = sp.ReuseCase(6, "half", "target", sp.VERBATIM, "web")
    parsed = _parsed(case, lang)
    for index in range(case.length):
        equivalent = index < case.prefix or case.prefix < index <= case.prefix + case.tail
        same_text = parsed.cached[index] == parsed.current[index]
        assert same_text == equivalent, (index, parsed.cached[index], parsed.current[index])


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_synonym_steps_differ_in_wording_but_not_in_keys(lang: str) -> None:
    """输入：同义写法。输出：无。

    预期行为：等价步两侧**写法不同、键相同**（这正是「语义可复用」那一层）；不同步的键不同。
    只比文本的人会在这里判错，只比键的人会漏掉「写法维」——两条都得查。
    """
    case = sp.ReuseCase(6, "half", "action", sp.SYNONYM, "web")
    parsed = _parsed(case, lang)
    words = locales.words()
    cached = judges.to_keys(parsed.cached, words)
    current = judges.to_keys(parsed.current, words)
    for index in range(case.length):
        equivalent = index < case.prefix or case.prefix < index <= case.prefix + case.tail
        assert (cached[index] == current[index]) == equivalent, index
        if equivalent:
            assert parsed.cached[index] != parsed.current[index], index
    assert cached[case.prefix] != current[case.prefix]


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_step_numbering_starts_at_one(lang: str) -> None:
    """输入：一条样本。输出：无。预期行为：state 里的步号从 1 起、连续、与步数一致。"""
    case = sp.ReuseCase(4, "few", "action", sp.VERBATIM, "web")
    spec = sp.sample_spec(__import__("random").Random(0), case, "train", 0, 0)
    state = render.build(spec, locales.load(lang))
    numbers = [
        int(line.strip().split(".")[0])
        for line in state.splitlines()
        if line.strip()[:1].isdigit()
    ]
    assert numbers == list(range(1, case.length + 1)) * 2


def test_parse_rejects_a_missing_section() -> None:
    """输入：少了当前轨迹那一段的 state。输出：无。预期行为：报「一步都没有」而不是给空序列。"""
    loc = locales.load("zh")
    case = sp.ReuseCase(4, "few", "action", sp.VERBATIM, "web")
    spec = sp.sample_spec(__import__("random").Random(0), case, "train", 0, 0)
    lines = render.build(spec, loc).splitlines()
    head = lines[: lines.index("[当前轨迹]")]
    with pytest.raises(ValueError, match="一步都没有"):
        parse.parse_state("\n".join(head) + "\n", parse.tables_of(loc))


def test_parse_rejects_a_gap_in_step_numbers() -> None:
    """输入：步号跳了一号的 state。输出：无。预期行为：报「跳号」，不把两步当成一步。"""
    loc = locales.load("zh")
    case = sp.ReuseCase(4, "few", "action", sp.VERBATIM, "web")
    spec = sp.sample_spec(__import__("random").Random(0), case, "train", 0, 0)
    state = render.build(spec, loc).replace("  2. ", "  3. ", 1)
    with pytest.raises(ValueError, match="跳号"):
        parse.parse_state(state, parse.tables_of(loc))


def test_parse_rejects_an_unknown_step_line() -> None:
    """输入：区块里多了一行认不出来的内容。输出：无。

    预期行为：报「认不出的步骤行」——静默跳过会把渲染写歪的一步当成不存在，步数就对不上了。
    """
    loc = locales.load("zh")
    case = sp.ReuseCase(4, "few", "action", sp.VERBATIM, "web")
    spec = sp.sample_spec(__import__("random").Random(0), case, "train", 0, 0)
    state = render.build(spec, loc).replace("[当前轨迹]\n", "[当前轨迹]\n  这是多出来的一行\n")
    with pytest.raises(ValueError, match="认不出的步骤行"):
        parse.parse_state(state, parse.tables_of(loc))


def test_parse_rejects_unequal_trace_lengths() -> None:
    """输入：两条轨迹步数不同的 state。输出：无。

    预期行为：报「步数不同」——逐步比对的题族全部建立在「步号一一对应」上，
    步数不同时按较短那条对齐会把答案算错而不报错。
    """
    loc = locales.load("zh")
    case = sp.ReuseCase(4, "few", "action", sp.VERBATIM, "web")
    spec = sp.sample_spec(__import__("random").Random(0), case, "train", 0, 0)
    lines = render.build(spec, loc).splitlines()
    del lines[-1]
    with pytest.raises(ValueError, match="步数不同"):
        parse.parse_state("\n".join(lines) + "\n", parse.tables_of(loc))


def test_parse_rejects_a_non_string_state() -> None:
    """输入：不是字符串的 state。输出：无。预期行为：报类型，不抛 AttributeError。"""
    with pytest.raises(ValueError, match="不是字符串"):
        parse.parse_state(123, parse.merged_tables())


def test_merged_tables_read_both_languages() -> None:
    """输入：中英两条 state。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，合并表必须两种语言都认得出——产物按
    `locales[seq % len]` 轮转渲染，反解只认一种语言就会把另一种整片判成坏数据。
    """
    case = sp.ReuseCase(4, "none", "target", sp.SYNONYM, "web")
    spec = sp.sample_spec(__import__("random").Random(0), case, "train", 0, 0)
    for lang in ("zh", "en"):
        parsed = parse.parse_state(render.build(spec, locales.load(lang)), parse.merged_tables())
        assert parsed.request_id == spec.query_id
        assert len(parsed.cached) == case.length
