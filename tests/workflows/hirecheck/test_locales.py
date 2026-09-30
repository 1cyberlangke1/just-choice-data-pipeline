"""hirecheck 语言文件：结构键对齐、占位符填得满、反面文案不误伤别的判据。"""

from __future__ import annotations

import pytest

from workflows.hirecheck import features, locales, questions, render

#: 两种语言各跑一遍。
LANGS: tuple[str, ...] = ("zh", "en")


def test_available_lists_both_languages() -> None:
    """输入：无。输出：无。预期行为：语言目录里就是 zh / en 两个文件。"""
    assert locales.available() == ["en", "zh"]


def test_unknown_language_reports_available() -> None:
    """输入：不存在的语言。输出：无。预期行为：报人话并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="zh"):
        locales.load("nope")


def test_pick_wraps_around() -> None:
    """输入：候选列表、越界下标。输出：无。预期行为：回绕取模，不抛异常。"""
    assert locales.pick(["a", "b"], 0) == "a"
    assert locales.pick(["a", "b"], 2) == "a"
    assert locales.pick(["a", "b"], 3) == "b"


@pytest.mark.parametrize("lang", LANGS)
def test_section_marks_are_pairwise_non_prefixes(lang: str) -> None:
    """输入：语言字典。输出：无。

    预期行为：四个抬头两两不互为前缀——互为前缀时，解析会把「判据表」那一行当成「定级规则」的开头，
    整段 state 从此错位。
    """
    marks = locales.load(lang)["state"]
    values = [str(marks[key]) for key in ("fact_mark", "rule_mark", "act_mark", "content_mark")]
    for left in values:
        for right in values:
            if left != right:
                assert not left.startswith(right), (lang, left, right)


@pytest.mark.parametrize("lang", LANGS)
def test_keys_line_up_with_the_code(lang: str) -> None:
    """输入：语言字典。输出：无。

    预期行为：`[fact.*]` / `[level.*]` / `[act.*]` / `[channel.*]` 的键与代码里的取值表**完全相等**
    （多一个少一个都红灯）——漏一个等级就等于有些题没有选项。
    """
    loc = locales.load(lang)
    assert set(loc["fact"]) == set(features.TAG_IDS)
    assert set(loc["level"]) == {f"L{level}" for level in features.LEVELS}
    assert set(loc["act"]) == set(features.ACTS)
    assert set(loc["channel"]) == set(features.CHANNELS)


@pytest.mark.parametrize("lang", LANGS)
def test_every_fact_entry_is_complete(lang: str) -> None:
    """输入：语言字典。输出：无。预期行为：每条判据都有 name / rule / avoid / require，且都非空。"""
    for tag in features.TAG_IDS:
        entry = render.fact_copy(locales.load(lang), tag)
        for key in ("name", "rule", "avoid", "require"):
            assert str(entry[key]).strip(), (lang, tag, key)


@pytest.mark.parametrize("lang", LANGS)
def test_avoid_lists_do_not_trip_other_tags(lang: str) -> None:
    """输入：语言字典。输出：无。

    预期行为：每条判据的 `avoid` 文案**不许误触别的判据**——它是给写手看的反面例子，误触了别的判据
    就等于在提示词里写了一句自相矛盾的规格。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        avoid = str(render.fact_copy(loc, tag)["avoid"])
        stray = set(features.hit_set(avoid)) - {tag}
        assert not stray, f"{lang}.{tag}: {avoid!r} 误触了 {sorted(stray)}"


@pytest.mark.parametrize("lang", LANGS)
def test_avoid_lists_trip_their_own_tag(lang: str) -> None:
    """输入：语言字典。输出：无。

    预期行为：每条判据的 `avoid` 文案**必须命中它自己那条判据**——它描述的就是「让这条判据命中」的那
    个形状（`miss_line` 把它印在「不许出现：」后面）。方向写反（描述成「不命中」那一侧）等于在提示词
    里给出一条**反方向**的指令。

    为什么这条尺子必须有：真机踩过。`hire-10a` 的 zh / en 各 10 条里各有 9 条被 `E_ASSEMBLE` 打回，
    理由清一色是「计划要的是无 / H1 / H4，写出来却命中了 H2+H3+H4」——三条判据的 `avoid` 全写成了
    「不命中」那一侧，写手照着「不许出现：在 `skills:` 那一行里写上 `kubernetes`」办，正好把技能表里
    的硬性技能**删掉**，于是 H2 命中。`test_avoid_lists_do_not_trip_other_tags` 抓不到它：写反的文案
    确实不误触别人，它误触的是**自己**。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        avoid = str(render.fact_copy(loc, tag)["avoid"])
        assert tag in features.hit_set(avoid), f"{lang}.{tag}: {avoid!r} 没有命中它自己的判据"


@pytest.mark.parametrize("lang", LANGS)
def test_require_lists_trip_no_tag_at_all(lang: str) -> None:
    """输入：语言字典。输出：无。

    预期行为：每条判据的 `require` 文案**一条判据都不许命中**——它是「反过来要写成」的那一侧，写手会
    照着它落笔，所以它自己必须是一个干干净净的「不命中」状态（连自己那条也不许命中）。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        require = str(render.fact_copy(loc, tag)["require"])
        assert features.hit_set(require) == (), f"{lang}.{tag}: {require!r} 命中了判据"


@pytest.mark.parametrize("lang", LANGS)
def test_fact_copy_never_carries_braces(lang: str) -> None:
    """输入：语言字典。输出：无。

    预期行为：判据原文与两侧的写法文案里一个花括号都不许有——它们会被拼进提示词，而花括号是填槽
    标记，多一个就会让严格填槽当场炸掉。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        entry = render.fact_copy(loc, tag)
        for key in ("rule", "avoid", "require"):
            assert "{" not in str(entry[key]) and "}" not in str(entry[key]), (lang, tag, key)


@pytest.mark.parametrize("lang", LANGS)
def test_format_spec_covers_every_key(lang: str) -> None:
    """输入：语言字典。输出：无。预期行为：`prompt.format` 的键与 `features.KEY_LINES` 完全相等。"""
    assert set(locales.load(lang)["prompt"]["format"]) == set(features.KEY_LINES)


def test_role_tables_are_the_same_length() -> None:
    """输入：两种语言的岗位方向词表。输出：无。

    预期行为：两张表等长——spec 里存的是**下标**，长短不一就会让某一种语言抽到越界下标（回绕之后
    两种语言的样本对不上号）。
    """
    lengths = {lang: len(locales.load(lang)["prompt"]["roles"]) for lang in LANGS}
    assert len(set(lengths.values())) == 1, lengths
    assert lengths["zh"] == features.ROLE_POOL


def test_name_tables_are_the_same_length() -> None:
    """输入：两种语言的姓名词表。输出：无。

    预期行为：与岗位方向表同理——spec 里存的是**下标**，两张表必须等长，否则某一种语言会抽到越界
    下标。另外每条姓名都要非空（空串等于没钉住，模型又会落回自己的先验）。
    """
    tables = {lang: locales.load(lang)["prompt"]["names"] for lang in LANGS}
    lengths = {lang: len(table) for lang, table in tables.items()}
    assert len(set(lengths.values())) == 1, lengths
    assert lengths["zh"] == features.NAME_POOL
    for lang, table in tables.items():
        assert all(str(name).strip() for name in table), lang


@pytest.mark.parametrize("lang", LANGS)
def test_verdict_words_are_present_and_distinct(lang: str) -> None:
    """输入：语言字典。输出：无。

    预期行为：结论词表非空、去重之后长度不变、且都是小写（匹配是大小写不敏感的，混着大小写只会让
    表看起来有两份）。
    """
    words = render.verdict_words(locales.load(lang))
    assert words
    assert len(set(words)) == len(words)
    assert all(word == word.lower() for word in words)


@pytest.mark.parametrize("lang", LANGS)
def test_missing_verdict_words_report_a_human_message(lang: str) -> None:
    """输入：把结论词表摘掉的语言字典。输出：无。预期行为：当场报人话，而不是静默放过。"""
    loc = dict(locales.load(lang))
    loc.pop("hygiene")
    with pytest.raises(ValueError, match="verdict_words"):
        render.verdict_words(loc)


@pytest.mark.parametrize("lang", LANGS)
def test_rule_formula_has_the_cap_slot(lang: str) -> None:
    """输入：语言字典。输出：无。预期行为：定级公式带一个上限占位符，渲染时填 `MAX_LEVEL`。"""
    assert "{max}" in str(locales.load(lang)["state"]["rule_formula"])
    lines = render.rule_lines(locales.load(lang))
    assert str(features.MAX_LEVEL) in lines[0]


@pytest.mark.parametrize("lang", LANGS)
def test_prompts_fill_every_slot_for_every_case(lang: str) -> None:
    """输入：全空间每一条 case 的提示词。输出：无。

    预期行为：每条提示词都能渲染出来、里面**一个花括号都不剩**（漏填会当场抛错）、并且该出现的东西
    真的出现了——命中侧的判据原文、未命中侧的反面写法、渠道描述、岗位方向、姓名、年限数字。
    """
    loc = locales.load(lang)
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        case = features.HireCheckCase(flags=value, channel=features.CHANNEL_BOARD)
        prompt = render.build_prompt(loc, case, role=0, years=4, name=0)
        assert "{" not in prompt and "}" not in prompt
        assert str(loc["language_name"]) in prompt
        assert render.channel_copy(loc, case.channel) in prompt
        assert locales.pick(loc["prompt"]["roles"], 0) in prompt
        assert locales.pick(loc["prompt"]["names"], 0) in prompt
        assert "4" in prompt
        for tag in features.TAG_IDS:
            entry = render.fact_copy(loc, tag)
            if tag in tags:
                assert str(entry["rule"]) in prompt, (lang, value, tag)
            else:
                # 未命中侧两侧的写法都要印出来：反面（不许出现）与正面（反过来要写成）。
                assert str(entry["avoid"]) in prompt, (lang, value, tag)
                assert str(entry["require"]) in prompt, (lang, value, tag)


@pytest.mark.parametrize("lang", LANGS)
def test_question_templates_fill_every_slot(lang: str) -> None:
    """输入：两种问法下四个题族的题面。输出：无。预期行为：题面都渲染得出来、没有剩下的花括号。"""
    loc = locales.load(lang)
    case = features.HireCheckCase(flags="H1+H4", channel=features.CHANNEL_BOARD)
    for phrasing in (0, 1):
        built = questions.build(features.plan_for(case, phrasing), loc, phrasing)
        assert [item.family for item in built] == ["flag"] * 4 + ["grade", "combo", "act"]
        for question in built:
            text = str(question.question["instructions"])
            assert "{" not in text and "}" not in text, (lang, phrasing, text)
