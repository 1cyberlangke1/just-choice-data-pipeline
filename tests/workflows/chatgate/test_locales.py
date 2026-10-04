"""chatgate 的语言文件测试：键与结构键一一对应、反面清单只踩自己那一条、提示词不漏槽。"""

from __future__ import annotations

import re

import pytest

from workflows.chatgate import features, judges, locales, questions, render

#: 全空间 45 组 case。
CASES = features.build_cases(512)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_marks_are_pairwise_non_prefixes(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：五个抬头两两不互为前缀——互为前缀时解析会认错块，标签跟着一起错。
    """
    marks = locales.load(lang)["state"]
    names = ("tag_mark", "flaw_mark", "rule_mark", "content_mark", "reply_mark")
    values = [str(marks[name]) for name in names]
    assert len(set(values)) == len(values)
    for left in values:
        for right in values:
            if left != right:
                assert not left.startswith(right), (left, right)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_tables_line_up_with_the_structural_keys(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：判据表 / 毛病表 / 场景表的键与结构键一一对应。"""
    loc = locales.load(lang)
    assert sorted(loc["tag"]) == sorted(features.TAG_IDS)
    assert sorted(loc["flaw"]) == sorted(features.FLAW_IDS)
    assert sorted(loc["scene"]) == sorted(features.SCENES)
    for tag in features.TAG_IDS:
        entry = render.tag_copy(loc, tag)
        assert str(entry["name"]).strip() and str(entry["rule"]).strip()
        assert str(entry["avoid"]).strip()
    for flaw in features.FLAW_IDS:
        entry = render.flaw_copy(loc, flaw)
        assert str(entry["name"]).strip() and str(entry["rule"]).strip()
        assert str(entry["avoid"]).strip()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_count_and_reply_option_tables_are_complete(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：count 有 5 档、reply 有 3 档，文案两两不同。"""
    loc = locales.load(lang)
    count_opts = loc["fam"]["count"]["opt"]
    assert sorted(count_opts) == [f"c{value}" for value in range(len(features.TAG_IDS) + 1)]
    reply_opts = loc["fam"]["reply"]["opt"]
    assert sorted(reply_opts) == [f"r{index}" for index in features.CANDIDATE_SLOTS]
    for table in (count_opts, reply_opts):
        assert len({str(value) for value in table.values()}) == len(table)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_avoid_lists_trip_exactly_their_own_criterion(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每条判据的「不许出现」文案**只**命中它自己那一条，每条毛病的也一样——它们是提示词
    里给模型看的反面清单，同时踩中两条就会把模型往错的方向带。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        avoid = str(render.tag_copy(loc, tag)["avoid"])
        assert features.hit_set(avoid) == (tag,), f"{lang}.{tag}: {avoid!r}"
    for flaw in features.FLAW_IDS:
        avoid = str(render.flaw_copy(loc, flaw)["avoid"])
        assert features.flaw_set(avoid) == (flaw,), f"{lang}.{flaw}: {avoid!r}"


def _rule_alternatives(rule: str) -> list[str]:
    """输入：一条 rule 文案。输出：它列举的每一种可选写法（剥掉引导语与标点）。

    预期行为：先切掉「：」/「:」前面的引导语，再把「，或」/「, or」当分隔符，最后按 `/` 切开。
    只服务于下面那条「反面清单有没有漏项」的检查，不参与任何判定。
    """
    tail = re.split(r"[：:]", rule, maxsplit=1)[-1]
    tail = tail.replace("，或", "/").replace(", or", "/")
    return [item for item in (part.strip(" 「」（）()，。；;") for part in tail.split("/")) if item]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_clean_candidate_avoid_covers_every_flaw_alternative(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每条毛病的反面写法要把**它自己规则里列出的每一种写法**都写进去。

    为什么值得钉（范式 §9.62）：干净的那条候选在提示词里**只印反面写法、不印规则原文**（命中那两条
    印的是规则），所以漏一项就等于模型不知道那一项也算踩中。真机 en 侧踩过这个坑：`avoid` 只列了
    `i promise / guaranteed / sure thing`，规则里还有 `consider it done / you have my word /
    no problem at all / i'll make sure`——模型照着短清单写，10 条里 3 条的「该发的那条候选」
    被 F1 拦下。判据表（`tag`）不在此列：它的反面写法旁边**印着规则原文**，模型看得到全表。
    """
    loc = locales.load(lang)
    for flaw in features.FLAW_IDS:
        entry = render.flaw_copy(loc, flaw)
        missing = [item for item in _rule_alternatives(str(entry["rule"])) if item not in str(entry["avoid"])]
        assert not missing, f"{lang}.{flaw} 的反面写法漏了这些：{missing}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：四族都有问法池，逐判据那一族有 true / false 判据。"""
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert isinstance(spec["q"], list) and spec["q"], f"{family} 没有问法"
    assert str(loc["fam"]["tag"]["criteria_true"]).strip()
    assert str(loc["fam"]["tag"]["criteria_false"]).strip()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompts_and_questions_leave_no_placeholder(lang: str) -> None:
    """输入：一种语言 × 全空间 45 组 case。输出：无。

    预期行为：提示词、反解提示词与题面渲染完之后**一个花括号都不剩**——漏填会在 `format` 上抛错，
    而多写的花括号会原样漏给模型，让它以为那是要填的槽。
    """
    loc = locales.load(lang)
    for case in CASES:
        plan = features.plan_for(case, 0)
        prompt = render.build_prompt(loc, case, plan)
        assert "{" not in prompt and "}" not in prompt, case.describe()
        for item in questions.build(plan, loc, 0):
            instructions = str(item.question["instructions"])
            assert "{" not in instructions and "}" not in instructions
    trace = judges.build_trace_prompt("[state]", loc)
    assert "{" not in trace and "}" not in trace


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_pins_the_language_and_the_scene(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：顶层有 `language_name`，提示词里有一句把语言钉死的约束；每个场景的来路描述都被
    原样印进提示词（不点明来路，模型写出来的东西会四不像）。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    assert "{language}" in str(loc["prompt"]["lang_rule"])
    for scene in features.SCENES:
        case = next(item for item in CASES if item.scene == scene)
        prompt = render.build_prompt(loc, case, features.plan_for(case, 0))
        assert str(loc["language_name"]) in prompt
        assert render.scene_copy(loc, scene) in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_forbids_pasting_the_criteria(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：提示词里明说不许把判据原文贴进消息或候选。"""
    loc = locales.load(lang)
    case = CASES[0]
    assert str(loc["prompt"]["vary_rule"]) in render.build_prompt(loc, case, features.plan_for(case, 0))


