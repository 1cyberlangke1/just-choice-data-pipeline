"""行级规则检查的多语言文案测试：词表与代码表必须一一对得上。"""

from __future__ import annotations

import pytest

from workflows.lint import features, locales, questions, render


def all_keys(node: object, path: str = "") -> set[str]:
    """输入：解析后的 TOML 节点、路径前缀。输出：全部叶子路径。

    预期行为：列表按「一个叶子」处理（不展开下标）——两个语言文件的**结构**要一致，
    值当然不一样。
    """
    if isinstance(node, dict):
        out: set[str] = set()
        for key, value in node.items():
            out |= all_keys(value, f"{path}.{key}")
        return out
    return {path}


def test_available_locales() -> None:
    """输入：无。输出：无。预期行为：就 zh / en 两个，按文件名排序。"""
    assert locales.available() == ["en", "zh"]


def test_missing_locale_reports_available() -> None:
    """输入：不存在的语言。输出：无。预期行为：报错并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="zh"):
        locales.load("nope")


def test_locales_have_the_same_structure() -> None:
    """输入：两个语言。输出：无。预期行为：叶子路径集合完全一致（少一条就是漏翻）。"""
    assert all_keys(locales.load("zh")) == all_keys(locales.load("en"))


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_rule_table_is_complete_and_brace_free(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：五条规则的短语与判据都在、都不为空，判据里**不含花括号**（渲染要经过严格
    填槽，带花括号会误报），而且短语两两不互为子串（反解靠它认规则名）。
    """
    loc = locales.load(lang)
    assert set(loc["rule"]) == set(features.RULES)
    names = [str(loc["rule"][rule]["name"]) for rule in features.RULES]
    for rule in features.RULES:
        name = str(loc["rule"][rule]["name"])
        criteria = str(loc["rule"][rule]["criteria"])
        assert name.strip() and criteria.strip()
        assert "{" not in criteria and "}" not in criteria
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert left not in right and right not in left


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_genre_leads_and_handlings_are_complete(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：三条体裁抬头填完 `{n}` 之后互不相同（反解靠它认体裁），三种处置都有词。
    """
    loc = locales.load(lang)
    assert set(loc["state"]["lead"]) == set(features.GENRES)
    assert set(loc["genre"]) == set(features.GENRES)
    leads = {
        genre: str(loc["state"]["lead"][genre]).format(n=features.SIZE)
        for genre in features.GENRES
    }
    assert len(set(leads.values())) == len(features.GENRES)
    assert set(loc["gate"]) == set(features.GATES)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_prompt_and_verify_keys_are_present(lang: str) -> None:
    """输入：一个语言。输出：无。预期行为：提示词与反解要用的槽一个不缺。"""
    loc = locales.load(lang)
    for key in (
        "intro",
        "hit_line",
        "hit_none",
        "clean_line",
        "clean_none",
        "rules_header",
        "rule_line",
        "output",
    ):
        assert str(loc["prompt"][key]).strip()
    for key in ("intro", "state_header", "q"):
        assert str(loc["verify"][key]).strip()
    for key in ("body_mark", "gate_mark", "rule_prefix", "criteria_prefix", "gate_row"):
        assert str(loc["state"][key]).strip()
    assert str(loc["meta"]["list_sep"])


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_families_cover_every_family_and_option(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：四个题族的题面都在；`count` 的选项名恒为 c0~c8、`gate` 恒为三种处置、
    `line` 的判据恒为 true / false——选项名与代码里的取值表**完全一致**（多一个少一个都会
    让「答对了」与「标签对了」不是一回事）。
    """
    loc = locales.load(lang)
    assert set(loc["fam"]) == set(questions.FAMILIES)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert [str(item).strip() for item in spec["q"]]
    assert set(loc["fam"]["count"]["opt"]) == {f"c{n}" for n in range(features.SIZE + 1)}
    assert set(loc["fam"]["gate"]["opt"]) == set(features.GATES)
    assert set(loc["fam"]["line"]["criteria"]) == {"true", "false"}
    assert str(loc["fam"]["spot"]["opt"]["row"]).strip()


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_rendered_state_is_parseable_in_that_language(lang: str) -> None:
    """输入：一个语言。输出：无。预期行为：这一语言的 state 能被反解回同一份计划。"""
    loc = locales.load(lang)
    plan = features.plan_for(features.LintCase(features.RULE_DEBUG, "log", 3), 0)
    lines = [
        'print("x")' if row in plan.hit_rows else "import os" for row in range(features.SIZE)
    ]
    parsed = render.parse_state(render.to_state(lines, plan, loc))
    assert parsed.rule == plan.rule and parsed.genre == plan.genre
    assert parsed.lines == tuple(lines)
    assert render.detect_locale(render.to_state(lines, plan, loc)) == lang
