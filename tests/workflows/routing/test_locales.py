"""语言文件的形状：两语结构一致、标签互不重叠、文案不重复。"""

from __future__ import annotations

import pytest

from workflows.routing import judges, locales, parse, questions, spec as sp

LANGS = ("zh", "en")
#: 有「一值一文案」的词表（判别器要把原文翻回键）。
VALUE_TABLES = ("area", "tier", "impact", "urgency", "compliance")


def _shape(node: dict, prefix: str = "") -> set[str]:
    """输入：TOML 节点。输出：摊平后的键路径集合（叶子到键为止）。

    预期行为：只比形状不比内容——两语的键路径必须完全一致，漏一条键就是漏一句文案。
    """
    out: set[str] = set()
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= _shape(value, f"{path}.")
        else:
            out.add(path)
    return out


@pytest.mark.parametrize("lang", LANGS)
def test_every_locale_has_the_same_shape(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：键路径与中文文案完全一致。"""
    assert _shape(locales.load(lang)) == _shape(locales.load("zh"))


def test_labels_do_not_overlap_across_languages() -> None:
    """输入：两语的标签。输出：无。

    预期行为：两语的标签集合不相交——`reparse` 用的是合并标签表，
    重叠会让同一行被两种语言各认一次（`merged_labels` 自己也会报错）。
    """
    zh = set(parse.label_index(locales.load("zh")))
    en = set(parse.label_index(locales.load("en")))
    assert not zh & en


def test_merged_labels_cover_both_languages() -> None:
    """输入：无。输出：无。预期行为：合并表里每种语言的每个标签都在，槽位对得上。"""
    merged = parse.merged_labels()
    for lang in LANGS:
        for text, slot in parse.label_index(locales.load(lang)).items():
            assert merged[text] == slot


@pytest.mark.parametrize("lang", LANGS)
def test_value_tables_have_no_duplicate_text(lang: str) -> None:
    """输入：一种语言的五个词表。输出：无。

    预期行为：每个词表内部原文不重复——重复会让「原文 → 键」反查有歧义，
    判别器就没法把工单读回结构化事实了。
    """
    loc = locales.load(lang)
    for table in VALUE_TABLES:
        judges.reverse(loc[table])


@pytest.mark.parametrize("lang", LANGS)
def test_option_keys_match_the_spec_constants(lang: str) -> None:
    """输入：一种语言的选项表。输出：无。

    预期行为：选项名与 spec 里的常量集合完全一致——多一个选项就是「永远不是答案」的死选项，
    少一个就是「答案不在选项里」的构造 bug。
    """
    loc = locales.load(lang)
    assert set(loc["option"]["severity"]) == set(sp.SEVERITIES)
    assert set(loc["option"]["queue"]) == set(sp.QUEUES)
    assert set(loc["option"]["escalation"]) == set(sp.ESCALATIONS)
    assert len(loc["fam"]["priority"]["levels"]) == len(sp.PRIORITY_LEVELS)


@pytest.mark.parametrize("lang", LANGS)
def test_every_family_has_a_prompt_and_criteria(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：八个题族都有题面；noul 族有 true / false 两行判据文案（官方支持 noul 带描述，
    带上它模型才知道「真」「假」分别意味着什么）。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        entry = loc["fam"][family]
        assert entry["q"], f"{family} 没有题面"
    for family in questions.NOUL_FAMILIES:
        criteria = loc["fam"][family]["criteria"]
        assert criteria["true"] and criteria["false"]
        assert criteria["true"] != criteria["false"]


@pytest.mark.parametrize("lang", LANGS)
def test_option_descriptions_are_unique(lang: str) -> None:
    """输入：一种语言的三张选项表。输出：无。

    预期行为：同一族的选项描述两两不同——描述重复等于两个选项看起来一样，考生没法分辨。
    """
    loc = locales.load(lang)
    for family in questions.CHOICE_FAMILIES:
        texts = list(loc["option"][family].values())
        assert len(set(texts)) == len(texts), f"{family} 的选项描述有重复"


@pytest.mark.parametrize("lang", LANGS)
def test_body_templates_pin_down_the_issue(lang: str) -> None:
    """输入：一种语言的问题句模板 × 四个产品域名。输出：无。

    预期行为：同一产品域下，五个问题句互不相同、也互不包含——判别器就是靠
    「哪一句原样出现在正文里」把问题类型认回来的，模板撞车会让反解出歧义。
    """
    loc = locales.load(lang)
    for area in sp.AREAS:
        texts = [
            loc["body"]["issue"][issue].format(area=loc["area"][area])
            for issue in sp.ISSUES
        ]
        assert len(set(texts)) == len(texts)
        for left in texts:
            for right in texts:
                if left is not right:
                    assert left not in right


@pytest.mark.parametrize("lang", LANGS)
def test_prompt_spells_out_every_rule_table(lang: str) -> None:
    """输入：一种语言的全部题面。输出：无。

    预期行为：判据原文真的在题面里——题面里不写清规则，读者就只能猜，
    这条数据就变成了「考模型会不会读心」。
    """
    loc = locales.load(lang)
    joined = "\n".join(
        prompt for family in questions.FAMILIES for prompt in loc["fam"][family]["q"]
    )
    for severity in sp.SEVERITIES:
        assert severity in joined
    for tier in sp.TIERS:
        assert loc["tier"][tier] in joined
    for impact in sp.IMPACTS:
        assert loc["impact"][impact] in joined
    for urgency in sp.URGENCIES:
        assert loc["urgency"][urgency] in joined
    for queue in sp.QUEUES:
        assert loc["option"]["queue"][queue].split("：")[0].split(":")[0] in joined
