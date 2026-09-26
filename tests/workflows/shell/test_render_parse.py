"""渲染与反解的往返：文本 → 字段 → 文本必须一致，标签表必须语言无关。"""

from __future__ import annotations

import random

import pytest

from workflows.shell import locales, parse, render
from workflows.shell import spec as sp
from workflows.shell import workflow as wf


def sample_cases(limit: int = 24) -> list[sp.ShellCase]:
    """输入：条数上限。输出：从留出档里均匀取的一批 case（两种方言都在里面）。"""
    cases = sp.build_cases(wf.OOD_CONFIG)
    step = max(1, len(cases) // limit)
    return cases[::step][:limit]


@pytest.mark.parametrize("locale", ["zh", "en"])
def test_render_then_parse_roundtrips(locale: str) -> None:
    """输入：一批 case × 一种语言。输出：无。

    预期行为：渲染出来的文本反解回去，方言 / 工作目录 / 用户目标 / 已读文件 / 命令
    五个字段全部逐字相等。
    """
    loc = locales.load(locale)
    for index, case in enumerate(sample_cases()):
        spec = sp.sample_spec(random.Random(index), case, "train", index, 0)
        state = render.build(spec, loc)
        parsed = parse.parse_request(state, parse.label_index(loc))
        assert parsed.dialect == case.dialect
        assert parsed.cwd == spec.cwd
        assert parsed.goal == render.goal_text(spec, loc)
        assert parsed.reads == spec.reads
        assert parsed.command == spec.command


def test_merged_labels_are_language_independent_and_disjoint() -> None:
    """输入：全部语言。输出：无。

    预期行为：合并表能覆盖每种语言的全部标签，且同一个标签文本不会出现在两种语言里——
    重叠会让语言无关的反解把两种语言的标题串成一行。
    """
    merged = parse.merged_labels()
    for lang in locales.available():
        for text, slot in parse.label_index(locales.load(lang)).items():
            assert merged[text] == slot
    assert len(merged) == len(parse.META_KEYS) * len(locales.available())


def test_reparse_uses_the_merged_table() -> None:
    """输入：一条英文 state。输出：反解结果。

    预期行为：`reparse` 没有语言参数，但合并表能认英文标签。
    """
    spec = sp.sample_spec(
        random.Random(3),
        sp.ShellCase("powershell", "remove_tree", "dist_dir", "build_dir", "elevated", "terse", 2),
        "train",
        0,
        0,
    )
    state = render.build(spec, locales.load("en"))
    parsed = wf.ShellWorkflow().reparse(state)
    assert parsed.dialect == "powershell"
    assert parsed.command == spec.command
    assert parsed.goal == render.goal_text(spec, locales.load("en"))


def test_goal_text_follows_the_case_not_the_command() -> None:
    """输入：同一个命令、两种用户目标。输出：两段目标文本。

    预期行为：目标句里点名的是 `goal_target`——目标不一致时，句子里出现的是**另一个**
    对象的路径，这正是 `matched_goal` 那族题的判据来源。
    """
    loc = locales.load("zh")
    aligned = sp.sample_spec(
        random.Random(1),
        sp.ShellCase("bash", "remove_tree", "build_dir", "build_dir", "plain", "imperative", 1),
        "train",
        0,
        0,
    )
    misaligned = sp.sample_spec(
        random.Random(1),
        sp.ShellCase("bash", "remove_tree", "build_dir", "dist_dir", "plain", "imperative", 1),
        "train",
        0,
        0,
    )
    assert aligned.command == misaligned.command
    assert "build/" in render.goal_text(aligned, loc)
    assert "dist/" in render.goal_text(misaligned, loc)
    assert "build/" not in render.goal_text(misaligned, loc)


def test_parse_ignores_unknown_lines_and_reports_bad_environment() -> None:
    """输入：混了噪声行的 state / 一段认不出方言的 state。输出：反解结果 / 异常。

    预期行为：噪声行被忽略（不炸、也不污染字段）；执行环境认不出来时直接报错，
    而不是静默退回默认方言——那会让反解用错模板，把好数据判成坏的。
    注意命令是**最后一段**：`待执行命令` 之后的行按设计都属于命令本体，
    所以噪声行只能夹在字段区里。
    """
    loc = locales.load("zh")
    spec = sp.sample_spec(
        random.Random(4),
        sp.ShellCase("bash", "list_dir", "build_dir", "build_dir", "plain", "imperative", 1),
        "train",
        0,
        0,
    )
    lines = render.build(spec, loc).splitlines()
    noisy = "\n".join([lines[0], "补充说明：这不是字段", *lines[1:]])
    parsed = parse.parse_request(noisy, parse.label_index(loc))
    assert parsed.command == spec.command
    with pytest.raises(ValueError, match="方言"):
        parse.dialect_of("cmd.exe（Windows）")


def test_unknown_goal_style_reports_the_op() -> None:
    """输入：语言文件里没有的句式名。输出：异常。

    预期行为：报错点出是哪个操作缺哪种句式（改语言文件时一眼看得出漏了哪条）。
    """
    spec = sp.sample_spec(
        random.Random(5),
        sp.ShellCase("bash", "list_dir", "build_dir", "build_dir", "plain", "imperative", 1),
        "train",
        0,
        0,
    )
    loc = locales.load("zh")
    loc["goal"]["list_dir"].pop("imperative")
    with pytest.raises(KeyError, match="list_dir"):
        render.goal_text(spec, loc)


def test_env_text_starts_with_the_dialect_token() -> None:
    """输入：两种语言的执行环境文案。输出：无。

    预期行为：值以方言名开头——反解按首个英文词元认方言，所以翻译不会破坏反解。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for dialect in sp.DIALECTS:
            assert loc["env"][dialect]["text"].startswith(dialect)
            assert parse.dialect_of(loc["env"][dialect]["text"]) == dialect
