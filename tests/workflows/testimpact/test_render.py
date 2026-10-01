"""testimpact 的文本外壳测试：state 往返、形状闸、提示词不带成品例子。"""

from __future__ import annotations

import pytest

from jc.workflow import DraftError
from tests.workflows.testimpact.conftest import (
    TITLE_EN,
    TITLE_ZH,
    diff_for,
    file_block,
    title_for,
)
from workflows.testimpact import features, locales, render
from workflows.testimpact.features import ImpactCase

#: 两种语言各测一遍（渲染是语言相关的，只测一种语言是这一族踩过的坑）。
LANGS = ("zh", "en")


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("touch", features.TOUCH_SETS)
def test_state_roundtrips_for_every_case(lang: str, touch: str) -> None:
    """输入：16 组触及组合 × 两种语言。输出：无。

    预期行为：`to_state` 出来的 state 能被 `parse_state` 反解回同一份落点与标题，覆盖表与档位表
    的编号 / 档位顺序都与 `features` 一致。
    """
    case = ImpactCase(touch=touch, flavor="fix")
    draft = render.ContentDraft(title=title_for(lang), diff=diff_for(case))
    render.check_draft(draft, case, locales.load(lang))
    state = render.to_state(draft, locales.load(lang))
    parsed = render.parse_state(state)
    assert parsed.paths == features.PATH_IDS
    assert parsed.levels == features.LEVELS
    assert parsed.title == title_for(lang)
    assert parsed.touched == case.touched()
    assert render.detect_locale(state) == lang


@pytest.mark.parametrize("lang", LANGS)
def test_prompt_carries_the_touched_files_and_no_finished_diff(lang: str) -> None:
    """输入：两种语言。输出：无。

    预期行为：提示词逐条印上必须改到的文件（这是注入特征，不是例子），但**一个成品 diff 头都不给**
    ——给了模型就会整段照抄（范式 §9.45 / §9.64）。形状只以占位形态出现。
    """
    case = ImpactCase(touch="P1+P2", flavor="feat")
    prompt = render.build_prompt(locales.load(lang), case)
    for pid in case.touched():
        assert features.PATH_FILES[pid] in prompt
    for pid in features.PATH_IDS:
        assert f"diff --git a/{features.PATH_FILES[pid]}" not in prompt
    assert str(features.MIN_DIFF_CHARS) in prompt
    assert str(features.MAX_DIFF_CHARS) in prompt


@pytest.mark.parametrize("lang", LANGS)
def test_check_draft_accepts_a_faithful_change(lang: str) -> None:
    """输入：完全兑现计划的改动。输出：无。预期行为：一条问题都不报。"""
    case = ImpactCase(touch="P1+P3+P5", flavor="refactor")
    draft = render.ContentDraft(title=title_for(lang), diff=diff_for(case))
    assert render.draft_problems(draft, case, locales.load(lang)) == []


def _problems(touch: str, draft: render.ContentDraft, lang: str = "zh") -> list[str]:
    """输入：触及组合、草稿、语言。输出：形状闸报出来的问题。"""
    case = ImpactCase(touch=touch, flavor="fix")
    return render.draft_problems(draft, case, locales.load(lang))


def test_check_draft_rejects_a_short_title() -> None:
    """输入：一个字的标题。输出：无。预期行为：报「标题长度」。"""
    case = ImpactCase(touch="P1", flavor="fix")
    problems = _problems("P1", render.ContentDraft(title="短", diff=diff_for(case)))
    assert any("提交标题长度" in item for item in problems)


def test_check_draft_rejects_a_missing_file() -> None:
    """输入：少写了一个文件。输出：无。预期行为：报「碰的文件」对不上。"""
    case = ImpactCase(touch="P1+P2", flavor="fix")
    only_p1 = "\n".join(file_block(features.PATH_FILES["P1"]))
    problems = _problems("P1+P2", render.ContentDraft(title=TITLE_ZH, diff=only_p1))
    assert any("碰的文件" in item for item in problems)
    assert case.touch == "P1+P2"


def test_check_draft_rejects_an_extra_file() -> None:
    """输入：多改了一个文件。输出：无。预期行为：同样报「碰的文件」对不上（一条不多一条不少）。"""
    diff = "\n".join(file_block(features.PATH_FILES["P3"])) + "\n" + "\n".join(
        file_block(features.PATH_FILES["P2"])
    )
    problems = _problems("P3", render.ContentDraft(title=TITLE_ZH, diff=diff))
    assert any("碰的文件" in item for item in problems)


def test_check_draft_rejects_prose_and_fences() -> None:
    """输入：散文、围栏、没有块头的「改动」。输出：无。预期行为：三种各报各的。"""
    case = ImpactCase(touch="P1", flavor="fix")
    prose = render.ContentDraft(title=TITLE_ZH, diff="这次改动调整了请求解析里的边界判断。" * 8)
    assert any("不像 diff 的行" in item for item in _problems("P1", prose))
    fenced = render.ContentDraft(
        title=TITLE_ZH, diff=f"```\n{diff_for(case)}\n```"
    )
    assert any("围栏" in item for item in _problems("P1", fenced))
    no_hunk = render.ContentDraft(
        title=TITLE_ZH,
        diff="\n".join(
            [
                "diff --git a/src/api/handlers.py b/src/api/handlers.py",
                "index 1a2b3c4..5d6e7f8 100644",
                "--- a/src/api/handlers.py",
                "+++ b/src/api/handlers.py",
                "+    request_id = payload[\"id\"]",
                "-    return lookup(None)",
                "+    return lookup(request_id)",
                "+    log.info(\"handled %s\", request_id)",
            ]
        ),
    )
    assert any("块头" in item for item in _problems("P1", no_hunk))


def test_check_draft_rejects_a_pasted_table_line() -> None:
    """输入：把覆盖表那一行整段贴进改动。输出：无。预期行为：报「整段贴了进去」。"""
    loc = locales.load("zh")
    pasted = render.coverage_line(loc, "P1")
    diff = "\n".join(file_block(features.PATH_FILES["P1"])) + "\n+ " + pasted
    problems = _problems("P1", render.ContentDraft(title=TITLE_ZH, diff=diff))
    assert any("整段贴了进去" in item for item in problems)


def test_check_draft_raises_draft_error() -> None:
    """输入：写歪的草稿。输出：无。预期行为：`check_draft` 抛 `DraftError`（管线据此重渲染）。"""
    case = ImpactCase(touch="P1", flavor="fix")
    with pytest.raises(DraftError, match="提交标题"):
        render.check_draft(render.ContentDraft(title="", diff=diff_for(case)), case, locales.load("zh"))


def test_pasted_rules_only_matches_whole_lines() -> None:
    """输入：整行表行、以及只抄了半行的改动。输出：无。预期行为：只认整行完全相同的那种。"""
    loc = locales.load("zh")
    whole = render.coverage_line(loc, "P2")
    assert render.pasted_rules(f"+ {whole}", loc) == [whole]
    assert render.pasted_rules("+ " + whole[: len(whole) // 2], loc) == []
    assert render.pasted_rules(diff_for(ImpactCase(touch="P2", flavor="fix")), loc) == []


@pytest.mark.parametrize(
    ("diff", "expect"),
    [
        ("@@ -1,3 +1,4 @@\n context\n+added\n", []),
        ("@@ -1,3 +1,4 @@\n+added\n这行是散文\n", ["这行是散文"]),
        ("index 1a2b3c4..5d6e7f8 100644\nnew file mode 100644\n", []),
        ("rename from src/api/handlers.py\nrename to src/api/handler.py\n", []),
        ("\\ No newline at end of file\n", []),
        # 光秃秃的 `@@`：算块头（模型常这么写，整块 `+` / `-` 行得跟着进块内）
        ("@@\n context\n+added\n", []),
        ("纯散文一行\n", ["纯散文一行"]),
    ],
)
def test_diff_shape_problems(diff: str, expect: list[str]) -> None:
    """输入：几种行形态。输出：无。

    预期行为：块内只许 `+` / `-` / 空格开头，块外只许 git 自己写的元信息行；别的都报出来。
    """
    assert render.diff_shape_problems(diff) == expect


@pytest.mark.parametrize(
    ("diff", "expect"),
    [
        # 行数写歪了：按块内实际行数重算（旧侧 1 行、新侧 2 行）
        ("@@ -10,2 +10,5 @@\n context\n+added\n", "@@ -10 +10,2 @@\n context\n+added"),
        # 行数本来就对：连 `,1` 该省略这件事也照 git 的写法来
        ("@@ -3 +3,2 @@ def f():\n context\n+added\n", "@@ -3 +3,2 @@ def f():\n context\n+added"),
        # 空块：两侧都写 `,0`（不是省略）
        ("@@ -1,4 +1,9 @@\n", "@@ -1,0 +1,0 @@"),
        # 一个文件里两个块：各算各的，起点与尾部上下文都不动
        (
            "@@ -1,9 +1,9 @@\n a\n-b\n+c\n@@ -20,5 +20,6 @@\n d\n+e\n",
            "@@ -1,2 +1,2 @@\n a\n-b\n+c\n@@ -20 +20,2 @@\n d\n+e",
        ),
        # 没有块头的正文原样返回
        ("diff --git a/x b/x\n纯散文\n", "diff --git a/x b/x\n纯散文"),
        # 光写 `@@`：起点按 1 补、行数照块内实际行数算（旧侧 1 行、新侧 2 行）
        ("@@\n context\n+added\n", "@@ -1 +1,2 @@\n context\n+added"),
        # 两个块：前一块带行号、后一块光写 `@@`，各算各的，别把后一块并进前一块
        (
            "@@ -5,2 +5,2 @@\n a\n-b\n+c\n@@\n d\n+e\n",
            "@@ -5,2 +5,2 @@\n a\n-b\n+c\n@@ -1 +1,2 @@\n d\n+e",
        ),
    ],
)
def test_normalize_hunks_recounts_every_hunk(diff: str, expect: str) -> None:
    """输入：块头行数写歪的改动。输出：无。

    预期行为：只重算行数——起点、尾部上下文与正文一行都不动（这步是算术，不是改写内容）。
    """
    assert render.normalize_hunks(diff) == expect


def test_normalize_hunks_leaves_a_canonical_diff_alone() -> None:
    """输入：一份行数本来就自洽的改动。输出：无。预期行为：逐字节不变（不许自作聪明改内容）。"""
    canonical = diff_for(ImpactCase(touch="P1+P3", flavor="fix"))
    assert render.normalize_hunks(canonical) == canonical


def test_bare_hunk_header_passes_the_gate_and_comes_out_canonical() -> None:
    """输入：块头写成光秃秃 `@@` 的改动。输出：无。

    预期行为：① `check_draft` 不抛——格式简写不该把整条样本判死；② 落进 state 的块头是规范形态
    （起点补 1、行数按块内实际行数算）。

    2026-10-01 真机小批量：`testimpact` 的 8 条拒稿全是这一种写法——`@@` 不被认，整块 `+` / `-`
    行就跟着落进「块外」，一条样本连报三条错，看着像模型写了散文，其实只差两个行号。
    """
    case = ImpactCase(touch="P1+P2", flavor="fix")
    canonical = diff_for(case)
    header = "@@ -12,4 +12,5 @@ def handle(request):"
    assert canonical.count(header) == 2
    bare = canonical.replace(header, "@@")
    assert render.diff_shape_problems(bare) == []
    draft = render.ContentDraft(title=TITLE_ZH, diff=bare)
    render.check_draft(draft, case, locales.load("zh"))
    state = render.to_state(draft, locales.load("zh"))
    assert state.count("@@ -1,4 +1,5 @@") == 2


def test_parse_state_is_language_independent_and_safe() -> None:
    """输入：非字符串、认不出的字符串。输出：无。预期行为：给空结果而不是抛异常。"""
    for value in (None, 42, "随便一段话"):
        parsed = render.parse_state(value)
        assert parsed.paths == ()
        assert parsed.levels == ()
        assert parsed.diff is None
        assert parsed.found is False


def test_detect_locale_separates_the_two_languages() -> None:
    """输入：两种语言各一份 state。输出：无。

    预期行为：认的是**改动抬头**（结构键 `P1` 两种语言一样，认它会把别的语言抢过来，见范式 §9.22）。
    """
    for lang, title in (("zh", TITLE_ZH), ("en", TITLE_EN)):
        case = ImpactCase(touch="P3", flavor="fix")
        state = render.to_state(
            render.ContentDraft(title=title, diff=diff_for(case)), locales.load(lang)
        )
        assert render.detect_locale(state) == lang


@pytest.mark.parametrize("lang", LANGS)
def test_coverage_and_ladder_lines_are_readable_by_the_parser(lang: str) -> None:
    """输入：两种语言。输出：无。

    预期行为：覆盖表每行的编号、档位表每行的档位都能按该语言的分隔符切回来——渲染与解析共用
    同一份格式常量，这一条把「两边漂移」变成机器可查。
    """
    loc = locales.load(lang)
    state = render.to_state(
        render.ContentDraft(
            title=title_for(lang), diff=diff_for(ImpactCase(touch="P1", flavor="fix"))
        ),
        loc,
    )
    parsed = render.parse_with(state, loc)
    assert parsed.paths == features.PATH_IDS
    assert parsed.levels == features.LEVELS
    for pid in features.PATH_IDS:
        assert render.coverage_line(loc, pid) in state
    for level in features.LEVELS:
        assert render.ladder_line(loc, level) in state
