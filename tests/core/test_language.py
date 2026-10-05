"""`jc.language`：产物文本的书写系统必须与声明语言一致。

输入：`jc.language` 的公开函数与阈值常量。
输出：无（断言）。
预期行为：判据本身只回答「这段文本配不配得上 zh / en 这个标签」，所以测试分三组——
比例算得对、越界才报、判不了就不报。**阈值那几条是拿真机产物的实测数字钉住的**：改阈值就得
同时改证据，不许悄悄挪（挪松了坏数据会溜过去，挪紧了正常数据会被拒）。

为什么值得进测试套件：这条判据在渲染期会拒掉样本（`DraftError` → 重渲染），判歪了要么漏掉
坏数据（`rerank` 的 en 侧整段写成中文就是这么溜进来的），要么把好数据反复重渲染烧 token。
"""

from __future__ import annotations

import pytest

from jc import language
from jc.workflow import DraftError


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("中文正文", 1.0),
        ("plain english text", 0.0),
        ("中文ab", 0.5),
        ("", 0.0),
        ("12345 ！？", 0.0),
    ],
)
def test_ratio_counts_only_han_and_latin(text: str, expect: float) -> None:
    """输入：几段文本。输出：无。

    预期行为：分母只数汉字与拉丁字母——数字、标点、空白都不算，纯符号文本按 0.0 处理。
    """
    assert language.script_ratio(text) == pytest.approx(expect)


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("中文正文", 4),
        ("plain english text", 16),
        ("中文 mixed 正文", 9),
        ("12345 ！？", 0),
    ],
)
def test_scripted_chars_counts_both_scripts(text: str, expect: int) -> None:
    """输入：几段文本。输出：无。预期行为：汉字 + 拉丁字母的个数，别的字符一个都不数。"""
    assert language.scripted_chars(text) == expect


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("plain english text", 3),
        ("mySecretToken 写进 settings.json", 3),
        ("中文正文", 0),
        ("a-b_c", 3),
    ],
)
def test_latin_words_counts_runs_not_letters(text: str, expect: int) -> None:
    """输入：几段文本。输出：无。预期行为：连续的字母算**一个**词，标识符不该被拆成一串词。"""
    assert language.latin_words(text) == expect


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("中文正文", 1.0),
        ("把 mySecretToken 写进 settings.json", 0.5),
        ("Sponsored by ABC", 0.0),
        ("", 0.0),
    ],
)
def test_han_share_compares_han_against_latin_words(text: str, expect: float) -> None:
    """输入：几段文本。输出：无。

    预期行为：zh 侧的分母是**词**不是字母——「一句中文夹几个标识符」是合格数据，
    按字母比会被算成「写成英文」（真机 `causechain` 的日志行就是这么被误杀的）。
    """
    assert language.han_share(text) == pytest.approx(expect)


def test_zh_side_spares_identifiers_but_not_english_prose() -> None:
    """输入：一句中文夹标识符 / 一整段英文。输出：无。

    预期行为：前者放行（它本来就是中文写的），后者照样拦下——改单位不能把真阳性一起放走。
    """
    assert language.language_mismatch("S4 /api/v3/status 403-error tr_jkl012 崩溃", "zh") is None
    assert language.language_mismatch("把 mySecretToken 写进 settings.json", "zh") is None
    assert language.language_mismatch("Sponsored by ABC", "zh") is not None


def test_latin_letters_do_not_include_other_scripts() -> None:
    """输入：无。输出：无。

    预期行为：西里尔 / 假名 / 韩文**不算拉丁字母**——`str.isalpha()` 会把它们一起算成字母，
    那样 en 侧的分母会被别种文字污染，写成俄语的产物反而「像英文」。
    """
    assert language.scripted_chars("Привет мир こんにちは 안녕하세요") == 0


@pytest.mark.parametrize(
    ("text", "locale", "flagged"),
    [
        ("这是一段写得挺长的中文正文，用来验证中文侧放行。", "zh", False),
        ("This is a reasonably long English paragraph used by the test.", "en", False),
        ("This is a reasonably long English paragraph used by the test.", "zh", True),
        ("这是一段写得挺长的中文正文，用来验证中文侧拦下。", "en", True),
    ],
)
def test_zh_and_en_are_judged_in_opposite_directions(text: str, locale: str, flagged: bool) -> None:
    """输入：中英各一段文本 × 两种语言标签。输出：无。预期行为：配不配得上标签，一报一放。"""
    assert (language.language_mismatch(text, locale) is not None) is flagged


def test_short_text_is_not_judged() -> None:
    """输入：几段短文本。输出：无。

    预期行为：有书写系统的字符不到 `MIN_SCRIPTED` 就一律放行——选项名、编号、布尔值这些短标签
    本来就推不出语言，判它只会误伤（`m1` 这种槽名混在正常样本里是常态）。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text in ["", "m1", "yes", "3", "【查询】", "alpha bravo"]:
        try:
            assert language.language_mismatch(text, "zh") is None
            assert language.language_mismatch(text, "en") is None
        except AssertionError as exc:
            bad.append(f"{(text,)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_unknown_locale_has_no_opinion() -> None:
    """输入：无。输出：无。

    预期行为：表外语言一律 None——「不知道」不等于「不一致」，把不知道报成不一致，新加一种
    语言就会全线误伤（这条是判据能安全推广到 `ja` / `fr` 的前提）。
    """
    assert language.language_mismatch("这是一段中文正文，够长了。", "ja") is None
    assert language.language_mismatch("this is english text, long enough", "fr") is None


def test_han_share_exactly_at_the_floor_passes() -> None:
    """输入：汉字占比**恰好等于** `ZH_FLOOR` 的 zh 文本。输出：无。

    预期行为：这是「下限」，等于下限算过——`share >= ZH_FLOOR` 里那个等号把「刚好够格」
    和「差一点」分开；改成 `>` 会把刚好够格的样本拒掉。真机上的占比是连续量，阈值附近
    每一分都有人踩，边界不能按「反正不会正好相等」跳过。
    """
    text = "甲 abcd efgh ijkl mnop"  # 1 个汉字 + 4 个拉丁词 = 0.20，恰好压在下限上
    assert language.han_share(text) == pytest.approx(language.ZH_FLOOR)
    assert language.language_mismatch(text, "zh") is None


def test_code_heavy_text_is_flagged_so_callers_must_not_feed_it() -> None:
    """输入：一段「中文抬头 + 一大段源码」的文本（`lint` 的 state 就长这样）。输出：无。

    预期行为：**判据本身照样报**——它只看比例，认不出「这段是代码」。这条测试把边界钉死：
    代码 / 命令 / ASCII 图为主的字段**不许喂给这个判据**（`lint` zh 侧真机占比低到 0.099、
    `shell` 是命令行、`maze` 是迷宫图），否则就是把好数据当坏数据拒。
    """
    code = "\n".join(f"const value{index} = computeSomething({index}, 'label');" for index in range(8))
    assert language.language_mismatch(f"【片段】一段源码\n{code}", "zh") is not None


def test_thresholds_match_the_measured_products() -> None:
    """输入：无。输出：无。

    预期行为：阈值与 `docs_local/可靠合成数据范式.md` 里记的实测数字一致（只统计真机产物）。
    en 侧按**字母**：正常最大 0.084（`browser`，元素名是中文固定词表）、坏样本 0.577 / 0.770。
    zh 侧按**词**（2026-10-02 改，理由见 `jc.language` 模块文档）：正常最小 0.333
    （`causechain` 的日志行）、真阳性 0.000（`adblock` 的 `Sponsored by ABC`）。
    下限夹在中间那条边界（`lint` 那种「中文抬头 + 一大段源码」，词比 0.158）与正常最小值
    之间——改阈值必须同时改证据。
    """
    assert language.ZH_FLOOR == 0.20
    assert language.EN_HAN_CEILING == 0.30
    assert language.MIN_SCRIPTED == 12
    assert language.LOCALE_SCRIPT == {"zh": "han", "en": "latin"}
    assert language.ZH_FLOOR < 0.333, "zh 侧下限不许高过正常产物的最小值（causechain 0.333）"
    assert language.ZH_FLOOR > 0.158, "zh 侧下限不许低过那条边界（lint 的代码重字段 0.158）"
    assert language.EN_HAN_CEILING > 0.084, "en 侧上限不许低过正常产物的最大值（browser 0.084）"


def test_require_language_names_the_offending_field() -> None:
    """输入：一组「一个字段写歪」的文本。输出：无。

    预期行为：抛 `DraftError` 并**点名那个字段**，而且带 `retryable_for_case`——管线据此
    重渲染一次（模型采样抖动，换个说法往往就对了），不是当生成器 bug 崩掉。
    """
    with pytest.raises(DraftError) as caught:
        language.require_language(
            {"查询": "这是一段写成中文的查询，写得足够长了，够判了。", "候选": "short"}, "en"
        )
    assert "查询" in str(caught.value)
    assert caught.value.retryable_for_case is True


def test_require_language_is_silent_when_every_field_matches() -> None:
    """输入：全部合规的字段。输出：无。预期行为：一个字都不报。"""
    language.require_language(
        {
            "查询": "这是一段写得挺长的中文查询，用来验证放行。",
            "候选": "这一条候选也写成中文，而且写得足够长。",
        },
        "zh",
    )


def test_sample_problem_accepts_any_declared_locale() -> None:
    """输入：一份声明了两种语言的产物里的一行英文。输出：无。

    预期行为：只要**有一种**声明语言对得上就算过——一次 run 会把两种语言轮转着写进同一份产物
    （`jc.pipeline` 按 `locales[seq % len]` 轮转），逐行问「配不配得上某一种」才是对的问法。
    """
    texts = {"正文": "this paragraph is written in english and it is long enough"}
    assert language.sample_problem(texts, ["zh", "en"]) is None
    assert language.sample_problem(texts, ["en", "zh"]) is None


def test_sample_problem_reports_when_no_declared_locale_matches() -> None:
    """输入：声明 zh / en，但正文是一整段中文。输出：无。

    预期行为：一种都过不了才报，而且报的是每一种声明语言各自的第一条理由（好定位）。
    """
    problem = language.sample_problem({"正文": "这一段写成了中文，而这一份产物声明的是英文。"}, ["en"])
    assert problem is not None
    assert "正文" in problem
    assert "en" in problem


def test_sample_problem_says_nothing_without_a_declared_locale() -> None:
    """输入：没有声明语言 / 只声明了表外语言。输出：无。

    预期行为：一律 None——没有声明就无从谈「与声明一致」；把「不知道」报成「不一致」会误伤
    老产物与新加的语言（这条也是判据能安全推广到 `ja` / `fr` 的前提）。
    """
    texts = {"正文": "这一段写成了中文，而这一份产物声明的是英文。"}
    assert language.sample_problem(texts, []) is None
    assert language.sample_problem(texts, ["ja", "fr"]) is None


def test_sample_problem_ignores_fields_that_are_too_short() -> None:
    """输入：一个短字段 + 一个写歪的长字段。输出：无。

    预期行为：短字段（推不出语言）不参与判定，也不该让整条样本免检——长字段照样被抓。
    """
    texts = {"选项": "m1", "正文": "这一段写成了中文，而这一份产物声明的是英文。"}
    assert language.sample_problem(texts, ["en"]) is not None
