"""渲染与反解：词表与对话逐格对得上、坏 state 当场报人话、两种语言都走得通。"""

from __future__ import annotations

from typing import Final

import pytest

from tests.workflows.sentiment.conftest import _make_spec
from workflows.sentiment import features as ft
from workflows.sentiment import locales, parse, render

LANGS: Final[tuple[str, ...]] = ("zh", "en")
ALL_CASES: Final[tuple[ft.SentimentCase, ...]] = tuple(ft.build_cases(512))


def state_of(case: ft.SentimentCase, locale: str = "zh") -> str:
    """输入：一个 case、语言。输出：渲染出来的 `state` 全文。"""
    return render.build(_make_spec(case.arc, case.topic), locales.load(locale))


def _state(
    turn_lines: list[str],
    lexicon: list[str] | None = None,
    topic_line: str | None = None,
    locale: str = "zh",
) -> str:
    """输入：对话行原文、可选的词表行与场景行、语言。输出：一段最小 state。

    预期行为：坏 state 的测试自己拼文本，不去篡改渲染结果——这样「渲染坏了」与
    「反解太宽松」是两条独立的证据链。
    """
    loc = locales.load(locale)
    meta = loc["meta"]
    lines = [f"{meta['request']} E-TEST"]
    lines.append(topic_line if topic_line is not None else f"[{meta['topic']}] 客服对话")
    lines.append(f"[{meta['words']}]")
    lines.extend(render.lexicon_lines(loc) if lexicon is None else lexicon)
    lines.append(f"[{meta['turns']}]")
    lines.extend(turn_lines)
    lines.append(f"[{meta['rules']}]")
    lines.append(f"{meta['rules_prefix']}规则一")
    return "\n".join(lines) + "\n"


def _turns(locale: str = "zh", word: str = "还不错", count: int = ft.TURNS) -> list[str]:
    """输入：语言、每回合用的词、回合数。输出：对话区块的每一行（说话人交替）。"""
    return [
        f"  {index + 1}. {render.speaker_text(locales.load(locale), 'support', index)} | {word}"
        for index in range(count)
    ]


def test_render_layout_is_the_expected_shape() -> None:
    """输入：全程中性的弧线、第一个场景。输出：无。

    预期行为：抬头一行、场景一行、词表五档、对话五回合（编号 1 起、两格）、规则段五行。
    """
    loc = locales.load("zh")
    meta = loc["meta"]
    lines = state_of(ft.SentimentCase(arc=10, topic="support")).splitlines()
    assert lines[0].startswith(str(meta["request"]))
    assert lines[1] == f"[{meta['topic']}] 客服对话"
    assert lines[2] == f"[{meta['words']}]"
    assert lines[3:8] == [
        "  +2 | 特别满意 / 非常开心 / 赞不绝口 / 拍手叫好",
        "  +1 | 还不错 / 有点高兴 / 可以接受 / 挺顺的",
        "  0 | 没什么感觉 / 一般般 / 说不上好 / 没什么波动",
        "  -1 | 有点烦 / 不太满意 / 有点失望 / 心里别扭",
        "  -2 | 非常生气 / 彻底失望 / 火冒三丈 / 心灰意冷",
    ]
    assert lines[8] == f"[{meta['turns']}]"
    # 取词下标 = (弧线下标 + 回合号) % 4，所以第 10 条弧线取到的是 2/3/0/1/2 这四个位置的词。
    words = ["说不上好", "没什么波动", "没什么感觉", "一般般", "说不上好"]
    assert lines[9:14] == [
        f"  {index + 1}. {('客户' if index % 2 == 0 else '客服')} | {words[index]}"
        for index in range(5)
    ]
    assert lines[14] == f"[{meta['rules']}]"
    assert len(lines[15:]) == len(loc["rules"]["lines"])


def test_render_is_deterministic() -> None:
    """输入：同一个 case 渲染两次。输出：无。预期行为：逐字节一致（纯函数，不掷骰子）。"""
    case = ft.SentimentCase(arc=3, topic="review")
    assert state_of(case) == state_of(case)
    assert state_of(case, "en") == state_of(case, "en")


def test_speaker_alternates_and_word_slot_is_bounded() -> None:
    """输入：几个场景与回合。输出：无。预期行为：两个称呼交替出现，取词下标不越界。"""
    loc = locales.load("zh")
    for topic in ft.TOPICS:
        speakers = [render.speaker_text(loc, topic, index) for index in range(ft.TURNS)]
        assert speakers[0] != speakers[1]
        assert speakers == [speakers[index % 2] for index in range(ft.TURNS)]
    assert render.word_text(loc, 2, 99) == render.word_text(loc, 2, 99 % ft.WORD_SLOTS)


def test_topic_and_word_lookups_report_bad_keys() -> None:
    """输入：语言表里没有的场景键 / 档位。输出：无。预期行为：报错，不悄悄退回键名。"""
    loc = locales.load("zh")
    with pytest.raises(ValueError, match="不在语言表"):
        render.topic_text(loc, "nope")
    with pytest.raises(ValueError, match="不在语言表"):
        render.word_text(loc, 9, 0)


@pytest.mark.parametrize("lang", LANGS)
def test_roundtrip_is_exact_for_every_case(lang: str) -> None:
    """输入：全空间 case × 两种语言。输出：无。

    预期行为：反解出来的场景、词表、每个回合的分值、请求号与规则段与真值规格逐项一致——
    这是「渲染把某个词写歪了」那条路的第一道闸。
    """
    loc = locales.load(lang)
    tables = parse.tables_of(loc)
    for case in ALL_CASES:
        spec = _make_spec(case.arc, case.topic)
        parsed = parse.parse_state(state_of(case, lang), tables)
        assert parsed.levels == case.levels, case.describe()
        assert parsed.topic == render.topic_text(loc, case.topic)[0]
        assert parsed.count == ft.TURNS
        assert parsed.request_id == spec.query_id
        assert parsed.rules == tuple(str(line) for line in loc["rules"]["lines"])


def test_parse_reads_a_minimal_state() -> None:
    """输入：手拼的最小 state。输出：无。预期行为：五个回合、档位与词表都对得上。"""
    parsed = parse.parse_state(_state(_turns()), parse.merged_tables())
    assert parsed.request_id == "E-TEST"
    assert parsed.topic == "客服对话"
    assert parsed.levels == (1, 1, 1, 1, 1)
    assert parsed.turns[0].speaker == "客户"
    assert set(parsed.words) == set(ft.LEVELS)


def test_parse_reads_the_negative_levels() -> None:
    """输入：全程用 -2 档的词。输出：无。预期行为：带负号的档位反解回负数。"""
    parsed = parse.parse_state(_state(_turns(word="火冒三丈")), parse.merged_tables())
    assert parsed.levels == (-2,) * ft.TURNS


def test_parse_skips_blank_lines_inside_a_section() -> None:
    """输入：对话行之间夹空行。输出：无。预期行为：空行跳过，编号仍然连续。"""
    lines = _turns()
    lines.insert(1, "")
    assert parse.parse_state(_state(lines), parse.merged_tables()).count == ft.TURNS


@pytest.mark.parametrize(
    ("turns", "message"),
    [
        ([], "一个回合都没有"),
        (["  2. 客户 | 还不错"], "跳号"),
        (["  1. 客户"], "格"),
        (
            ["  1. 客户 | 乱写", *[f"  {index + 2}. 客服 | 还不错" for index in range(4)]],
            "情绪词不在词表里",
        ),
    ],
)
def test_parse_reports_broken_turn_rows(turns: list[str], message: str) -> None:
    """输入：各种写歪的对话行。输出：无。预期行为：每一条都报人话，不静默算出一个假答案。"""
    with pytest.raises(ValueError, match=message):
        parse.parse_state(_state(turns), parse.merged_tables())


def test_parse_reports_a_wrong_turn_count() -> None:
    """输入：只有四个回合。输出：无。预期行为：报出应有几个回合。"""
    with pytest.raises(ValueError, match="应为 5 个"):
        parse.parse_state(_state(_turns(count=4)), parse.merged_tables())


def _lexicon_with(locale: str = "zh", index: int = 0, text: str = "") -> list[str]:
    """输入：语言、要替换的行号（0 = +2 档）、替换后的原文。输出：改过的词表行。"""
    lines = render.lexicon_lines(locales.load(locale))
    lines[index] = text
    return lines


@pytest.mark.parametrize(
    ("lexicon", "message"),
    [
        (None, "词表缺档位"),
        ("dup_level", "出现了两次"),
        ("bad_level", "档位不在"),
        ("no_sep", "缺少格分隔符"),
        ("empty", "词为空"),
        ("dup_word", "词表里有重复的词"),
        ("garbage", "认不出的词表行"),
    ],
)
def test_parse_reports_broken_lexicon_rows(lexicon: str | None, message: str) -> None:
    """输入：各种写歪的词表。输出：无。预期行为：每一条都报人话——词表是判据的一部分。"""
    loc = locales.load("zh")
    if lexicon is None:
        lines = render.lexicon_lines(loc)[:4]
    elif lexicon == "dup_level":
        lines = _lexicon_with(index=1, text=render.lexicon_lines(loc)[0])
    elif lexicon == "bad_level":
        lines = _lexicon_with(text="  +9 | 乱写")
    elif lexicon == "no_sep":
        lines = _lexicon_with(text="  +2 特别满意")
    elif lexicon == "empty":
        lines = _lexicon_with(text="  +2 | ")
    elif lexicon == "dup_word":
        lines = _lexicon_with(index=1, text="  +1 | 特别满意 / 有点高兴 / 可以接受 / 挺顺的")
    else:
        lines = _lexicon_with(text="乱写一行")
    with pytest.raises(ValueError, match=message):
        parse.parse_state(_state(_turns(), lexicon=lines), parse.merged_tables())


def test_parse_reports_a_missing_or_empty_topic() -> None:
    """输入：场景抬头后面没有场景名 / 整段没有场景抬头。输出：无。预期行为：两种都报人话。"""
    with pytest.raises(ValueError, match="没有场景名"):
        parse.parse_state(_state(_turns(), topic_line="[场景]"), parse.merged_tables())
    text = _state(_turns()).replace("[场景] 客服对话\n", "")
    with pytest.raises(ValueError, match="没有场景抬头"):
        parse.parse_state(text, parse.merged_tables())


def test_parse_reports_a_broken_header_and_short_input() -> None:
    """输入：抬头认不出的文本、太短的文本、非字符串。输出：无。预期行为：三种都当场报人话。"""
    with pytest.raises(ValueError, match="抬头认不出来"):
        parse.parse_state(_state(_turns()).replace("情绪弧线请求号", "乱写"), parse.merged_tables())
    with pytest.raises(ValueError, match="太短"):
        parse.parse_state("a\nb\nc", parse.merged_tables())
    with pytest.raises(ValueError, match="不是字符串"):
        parse.parse_state(123, parse.merged_tables())


def test_parse_reports_content_before_any_section() -> None:
    """输入：区块抬头之前就有一行内容。输出：无。预期行为：报错——那说明渲染漏了抬头。"""
    text = "情绪弧线请求号 E-TEST\n乱写一行\n[对话]\n  1. 客户 | 还不错\n"
    with pytest.raises(ValueError, match="任何区块抬头之前"):
        parse.parse_state(text, parse.merged_tables())


def test_tables_of_rejects_empty_labels() -> None:
    """输入：抬头 / 场景抬头 / 两种分隔符被写成空白。输出：无。预期行为：当场报错。"""
    for key, message in (
        ("request", "抬头不能是空串"),
        ("topic", "场景抬头不能是空串"),
        ("cell_sep", "格分隔符不能是空白"),
        ("word_sep", "词分隔符不能是空白"),
    ):
        loc = dict(locales.load("zh"))
        loc["meta"] = dict(locales.load("zh")["meta"], **{key: " "})
        with pytest.raises(ValueError, match=message):
            parse.tables_of(loc)


def test_merged_tables_cover_both_languages() -> None:
    """输入：无。输出：无。

    预期行为：合并表收齐两种语言的抬头、场景抬头、区块抬头与三种结构词元
    ——`Workflow.reparse` 没有语言参数，全靠它认行。
    """
    tables = parse.merged_tables()
    assert set(tables.request) == {"情绪弧线请求号", "Sentiment arc request"}
    assert set(tables.topic_labels) == {"场景", "Scene"}
    assert set(tables.sections) == {
        "[情绪词表]",
        "[对话]",
        "[判读规则]",
        "[Emotion words]",
        "[Dialogue]",
        "[Reading rules]",
    }
    assert tables.cell_sep == "|" and tables.word_sep == "/" and tables.rules_prefix == "- "


def test_merged_tables_reparse_works_for_both_languages() -> None:
    """输入：两种语言渲染的同一个 case。输出：无。预期行为：合并表能把两边都反解成同一串分值。"""
    case = ft.SentimentCase(arc=8, topic="neighbors")
    tables = parse.merged_tables()
    zh = parse.parse_state(state_of(case, "zh"), tables)
    en = parse.parse_state(state_of(case, "en"), tables)
    assert zh.levels == en.levels == case.levels
