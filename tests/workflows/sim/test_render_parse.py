"""渲染与反解的测试：版式、逐格往返、合并表、以及各类硬伤要报人话。"""

from __future__ import annotations
from jc.state_parse import read_field_value


import pytest

from tests.workflows.sim.conftest import _make_spec
from workflows.sim import locales, parse, render, spec as sp


def test_layout_is_fixed() -> None:
    """输入：一份规格。输出：无。

    预期行为：标题 / 单号 / 回合 / 资源 / 战利品 / 空行 / 六行四格——抬头三格与六行动作
    就是判据的全部输入，缺一行都不行。
    """
    spec = _make_spec()
    loc = locales.load("zh")
    labels = loc["label"]
    lines = render.build(spec, loc).splitlines()
    assert lines[0] == loc["meta"]["title"]
    colon = loc["meta"]["colon"]
    assert lines[1] == f"{labels['request']}{colon}{spec.request_id}"
    assert lines[2] == f"{labels['turn']}{colon}{spec.turn}"
    assert lines[3] == f"{labels['resources']}{colon}{spec.resources}"
    assert lines[4] == f"{labels['haul']}{colon}{spec.haul}"
    assert lines[5] == ""
    assert len(lines) == 6 + len(sp.ACTIONS)
    for index, (line, row) in enumerate(zip(lines[6:], spec.rows), start=1):
        assert line == render.row_line(index, row, loc)


def test_row_line_joins_the_four_cells() -> None:
    """输入：局面的一行。输出：无。预期行为：序号 + 四格（动作 · 耗 · 收益 · 冷却）。"""
    loc = locales.load("zh")
    row = sp.rows_of(sp.board_of("b1"), "l2")[0]
    assert render.row_line(1, row, loc) == "[1] " + str(loc["meta"]["cell_sep"]).join(
        (
            loc["action"][row.action],
            loc["cost"][row.cost],
            loc["gain"][row.gain],
            loc["cooldown"][row.cooldown],
        )
    )
    assert render.field_line("单号", "值", loc) == "单号" + str(loc["meta"]["colon"]) + "值"


def test_roundtrip_over_the_whole_grid_in_both_languages() -> None:
    """输入：全部 288 个 case × 两门语言。输出：无。

    预期行为：渲染出来的文本反解回来与真值逐格一致——「渲染写歪」会在这里当场现形。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        labels = parse.labels_of(loc)
        for case in sp.build_cases(sp.SimConfig()):
            spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
            board = parse.parse_board(render.build(spec, loc), labels)
            assert board.request_id == spec.request_id
            assert (board.turn, board.resources, board.haul) == (
                spec.turn,
                spec.resources,
                spec.haul,
            )
            assert [
                (row.action, row.cost, row.gain, row.cooldown) for row in board.rows
            ] == [
                (row.action, row.cost, row.gain, row.cooldown) for row in spec.rows
            ]


def test_merged_table_reads_every_language() -> None:
    """输入：两门语言渲染出来的同一份规格。输出：无。

    预期行为：合并表（`reparse` 用的那张）能把两门语言都反解成同一份键——语言无关的反解
    靠的就是这个。
    """
    spec = _make_spec(board="b6", layout="l3", phrasing=1)
    merged = parse.merged()
    results: set[tuple[str, int, int, int, tuple[tuple[str, str, str, str], ...]]] = set()
    for lang in ("zh", "en"):
        board = parse.parse_board(render.build(spec, locales.load(lang)), merged)
        results.add(
            (
                board.request_id,
                board.turn,
                board.resources,
                board.haul,
                tuple((row.action, row.cost, row.gain, row.cooldown) for row in board.rows),
            )
        )
    assert len(results) == 1
    request_id, turn, resources, haul, rows = results.pop()
    assert (request_id, turn, resources, haul) == (
        spec.request_id,
        spec.turn,
        spec.resources,
        spec.haul,
    )
    assert rows == tuple(
        (row.action, row.cost, row.gain, row.cooldown) for row in spec.rows
    )


def test_field_value_prefers_the_longest_label() -> None:
    """输入：两个互为前缀的候选标签。输出：无。

    预期行为：取最长的那个——否则「单」会把「单号」那一格提前吃掉。
    """
    assert read_field_value("单号：S-1", ("单", "单号")) == "S-1"
    assert read_field_value("单：S-1", ("单", "单号")) == "S-1"


def test_field_value_needs_a_separator_right_after_the_label() -> None:
    """输入：以标签开头但没有分隔符的一行。输出：无。

    预期行为：不算这一格——标题行确实不是「标签 + 分隔符」开头，收紧才不会误认。
    """
    assert read_field_value("局面", ("回合",)) is None
    assert read_field_value("回合 3", ("回合",)) is None
    assert read_field_value("今天天气不错。", ("回合",)) is None
    assert read_field_value("Round: 3", ("Round",)) == "3"


def test_parse_cells_reports_hard_damage() -> None:
    """输入：缺右括号 / 格数不对 / 认不出的格 的局面行。输出：无。预期行为：都报人话。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(ValueError, match="右方括号"):
        parse._parse_cells("1 前进 · 耗 1 · 收益 2 · 冷却 0", labels)
    with pytest.raises(ValueError, match="格"):
        parse._parse_cells("[1] 前进 · 耗 1 · 收益 2", labels)
    with pytest.raises(ValueError, match="action"):
        parse._parse_cells("[1] 不存在 · 耗 1 · 收益 2 · 冷却 0", labels)


def test_parse_number_reports_bad_values() -> None:
    """输入：不是整数的抬头格 / 出了取值域的抬头格。输出：无。预期行为：都报人话。"""
    with pytest.raises(ValueError, match="不是整数"):
        parse._parse_number("turn", "四")
    with pytest.raises(ValueError, match="出了取值域"):
        parse._parse_number("turn", "9")
    assert parse._parse_number("haul", "5") == 5


def test_parse_board_reports_hard_damage() -> None:
    """输入：四种坏文本。输出：无。预期行为：各报人话，不静默猜。"""
    loc = locales.load("zh")
    labels = parse.labels_of(loc)
    good = render.build(_make_spec(), loc).splitlines()
    with pytest.raises(TypeError, match="字符串"):
        parse.parse_board(123, labels)
    with pytest.raises(ValueError, match="缺这几格"):
        parse.parse_board("\n".join([*good[:1], *good[2:]]), labels)
    with pytest.raises(ValueError, match="局面应该是"):
        parse.parse_board("\n".join(good[:5]), labels)
    with pytest.raises(ValueError, match="action"):
        broken = list(good)
        broken[6] = broken[6].replace(str(loc["action"]["advance"]), "不存在")
        parse.parse_board("\n".join(broken), labels)


def test_parse_board_ignores_unknown_lines() -> None:
    """输入：混进两行噪声的局面。输出：无。预期行为：噪声被忽略，其余照常读回。"""
    labels = parse.labels_of(locales.load("zh"))
    good = render.build(_make_spec(), locales.load("zh"))
    noisy = "\n".join(["随便一行说明", *good.splitlines(), "再来一行"])
    board = parse.parse_board(noisy, labels)
    assert len(board.rows) == len(sp.ACTIONS)


def test_labels_table_lookups_report_unknown_fields() -> None:
    """输入：两个不存在的字段名。输出：无。预期行为：都报 KeyError 并列出可用值。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(KeyError, match="词表"):
        labels.table("nope")
    with pytest.raises(KeyError, match="标签"):
        labels.labels_of("nope")
    assert labels.labels_of("turn") == labels.labels_of("turn")
    assert labels.table("cost") == labels.cost
    assert labels.table("gain") == labels.gain


def test_labels_of_rejects_two_keys_sharing_one_text() -> None:
    """输入：把两档耗写成同一个原文。输出：无。预期行为：报人话（反解会认错键）。"""
    loc = dict(locales.load("zh"))
    loc["cost"] = dict(loc["cost"])
    loc["cost"]["c2"] = loc["cost"]["c1"]
    with pytest.raises(ValueError, match="共用同一个原文"):
        parse.labels_of(loc)


def _patched_merged(monkeypatch: pytest.MonkeyPatch, mutate: object) -> None:
    """输入：monkeypatch、一个改语言字典的回调。输出：无。

    预期行为：把两门语言换成「zh + 改过的 en」，好单独验合并表的三种冲突。
    """
    zh = locales.load("zh")
    en = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in locales.load("en").items()
    }
    mutate(en, zh)
    monkeypatch.setattr(locales, "available", lambda: ["zh", "en"])
    monkeypatch.setattr(locales, "load", lambda lang: {"zh": zh, "en": en}[lang])


def test_merged_rejects_colliding_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言的抬头字段名写成一样。输出：无。预期行为：报人话（合并表有歧义）。"""
    _patched_merged(
        monkeypatch,
        lambda en, zh: en["label"].update({"request": zh["label"]["request"]}),
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_rejects_colliding_vocabularies(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言把同一个动作写成同一个词。输出：无。预期行为：报人话。"""
    _patched_merged(
        monkeypatch,
        lambda en, zh: en["action"].update({"advance": zh["action"]["advance"]}),
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()
