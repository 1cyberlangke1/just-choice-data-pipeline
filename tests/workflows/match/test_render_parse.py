"""渲染与反解：版式、三种关系的写法、额外栏、报错路径。"""

from __future__ import annotations

import json

import pytest

from tests.workflows.match.conftest import _make_spec
from workflows.match import judges, locales, parse, render
from workflows.match import spec as sp

LOC = locales.load("zh")
EN = locales.load("en")


def lines_of(state: str) -> list[str]:
    """输入：state 文本。输出：去掉缩进与空行后的行。"""
    return [line.strip() for line in state.splitlines() if line.strip()]


def test_render_is_deterministic() -> None:
    """输入：同一条规格渲染两次。输出：无。预期行为：逐字节一致（渲染是纯函数）。"""
    spec = _make_spec(1)
    assert render.build(spec, LOC) == render.build(spec, LOC)


def test_layout_orders_the_field_lines() -> None:
    """输入：三种显示顺序。输出：无。

    预期行为：记录里的栏序严格按 `LAYOUT_ORDERS` 排——显示顺序只是文本，不改任何判据。
    """
    for layout, order in sp.LAYOUT_ORDERS.items():
        spec = _make_spec(2, layout=layout)
        lines = lines_of(render.build(spec, LOC))
        # 抬头一行 + 两条记录各（侧标题 + 五栏）
        assert len(lines) == 1 + 2 * (1 + len(sp.FIELDS))
        assert lines[0].startswith(LOC["meta"]["request"])
        want = [LOC["field"][sp.FIELDS[index]] for index in order]
        assert [line.split(LOC["meta"]["sep"])[0] for line in lines[2:7]] == want
        assert lines[7] == f"{LOC['meta']['right']}{LOC['meta']['sep']}"
        assert [line.split(LOC["meta"]["sep"])[0] for line in lines[8:13]] == want


def test_extra_line_is_written_only_when_asked() -> None:
    """输入：`extra = none` 与 `origin`。输出：无。

    预期行为：`origin` 时每条记录末尾多一行产地，且两侧的值不同（它是干扰项，不是可比栏）。
    """
    without = lines_of(render.build(_make_spec(3, extra="none"), LOC))
    assert not any(LOC["extra"]["origin_label"] in line for line in without)
    with_extra = lines_of(render.build(_make_spec(3, extra="origin"), LOC))
    assert len(with_extra) == len(without) + 2
    assert with_extra[7].endswith(LOC["extra"]["origin_left"])
    assert with_extra[-1].endswith(LOC["extra"]["origin_right"])


def test_same_and_variant_columns_read_as_designed() -> None:
    """输入：十二种字段关系 × 三种写法对。输出：无。

    预期行为：`same` 那一栏两侧**逐字相同**（否则会被读成 variant），`variant` 那一栏
    文本不同但归一化后相同（否则会被读成 diff），`diff` 那一栏归一化后确实不同——
    这是整个 workflow 的地基。
    """
    for pattern in sp.PATTERNS:
        for spelling in sp.SPELLINGS:
            case = sp.MatchCase("lager", pattern, spelling, "plain", "none")
            left, right = render.cells_of(case, sp.diff_pool("train"))
            for index, field in enumerate(sp.FIELDS):
                kind = pattern[index]
                first = render.cell(field, left[index], render.spelling_of(case, 0, kind), LOC)
                second = render.cell(field, right[index], render.spelling_of(case, 1, kind), LOC)
                if kind == sp.SAME:
                    assert first == second, (pattern, field)
                elif kind == sp.VARIANT:
                    assert first != second, (pattern, field)
                    assert judges.normalize(field, first, LOC) == judges.normalize(field, second, LOC)
                else:
                    assert judges.normalize(field, first, LOC) != judges.normalize(field, second, LOC)


def test_diff_columns_swap_in_another_product() -> None:
    """输入：`diff` 那一栏。输出：无。预期行为：两侧的值来自不同商品（归一化后确实不同）。"""
    for pattern in sp.PATTERNS:
        case = sp.MatchCase("cider", pattern, sp.SPELLINGS[0], "plain", "none")
        left, right = render.cells_of(case, sp.diff_pool("train"))
        for index, field in enumerate(sp.FIELDS):
            if pattern[index] == sp.DIFF:
                assert left[index] != right[index], (pattern, field)


def test_roundtrip_reads_back_every_cell() -> None:
    """输入：全部 1728 个 case × 两种语言。输出：无。

    预期行为：反解出来的每一格与渲染时写的原文逐字一致，额外栏按侧分开读回；
    没写额外栏时两侧都读到空表。
    """
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        tables = parse.tables_of(loc)
        for case in sp.build_cases(sp.MatchConfig()):
            spec = sp.sample_spec(None, case, "train", 0, 0)
            parsed = parse.parse_state(render.build(spec, loc), tables)
            assert parsed.request_id == spec.query_id
            left, right = render.cells_of(case, sp.diff_pool(spec.split))
            for side, values in enumerate((left, right)):
                got = parsed.side(side)
                for index, field in enumerate(sp.FIELDS):
                    spelling = render.spelling_of(case, side, case.pattern[index])
                    assert got[field] == render.cell(field, values[index], spelling, loc)
            if case.extra == "origin":
                assert parsed.extra_of(0)["origin"] == loc["extra"]["origin_left"]
                assert parsed.extra_of(1)["origin"] == loc["extra"]["origin_right"]
            else:
                assert parsed.extra_of(0) == {} and parsed.extra_of(1) == {}


def test_reparse_is_language_agnostic() -> None:
    """输入：两种语言渲染的同一份规格。输出：无。预期行为：合并表都能反解，槽位一致。"""
    spec = _make_spec(4, extra="origin")
    zh = parse.parse_state(render.build(spec, LOC), parse.merged_tables())
    en = parse.parse_state(render.build(spec, EN), parse.merged_tables())
    assert set(zh.left) == set(en.left) == set(sp.FIELDS)
    assert zh.request_id == en.request_id == spec.query_id
    # 额外栏的**值**随语言变（产地名本来就要本地化），能对齐的是槽位
    assert set(zh.extra_of(0)) == set(en.extra_of(0)) == {"origin"}
    assert set(zh.extra_of(1)) == set(en.extra_of(1)) == {"origin"}


def test_parse_reports_missing_request_line() -> None:
    """输入：没有抬头那一行。输出：无。预期行为：报人话。"""
    state = render.build(_make_spec(5), LOC)
    broken = "\n".join(lines_of(state)[1:])
    with pytest.raises(ValueError, match="没有抬头"):
        parse.parse_state(broken, parse.tables_of(LOC))


def test_parse_reports_missing_side() -> None:
    """输入：删掉记录 B 那一块。输出：无。预期行为：报缺哪条记录。"""
    lines = render.build(_make_spec(6), LOC).splitlines()
    with pytest.raises(ValueError, match="缺这几条记录"):
        parse.parse_state("\n".join(lines[:7]), parse.tables_of(LOC))


def test_parse_reports_missing_field() -> None:
    """输入：删掉记录 A 的某一栏。输出：无。预期行为：报哪条记录缺哪几栏。"""
    lines = render.build(_make_spec(7), LOC).splitlines()
    broken = "\n".join(lines[:3] + lines[4:])
    with pytest.raises(ValueError, match="缺这几栏"):
        parse.parse_state(broken, parse.tables_of(LOC))


def test_parse_reports_unknown_line() -> None:
    """输入：插一行认不出的文本。输出：无。预期行为：报人话（渲染漏写模板是硬伤）。"""
    broken = render.build(_make_spec(8), LOC) + "备注：随便写点\n"
    with pytest.raises(ValueError, match="认不出来"):
        parse.parse_state(broken, parse.tables_of(LOC))


def test_parse_reports_duplicate_side() -> None:
    """输入：同一个侧标题出现两次。输出：无。预期行为：报错（不然第二条记录会被悄悄覆盖）。"""
    lines = render.build(_make_spec(9), LOC).splitlines()
    broken = "\n".join([*lines, lines[1], lines[2]])
    with pytest.raises(ValueError, match="出现了两次"):
        parse.parse_state(broken, parse.tables_of(LOC))


def test_parse_reports_field_before_any_side() -> None:
    """输入：抬头之后直接写栏。输出：无。预期行为：报错（没有侧标题就不知道该进哪条记录）。"""
    lines = render.build(_make_spec(10), LOC).splitlines()
    with pytest.raises(ValueError, match="还没出现侧标题"):
        parse.parse_state("\n".join([lines[0], lines[2]]), parse.tables_of(LOC))


def test_parse_accepts_both_colon_styles() -> None:
    """输入：把全角冒号换成半角。输出：无。预期行为：两种都认（渲染与反解不共用常量）。"""
    spec = _make_spec(11)
    state = render.build(spec, LOC).replace(LOC["meta"]["sep"], ": ")
    parsed = parse.parse_state(state, parse.tables_of(LOC))
    assert parsed.request_id == spec.query_id
    assert set(parsed.left) == set(sp.FIELDS)


def test_match_label_takes_the_longest_match() -> None:
    """输入：标签之间有前缀关系的一行。输出：无。预期行为：取最长的那个标签。"""
    labels = {"名称": "field:name", "名称备注": "field:brand"}
    assert parse.match_label("名称备注：x", labels) == ("field:brand", "x")
    assert parse.match_label("名称：x", labels) == ("field:name", "x")
    assert parse.match_label("别的：x", labels) is None
    assert parse.match_label("名称x", labels) is None


def test_tables_of_rejects_extra_label_collision() -> None:
    """输入：额外栏标签与某一栏同名。输出：无。预期行为：报错（反解会把两栏认成一栏）。"""
    loc = json.loads(json.dumps(locales.load("zh")))
    loc["extra"]["origin_label"] = loc["field"]["name"]
    with pytest.raises(ValueError, match="重名"):
        parse.tables_of(loc)


def test_tables_of_rejects_identical_side_labels() -> None:
    """输入：两条记录的侧标题同名。输出：无。预期行为：报错。"""
    loc = json.loads(json.dumps(locales.load("zh")))
    loc["meta"]["right"] = loc["meta"]["left"]
    with pytest.raises(ValueError, match="侧标题"):
        parse.tables_of(loc)


def test_merged_tables_rejects_cross_language_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两种语言共用同一个栏标签。输出：无。预期行为：报错而不是静默取其一。"""
    zh = locales.load("zh")
    en = json.loads(json.dumps(locales.load("en")))
    en["field"]["name"] = zh["field"]["name"]
    monkeypatch.setattr(locales, "load", lambda lang: zh if lang == "zh" else en)
    with pytest.raises(ValueError, match="多种语言"):
        parse.merged_tables()


def test_merged_tables_rejects_duplicate_request_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两种语言的抬头标签相同。输出：无。预期行为：报错。"""
    zh = locales.load("zh")
    en = json.loads(json.dumps(locales.load("en")))
    en["meta"]["request"] = zh["meta"]["request"]
    monkeypatch.setattr(locales, "load", lambda lang: zh if lang == "zh" else en)
    with pytest.raises(ValueError, match="抬头标签"):
        parse.merged_tables()


def test_merged_tables_rejects_cross_language_side_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """输入：两种语言共用同一个侧标题。输出：无。预期行为：报错。"""
    zh = locales.load("zh")
    en = json.loads(json.dumps(locales.load("en")))
    en["meta"]["left"] = zh["meta"]["left"]
    monkeypatch.setattr(locales, "load", lambda lang: zh if lang == "zh" else en)
    with pytest.raises(ValueError, match="侧标题"):
        parse.merged_tables()


def test_held_out_product_never_appears_on_the_id_side() -> None:
    """输入：同分布侧的全部 case × 两种语言，以及 OOD 侧的全部 case。输出：无。

    预期行为：留出商品的名称与品牌在**同分布侧渲染出来的文本**里一个都不出现——「值留出」
    要成立就得在文本上也成立；只锁 case 维度的话，`tea` 的 `diff` 那一栏会顺着目录绕到留出
    商品身上（实测训练侧 604 条里 57 条带留出商品的名字）。反向断言 OOD 侧**确实**写出来，
    免得留出悄悄退化成「两边都没有」。
    """
    held = sp.PRODUCTS[-1]
    id_cases = sp.build_cases(sp.MatchConfig(products=sp.ID_PRODUCTS, spellings=sp.ID_SPELLINGS))
    ood_cases = sp.build_cases(sp.MatchConfig(ood_only=True))
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        forbidden = set(loc["name"][held]) | set(loc["brand"][held])
        for case in id_cases:
            state = render.build(sp.sample_spec(None, case, "train", 0, 0), loc)
            hit = sorted(token for token in forbidden if token in state)
            assert not hit, (lang, case.describe(), hit)
        ood_states = [
            render.build(sp.sample_spec(None, case, "test_ood", 0, 0), loc) for case in ood_cases
        ]
        assert any(any(token in state for token in forbidden) for state in ood_states), (
            f"{lang}：OOD 侧一条都没写出留出商品的名字，留出可能退化了"
        )
