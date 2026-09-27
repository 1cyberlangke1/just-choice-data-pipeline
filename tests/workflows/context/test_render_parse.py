"""渲染与反解：版式、纯函数性、往返一致，以及反解的每一条报错分支。"""

from __future__ import annotations

import random

import pytest

from workflows.context import locales, parse, render
from workflows.context import spec as sp
from workflows.context.workflow import ContextWorkflow

WF = ContextWorkflow()
LOC = locales.load("zh")
LABELS = parse.labels_of(LOC)


def _spec(**overrides: object) -> sp.SessionSpec:
    """输入：case 覆盖项。输出：一条真值规格（固定种子，测试里可复现）。"""
    fields = {
        "keep_form": "err_small",
        "drop_form": "off_small",
        "kind_set": "read_run",
        "extra": "keep",
        "order": "p0",
        "phrasing": 0,
    }
    fields.update(overrides)
    return sp.sample_spec(sp.SessionCase(**fields), "train", 0, 0, random.Random(3))  # type: ignore[arg-type]


def test_render_shape() -> None:
    """输入：一条规格。输出：无。

    预期行为：标题 → 会话号 → 预算 → 当前任务 → 空行 → 五个块（抬头 + 正文），
    抬头行里四格按固定顺序、用语言文件的分隔符连起来。
    """
    spec = _spec()
    text = render.build(spec, LOC)
    lines = text.splitlines()
    meta = LOC["meta"]
    assert lines[0] == meta["title"]
    assert lines[1] == f"{meta['session_label']}{meta['colon']}{spec.session_id}"
    assert lines[2] == f"{meta['budget_label']}{meta['colon']}{spec.budget} {meta['token_unit']}"
    assert lines[3] == f"{meta['task_label']}{meta['colon']}{meta['task_text']}"
    assert lines[4] == ""
    assert len(lines) == 5 + 2 * len(spec.blocks)
    for index, block in enumerate(spec.blocks, start=1):
        header = lines[5 + 2 * (index - 1)]
        assert header == render.block_header(index, block, LOC)
        assert header.startswith(f"[{index}] ")
        assert len(header.split(meta["block_sep"])) == 4
        assert lines[6 + 2 * (index - 1)] == render.block_content(block, LOC)
    assert "{" not in text and "}" not in text


def test_render_is_a_pure_function() -> None:
    """输入：同一份规格渲染两次。输出：无。预期行为：逐字节一致。"""
    spec = _spec(order="p3")
    assert render.build(spec, LOC) == render.build(spec, LOC)


def test_render_reports_a_missing_content_template() -> None:
    """输入：把正文模板表挖掉一格。输出：无。

    预期行为：报人话并点名那个键——静默渲染出空行会让「哪一块没写正文」变得难查。
    """
    spec = _spec()
    broken = dict(LOC)
    broken["content"] = dict(LOC["content"])
    key = f"{spec.blocks[0].kind}_{spec.blocks[0].size}_{spec.blocks[0].age}_{spec.blocks[0].topic}"
    del broken["content"][key]
    with pytest.raises(KeyError, match="没有正文模板"):
        render.build(spec, broken)


def test_roundtrip_over_the_whole_training_grid() -> None:
    """输入：训练侧整张 case 矩阵。输出：无。

    预期行为：渲染 → 反解之后，会话号、预算与五个块的四格全都与真值一致——
    这是「state 能读回来」这条的验收，覆盖到每一种类型 / 大小 / 年龄 / 主题的组合。
    """
    cases = sp.build_cases(WF.config_for("train"))
    assert cases
    rng = random.Random(20260927)
    for index, case in enumerate(cases):
        spec = sp.sample_spec(case, "train", index, case.phrasing, rng)
        session = parse.parse_session(render.build(spec, LOC), LABELS)
        assert session.session_id == spec.session_id
        assert session.budget == spec.budget
        assert [block.describe() for block in session.blocks] == [
            block.describe() for block in spec.blocks
        ]
        assert [block.content for block in session.blocks] == [
            render.block_content(block, LOC).strip() for block in spec.blocks
        ]


def test_roundtrip_works_through_the_merged_table() -> None:
    """输入：中英两门语言的 state。输出：无。

    预期行为：合并表吃得下两门语言——`Workflow.reparse` 没有语言参数，只靠它。
    """
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        spec = sp.sample_spec(
            sp.SessionCase("err_small", "stale", "run_ask", "digest", "p4", 0),
            "train",
            0,
            0,
            random.Random(1),
        )
        session = WF.reparse(render.build(spec, loc))
        assert session.session_id == spec.session_id
        assert [block.kind for block in session.blocks] == [
            block.kind for block in spec.blocks
        ]


def test_parse_rejects_a_non_string_state() -> None:
    """输入：非字符串。输出：无。预期行为：报类型错（人话，不是 AttributeError）。"""
    with pytest.raises(TypeError, match="state 必须是字符串"):
        parse.parse_session(123, LABELS)  # type: ignore[arg-type]


def test_parse_rejects_a_missing_session_id() -> None:
    """输入：挖掉会话号那一格。输出：无。预期行为：报人话。"""
    text = render.build(_spec(), LOC)
    broken = "\n".join(line for line in text.splitlines() if not line.startswith("会话号"))
    with pytest.raises(ValueError, match="缺这几格"):
        parse.parse_session(broken, LABELS)


def test_parse_rejects_a_missing_budget() -> None:
    """输入：挖掉预算那一格。输出：无。预期行为：报人话。"""
    text = render.build(_spec(), LOC)
    broken = "\n".join(line for line in text.splitlines() if not line.startswith("预算"))
    with pytest.raises(ValueError, match="缺这几格"):
        parse.parse_session(broken, LABELS)


@pytest.mark.parametrize("value", ["1024", "一千 token", "abc token"])
def test_parse_rejects_a_bad_budget(value: str) -> None:
    """输入：缺单位 / 不是数字 / 混合的预算。输出：无。预期行为：报人话。"""
    text = render.build(_spec(), LOC)
    broken = "\n".join(
        f"预算：{value}" if line.startswith("预算") else line for line in text.splitlines()
    )
    with pytest.raises(ValueError, match="预算"):
        parse.parse_session(broken, LABELS)


def test_parse_rejects_a_header_without_a_bracket() -> None:
    """输入：抬头行少了右方括号。输出：无。预期行为：报人话。"""
    text = render.build(_spec(), LOC).replace("[1] ", "[1 ", 1)
    with pytest.raises(ValueError, match="缺右方括号"):
        parse.parse_session(text, LABELS)


def test_parse_rejects_a_header_with_the_wrong_cell_count() -> None:
    """输入：抬头行少一格。输出：无。预期行为：报人话并点明应该是几格。"""
    text = render.build(_spec(), LOC)
    header = next(line for line in text.splitlines() if line.startswith("[1] "))
    cells = header.split(" · ")
    broken = text.replace(header, " · ".join(cells[:-1]), 1)
    with pytest.raises(ValueError, match="应该是 4 格"):
        parse.parse_session(broken, LABELS)


def test_parse_rejects_an_unknown_cell_value() -> None:
    """输入：把类型那一格换成认不出的词。输出：无。预期行为：报人话并点名那一维。"""
    text = render.build(_spec(), LOC)
    broken = text.replace("用户提问 · ", "天上掉馅饼 · ", 1)
    with pytest.raises(ValueError, match="kind 认不出来"):
        parse.parse_session(broken, LABELS)


def test_parse_rejects_a_block_without_a_body_line() -> None:
    """输入：最后一块没有正文行。输出：无。预期行为：报人话（那是渲染硬伤）。"""
    lines = render.build(_spec(), LOC).splitlines()
    with pytest.raises(ValueError, match="缺正文行"):
        parse.parse_session("\n".join(lines[:-1]), LABELS)


def test_parse_rejects_a_state_without_blocks() -> None:
    """输入：只有头部几行的 state。输出：无。预期行为：报人话。"""
    lines = render.build(_spec(), LOC).splitlines()
    with pytest.raises(ValueError, match="一个块都没有"):
        parse.parse_session("\n".join(lines[:4]), LABELS)


def test_parse_ignores_unknown_lines() -> None:
    """输入：夹了几行认不出的文本。输出：无。预期行为：忽略它们，块照样读出来。"""
    text = render.build(_spec(), LOC)
    noisy = text.replace("\n[1] ", "\n今天天气不错。\n[1] ", 1)
    session = parse.parse_session(noisy, LABELS)
    assert len(session.blocks) == 5


def test_labels_of_rejects_duplicate_texts() -> None:
    """输入：把两个类型的文案改成一样。输出：无。预期行为：报人话（重复会让反解有歧义）。"""
    broken = dict(LOC)
    broken["kind"] = dict(LOC["kind"])
    broken["kind"]["read"] = LOC["kind"]["ask"]
    with pytest.raises(ValueError, match="文案重复"):
        parse.labels_of(broken)


def test_labels_accessors_reject_unknown_names() -> None:
    """输入：不存在的属性名 / 标签名。输出：无。预期行为：都报人话。"""
    with pytest.raises(KeyError, match="没有属性"):
        LABELS.table("nope")
    with pytest.raises(KeyError, match="没有标签"):
        LABELS.labels_of("nope")
