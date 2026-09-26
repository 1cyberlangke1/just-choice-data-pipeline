"""渲染与反解的测试：两种版式、语言无关、草稿校验、提示词不泄骨架。"""

from __future__ import annotations

import dataclasses
import random
import re
from typing import Any

import pytest

from jc.backend.types import DraftInvalidError
from workflows.hybrid import locales, render
from workflows.hybrid import spec as sp

BULLET_START = re.compile(r"^\d+\. \d{2}:00–\d{2}:00 · .+ · .+$")


def _render(case: sp.HybridCase, lang: str, seed: int = 5) -> sp.HybridSpec:
    """输入：case、语言、种子。输出：文本已填好并渲染过的规格。

    预期行为：`written` 由 `check_draft` 填好（走生产同一条路），返回的规格可直接 `build`。
    """
    spec = sp.sample_spec(random.Random(seed), case, "train", 0, 0, 0)
    loc = locales.load(lang)
    backend = _fake()
    draft = backend.generate(render.build_task(spec, loc, spec.text_task_id))
    spec.written.update(render.check_draft(spec, loc, draft.data))
    return spec


def _fake(**kwargs: Any) -> Any:
    """输入：假后端开关。输出：假后端实例。预期行为：延迟导入 conftest 里的类。"""
    from tests.workflows.hybrid.conftest import FakeBackend

    return FakeBackend(**kwargs)


def test_roundtrip_for_every_layout_and_language() -> None:
    """输入：全枚举 case × 两种语言。输出：无。

    预期行为：`build` 出来的日程必须被 `parse_state` **逐字段**读回原样——
    渲染与反解是同一枚硬币的两面，错位就会让判别器把好数据判坏（或反过来）。
    """
    bad: list[str] = []
    total = 0
    for case in sp.build_cases(sp.HybridConfig()):
        for lang in locales.available():
            spec = _render(case, lang)
            loc = locales.load(lang)
            state = render.build(spec, loc)
            parsed = render.parse_state(state)
            total += 1
            if parsed.date != spec.date:
                bad.append(f"{lang} {case.describe()}: 日期 {parsed.date} != {spec.date}")
            want = render.expected_meetings(spec, loc)
            if list(parsed.meetings) != want:
                bad.append(f"{lang} {case.describe()}: 会议对不上\n{state}\n{parsed.meetings}\n{want}")
    assert total > 0
    assert bad[:3] == []


def test_bullet_layout_shape() -> None:
    """输入：列表版式的一份日程。输出：文本行。

    预期行为：抬头一行 + 空行 + 每场会三行（起始行 / 标题 / 议题），
    起始行是「序号. 起–止 · 会议室 · 主持人」，文本行带缩进与标签。
    """
    spec = _render(sp.HybridCase(3, "early", "none", "bullet", 3, "short"), "zh")
    lines = render.build(spec, locales.load("zh")).splitlines()
    assert lines[0].startswith("会议日程 · ") and lines[1] == ""
    body = lines[2:]
    assert len(body) == 3 * 3
    for index in range(3):
        start, title, agenda = body[index * 3 : index * 3 + 3]
        assert BULLET_START.match(start), start
        assert title.startswith("   标题：") and agenda.startswith("   议题：")
    first = spec.meetings[0]
    assert body[0].startswith(f"1. {render.time_text(first.start, locales.load('zh'))}–")


def test_table_layout_shape() -> None:
    """输入：表格版式的一份日程。输出：文本行。

    预期行为：抬头 + 空行 + 表头 + 分隔行 + 每场一行，每行 7 格（序号 / 起 / 止 / 房 / 人 / 题 / 议题）。
    """
    spec = _render(sp.HybridCase(4, "early", "overlap", "table", 6, "long"), "zh")
    lines = render.build(spec, locales.load("zh")).splitlines()
    assert lines[0].startswith("会议日程 · ")
    assert lines[2] == "| # | 开始 | 结束 | 会议室 | 主持人 | 标题 | 议题 |"
    assert set(lines[3].replace("|", " ").split()) == {"---"}
    rows = [line for line in lines[4:] if line.startswith("|")]
    assert len(rows) == 4
    for position, row in enumerate(rows):
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 7
        assert cells[0] == str(position + 1)
        assert re.fullmatch(r"\d{2}:00", cells[1]) and re.fullmatch(r"\d{2}:00", cells[2])


def test_build_needs_written_text() -> None:
    """输入：文本还没写的规格。输出：KeyError。

    预期行为：渲染阶段必须先拿到后端文本；没写就直接报错，不许写出半份日程。
    """
    spec = sp.sample_spec(
        random.Random(3),
        sp.HybridCase(2, "early", "none", "bullet", 1, "short"),
        "train",
        0,
        0,
        0,
    )
    with pytest.raises(KeyError, match="还没写"):
        render.build(spec, locales.load("zh"))


@pytest.mark.parametrize("state", [None, 5, {"a": 1}, ["x"], ""])
def test_parse_state_tolerates_non_schedule_states(state: Any) -> None:
    """输入：非日程的 state。输出：空日程。

    预期行为：不崩——判别器据此判「与真值不符」，而不是在解析处抛异常。
    """
    parsed = render.parse_state(state)
    assert parsed.date is None and parsed.meetings == ()


def test_parse_state_ignores_unrelated_lines() -> None:
    """输入：夹了几行杂音的日程。输出：反解结果。

    预期行为：只认得出结构的行（带竖线 / 行首编号 + 时刻），杂音行忽略。
    """
    spec = _render(sp.HybridCase(2, "early", "none", "bullet", 3, "short"), "zh")
    noisy = "题外话\n" + render.build(spec, locales.load("zh")) + "\n\n（本日程仅供参考）"
    assert len(render.parse_state(noisy).meetings) == 2


def test_parse_state_skips_rows_with_the_wrong_cell_count() -> None:
    """输入：表格里混进一行格数不对的。输出：反解结果。

    预期行为：认不出的行跳过（不崩），认得出的照读——反解是给判别器用的，
    宁可少认一场让它判「与真值不符」，也不要在这里抛异常。
    """
    state = (
        "会议日程 · 2026-09-16\n\n"
        "| # | 开始 | 结束 | 会议室 | 主持人 | 标题 | 议题 |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| 1 | 09:00 | 10:00 | 海棠厅 |\n"
        "| 2 | 11:00 | 12:00 | 青松厅 | 李强 | 乙会 | 乙的议题 |\n"
    )
    parsed = render.parse_state(state)
    assert len(parsed.meetings) == 1 and parsed.meetings[0].start == 11


def test_parse_state_reads_the_english_locale() -> None:
    """输入：英文日程。输出：反解结果。

    预期行为：反解是语言无关的——英文标签也在合并表里，一样读得出来。
    """
    spec = _render(sp.HybridCase(3, "noon", "overlap", "table", 4, "long"), "en")
    state = render.build(spec, locales.load("en"))
    assert render.parse_state(state).meetings == tuple(
        render.expected_meetings(spec, locales.load("en"))
    )


def test_prompt_carries_no_skeleton_facts(make_spec: Any) -> None:
    """输入：一份日程的提示词。输出：提示词文本。

    预期行为：只给场数与规则——不能出现任何一场会的时刻、会议室名或主持人姓名，
    否则就是在邀请模型把骨架抄进文本（抄了会被 `check_draft` 拒掉，白烧一次配额）。
    """
    spec = make_spec(11)
    loc = locales.load("zh")
    prompt = render.build_prompt(spec, loc)
    assert "共 3 场会议" in prompt
    for meeting in spec.meetings:
        assert render.time_text(meeting.start, loc) not in prompt
        assert render.time_text(meeting.end, loc) not in prompt
    for index in range(len(spec.meetings)):
        assert render.room_text(spec, index, loc) not in prompt
        assert render.lead_text(spec, index, loc) not in prompt


def test_prompt_carries_the_theme_direction(make_spec: Any) -> None:
    """输入：两份只差主题方向的规格。输出：两段提示词。

    预期行为：提示词里出现该语言的**主题词**，且换个方向就换一段提示词。写文本的提示词里
    只有「共 N 场会」一个变量时，同场数的样本提示词**逐字相同**，模型会把标题塌成套话
    （真端点实测：114 个标题里只有 75 个不同），这条锁住那个不变量。
    """
    loc = locales.load("zh")
    themes = loc["prompt"]["themes"]
    first = render.build_prompt(dataclasses.replace(make_spec(11), theme=0), loc)
    second = render.build_prompt(dataclasses.replace(make_spec(11), theme=5), loc)
    assert themes[0] in first and themes[5] in second
    assert first != second


def test_build_task_only_exposes_the_meeting_count(make_spec: Any) -> None:
    """输入：一份规格。输出：后端任务。

    预期行为：`spec` 里只有场数（不是骨架），题型标 `text`，响应模型是草稿模型；
    任务 id 由规格派生，断点续跑能按它查进度。
    """
    spec = make_spec(13)
    loc = locales.load("zh")
    task = render.build_task(spec, loc, spec.text_task_id)
    assert task.task_id == "hybrid-train-0000000-text"
    assert task.kind == "text" and task.domain == "schedule"
    assert task.spec == {"count": 3}
    assert task.response_model is render.ScheduleDraft
    assert "meetings" in task.schema["properties"]


def test_check_draft_accepts_a_clean_draft(make_spec: Any) -> None:
    """输入：干净草稿。输出：{槽名: 文本}。

    预期行为：条数对得上、文本干净、标题 / 议题两两不同就收下，顺序与日程一致。
    """
    spec = sp.sample_spec(
        random.Random(5),
        sp.HybridCase(3, "early", "none", "bullet", 3, "short"),
        "train",
        0,
        0,
        0,
    )
    loc = locales.load("zh")
    draft = _fake().generate(render.build_task(spec, loc, "t"))
    written = render.check_draft(spec, loc, draft.data)
    assert list(written) == ["m1", "m2", "m3"]
    assert written["m1"].title == "季度预算评审"
    _ = make_spec


def test_check_draft_rejects_bad_drafts(make_spec: Any) -> None:
    """输入：四种坏草稿。输出：DraftInvalidError（带人话说明）。

    预期行为：场数不对 / 不是数组 / 夹带骨架事实 / 标题重复，都要在生成期炸掉——
    炸成「草稿不合格」而不是 provider 故障，才不会误伤熔断。
    """
    spec = make_spec(5)
    loc = locales.load("zh")
    task = render.build_task(spec, loc, "t")
    with pytest.raises(DraftInvalidError, match="没有 meetings 数组"):
        render.check_draft(spec, loc, _fake(broken=True).generate(task).data)
    with pytest.raises(DraftInvalidError, match="草稿给了"):
        render.check_draft(spec, loc, _fake(count_delta=1).generate(task).data)
    with pytest.raises(DraftInvalidError, match="夹带了时刻"):
        render.check_draft(spec, loc, _fake(dirty=True).generate(task).data)
    duplicated = {
        "meetings": [
            {"title": "同一个标题", "agenda": "第一条议题内容"},
            {"title": "同一个标题", "agenda": "第二条议题内容"},
            {"title": "另一个标题", "agenda": "第三条议题内容"},
        ]
    }
    with pytest.raises(DraftInvalidError, match="标题有重复"):
        render.check_draft(spec, loc, duplicated)


def test_check_draft_rejects_a_non_object_meeting(make_spec: Any) -> None:
    """输入：一条不是对象的会议草稿。输出：DraftInvalidError。"""
    spec = make_spec(5)
    loc = locales.load("zh")
    data = {"meetings": ["不是对象", "也不是", "还不是"]}
    with pytest.raises(DraftInvalidError, match="不是对象"):
        render.check_draft(spec, loc, data)


@pytest.mark.parametrize(
    ("text", "keyword"),
    [
        ("", "是空的"),
        ("下午 3 点的评审", "夹带了时刻或日期"),
        ("9 月 3 日的评审", "夹带了时刻或日期"),
        ("周三的评审", "夹带了星期"),
        ("星期三的评审", "夹带了星期"),
        ("在青松厅开会", "夹带了会议室名"),
        ("标题|带竖线", "有竖线"),
    ],
)
def test_text_problems_catch_every_leak(text: str, keyword: str) -> None:
    """输入：各种不干净的文本。输出：问题列表。

    预期行为：空 / 时刻 / 日期 / 星期 / 会议室名 / 竖线都要点名，
    理由里带上具体片段方便排查。
    """
    problems = render.text_problems(text, "测试")
    assert problems and any(keyword in item for item in problems)


def test_text_problems_accepts_a_clean_text() -> None:
    """输入：一段干净文本。输出：空列表。预期行为：不误判——误判会拒掉好数据。"""
    assert render.text_problems("季度预算评审", "测试") == []
    assert render.text_problems("核对各部门的预算执行情况", "测试") == []


def test_text_problems_catches_an_embedded_newline() -> None:
    """输入：一段带换行的文本。输出：问题列表。预期行为：一场会的文本只能占一行。"""
    problems = render.text_problems("第一行\n第二行", "测试")
    assert any("多行" in item for item in problems)


def test_text_helpers_reject_out_of_range_values() -> None:
    """输入：越界的时长 / 星期。输出：IndexError。

    预期行为：时长只能是 1~4 小时、星期只能是 1~7；越界说明 spec 坏了，不许静默取模。
    """
    loc = locales.load("zh")
    assert render.duration_text(2, loc) == "2 小时"
    assert render.weekday_text(7, loc) == "星期日"
    with pytest.raises(IndexError):
        render.duration_text(5, loc)
    with pytest.raises(IndexError):
        render.weekday_text(0, loc)


def test_expected_meetings_matches_the_rendered_text(make_spec: Any) -> None:
    """输入：一份规格。输出：期望值。

    预期行为：`expected_meetings` 与「把渲染文本反解一遍」得到的结果完全一致——
    判别器拿前者当金标准，两者一旦分家，好数据会被判坏。
    """
    spec = make_spec(17)
    loc = locales.load("zh")
    assert render.expected_meetings(spec, loc) == list(render.parse_state(render.build(spec, loc)).meetings)
