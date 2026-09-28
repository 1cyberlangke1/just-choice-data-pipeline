"""判别器的测试：三个零 token 判别器要能放行好数据、抓住坏数据。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random


import pytest

from jc import assemble
from workflows.hybrid import judges, locales, questions, render
from workflows.hybrid import spec as sp
from workflows.hybrid import workflow as hy_workflow


def _sample(spec: sp.HybridSpec, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：规格、语言。输出：组装好的契约样本。预期行为：走生产同一条组装路径。"""
    rng = random.Random(7)
    labeled = questions.build(spec, locales.load(lang), spec.phrasing, rng)
    return assemble.assemble_sample(
        f"hybrid-{spec.split}-{spec.seq:07d}", render.build(spec, locales.load(lang)), labeled, rng
    )


def _all_judges(lang: str = "zh") -> tuple[object, ...]:
    """输入：语言。输出：三个判别器。预期行为：与 workflow 声明的默认名单一致。"""
    return (
        judges.SkeletonJudge(locale=lang),
        judges.AnswerJudge(locale=lang),
        judges.TextJudge(locale=lang),
    )


def test_every_judge_passes_a_good_sample(make_spec: object) -> None:
    """输入：一条正常样本。输出：三个判别器。预期行为：全通过，理由为空。"""
    spec = make_spec(5)
    sample = _sample(spec)
    for judge in _all_judges():
        verdict = judge.judge(sample, spec)
        assert verdict.ok, f"{judge.name}: {verdict.reason}"


def test_skeleton_judge_skips_without_spec(make_spec: object) -> None:
    """输入：没有 spec。输出：Verdict。预期行为：跳过而不是误判（没真值就无从比对）。"""
    verdict = judges.SkeletonJudge().judge(_sample(make_spec(5)))
    assert verdict.ok and verdict.raw["skipped"] is True


def test_skeleton_judge_catches_a_corrupted_time(make_spec: object) -> None:
    """输入：把某一场的开始时刻改掉。输出：Verdict。

    预期行为：不通过并点名是哪一场的哪个字段——渲染错位是这套管线最想抓的一类 bug。
    """
    spec = make_spec(5)
    sample = _sample(spec)
    sample["state"] = sample["state"].replace("09:00", "10:00", 1)
    verdict = judges.SkeletonJudge().judge(sample, spec)
    assert not verdict.ok and "start" in verdict.reason


def test_skeleton_judge_catches_a_dropped_meeting(make_spec: object) -> None:
    """输入：删掉一场会的三行。输出：Verdict。预期行为：场数对不上要报出来。"""
    spec = make_spec(5, meetings=3)
    sample = _sample(spec)
    lines = sample["state"].splitlines()
    sample["state"] = "\n".join(lines[:-3])
    verdict = judges.SkeletonJudge().judge(sample, spec)
    assert not verdict.ok and "场数" in verdict.reason


def test_skeleton_judge_catches_a_wrong_date(make_spec: object) -> None:
    """输入：把抬头日期换一天。输出：Verdict。预期行为：日期对不上要报出来。"""
    spec = make_spec(5)
    sample = _sample(spec)
    year, month, day = spec.date
    sample["state"] = sample["state"].replace(
        f"{year:04d}-{month:02d}-{day:02d}", "2030-01-01", 1
    )
    verdict = judges.SkeletonJudge().judge(sample, spec)
    assert not verdict.ok and "日期" in verdict.reason


@pytest.mark.parametrize("family", ["earliest_start", "duration_of", "meeting_day"])
def test_answer_judge_catches_a_wrong_choice(make_spec: object, family: str) -> None:
    """输入：把某道 choice 的答案指向另一个选项。输出：Verdict。

    预期行为：不通过——判别器只读日程与题面，重算出来的答案与数据里的不一样。
    """
    spec = make_spec(5)
    sample = _sample(spec)
    qid = next(name for name in sample["questions"] if name == family)
    options = len(sample["questions"][qid]["criteria"])
    sample["answers"][qid]["choice"] = (sample["answers"][qid]["choice"] + 1) % options
    verdict = judges.AnswerJudge().judge(sample, spec)
    assert not verdict.ok and "重算" in verdict.reason


def test_answer_judge_catches_a_wrong_bool_and_score(make_spec: object) -> None:
    """输入：把 noul 与 score 的答案改掉。输出：Verdict。预期行为：两个都要报出来。"""
    spec = make_spec(5, **{"meetings": 3, "clash": "overlap"})
    sample = _sample(spec)
    sample["answers"]["no_overlap"]["noul"] = True  # 真值是 False（确实有重叠）
    sample["answers"]["busy_level"]["score"] = (sample["answers"]["busy_level"]["score"] + 1) % 5
    verdict = judges.AnswerJudge().judge(sample, spec)
    assert not verdict.ok
    assert "no_overlap" in verdict.reason and "busy_level" in verdict.reason


def test_answer_judge_works_without_the_spec(make_spec: object) -> None:
    """输入：不给 spec。输出：Verdict。

    预期行为：照样能判（它只读文本与题面）——这正是「独立重算」而不是「拿答案验证答案」。
    """
    sample = _sample(make_spec(5))
    assert judges.AnswerJudge().judge(sample, None).ok


def test_answer_judge_reports_an_unrecognized_question(make_spec: object) -> None:
    """输入：一条题面认不出来的题。输出：Verdict。预期行为：点名而不是静默放过。"""
    spec = make_spec(5)
    sample = _sample(spec)
    sample["questions"]["earliest_start"]["instructions"] = "这句话不在任何问法里"
    verdict = judges.AnswerJudge().judge(sample, spec)
    assert not verdict.ok and "认不出题面" in verdict.reason


def test_answer_judge_catches_an_out_of_range_index(make_spec: object) -> None:
    """输入：答案编号越界。输出：Verdict。预期行为：报越界而不是抛异常。"""
    spec = make_spec(5)
    sample = _sample(spec)
    sample["answers"]["earliest_start"]["choice"] = 99
    verdict = judges.AnswerJudge().judge(sample, spec)
    assert not verdict.ok and "越界" in verdict.reason


def test_text_judge_passes_a_good_sample(make_spec: object) -> None:
    """输入：一条正常样本。输出：Verdict。预期行为：通过。"""
    spec = make_spec(5)
    assert judges.TextJudge().judge(_sample(spec), spec).ok


def test_text_judge_catches_dirty_and_duplicated_text(make_spec: object) -> None:
    """输入：夹带时刻 / 标题重复的日程。输出：Verdict。

    预期行为：两种都要报出来——渲染阶段虽然拦过，但判别器查的是**最终产物**，
    能抓出渲染阶段的 bug（两道查的不是同一样东西）。
    """
    spec = make_spec(5)
    sample = _sample(spec)
    dirty = sample["state"].replace(spec.text_of("m1").title, "下午 3 点的评审", 1)
    assert not judges.TextJudge().judge({**sample, "state": dirty}, spec).ok
    duplicated = sample["state"].replace(spec.text_of("m2").title, spec.text_of("m1").title, 1)
    verdict = judges.TextJudge().judge({**sample, "state": duplicated}, spec)
    assert not verdict.ok and "重复" in verdict.reason


def test_text_judge_catches_an_empty_state() -> None:
    """输入：读不出任何会议的 state。输出：Verdict。预期行为：不通过。"""
    verdict = judges.TextJudge().judge({"state": "随便一段文字", "questions": {}, "answers": {}})
    assert not verdict.ok and "一场会都没有" in verdict.reason


def _parsed(*rows: tuple[int, int, str, str]) -> render.ParsedSchedule:
    """输入：[(开始, 结束, 会议室, 主持人)]。输出：反解结果（文本字段留空）。

    预期行为：手搓结构，用来把三个重算函数钉死——它们只看时刻与会议室。
    """
    return render.ParsedSchedule(
        date=(2026, 9, 23),
        meetings=tuple(
            render.ParsedMeeting(start, end, room, lead, "题", "议题")
            for start, end, room, lead in rows
        ),
    )


def test_recompute_helpers_on_handmade_schedules() -> None:
    """输入：几种手搓日程。输出：三个重算函数的结果。

    预期行为：重叠只看时间；房间冲突要求「同房 **且** 时间重叠」；
    繁忙度只数 `[BUSY_FROM, BUSY_TO)` 窗口内被覆盖的整点。
    """
    assert judges._overlap(_parsed((9, 10, "甲", "A"), (11, 12, "甲", "B"))) is False
    assert judges._overlap(_parsed((9, 11, "甲", "A"), (10, 12, "乙", "B"))) is True
    assert judges._room_clash(_parsed((9, 11, "甲", "A"), (10, 12, "甲", "B"))) is True
    assert judges._room_clash(_parsed((9, 10, "甲", "A"), (10, 11, "甲", "B"))) is False
    assert judges._busy_level(_parsed((8, 9, "甲", "A"), (13, 15, "乙", "B"))) == 2
    assert judges._busy_level(_parsed((12, 20, "甲", "A"))) == sp.BUSY_CAP


def test_make_judges_builds_the_declared_names() -> None:
    """输入：默认名单 / 未知名字。输出：判别器元组 / KeyError。

    预期行为：名字与 `workflow.judges` 一一对应；未知名字直接报错（别静默不跑）。
    """
    workflow = hy_workflow.HybridWorkflow()
    built = workflow.make_judges(workflow.judges, "zh")
    assert [judge.name for judge in built] == list(workflow.judges)
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("made_up",), "zh")


def test_answer_judge_reports_an_unrecognized_family_question(make_spec: object) -> None:
    """输入：把 noul 题的题面改坏。输出：Verdict。预期行为：一样要点名，不能只查 choice。"""
    spec = make_spec(5)
    sample = _sample(spec)
    sample["questions"]["no_overlap"]["instructions"] = "这句话不在任何问法里"
    verdict = judges.AnswerJudge().judge(sample, spec)
    assert not verdict.ok and "认不出题面" in verdict.reason


def test_recompute_choice_reports_unusable_targets() -> None:
    """输入：几种推不出期望值的输入。输出：(None, 问题列表)。

    预期行为：目标槽认不出来 / 越界、日程里一场会都没有、读不出日期、会议室名不在词表里，
    都要返回 None 并说明原因——判别器宁可判不通过，也不能瞎猜一个期望值。
    """
    loc = locales.load("zh")
    one = _parsed((9, 10, "甲", "A"))
    assert judges.recompute_choice("duration_of", "zz", one, loc)[0] is None
    assert "认不出来" in judges.recompute_choice("duration_of", "zz", one, loc)[1][0]
    assert "超出日程" in judges.recompute_choice("duration_of", "m9", one, loc)[1][0]
    empty = render.ParsedSchedule(None, ())
    assert "一场会都没有" in judges.recompute_choice("earliest_start", "", empty, loc)[1][0]
    assert "读不出日期" in judges.recompute_choice("meeting_day", "", empty, loc)[1][0]
    unknown = _parsed((9, 10, "不存在的厅", "A"))
    assert "不在词表里" in judges.recompute_choice("room_of", "m1", unknown, loc)[1][0]
    assert judges.recompute_choice("lead_of", "m1", one, loc) == ("m1", [])
    assert judges.recompute_choice("title_of", "m1", one, loc)[0] == "t1"
    assert judges.recompute_choice("agenda_of", "m1", one, loc)[0] == "a1"


def test_answer_judge_reports_a_room_it_cannot_map(make_spec: object) -> None:
    """输入：日程里出现了词表外的会议室名。输出：Verdict。

    预期行为：不通过并说明「不在词表里」——推不出期望值时必须说清楚，
    否则判别器会静默放过一条可能错的数据。
    """
    spec = make_spec(5)
    sample = _sample(spec)
    loc = locales.load("zh")
    for index in range(len(spec.meetings)):
        sample["state"] = sample["state"].replace(render.room_text(spec, index, loc), "未知厅", 1)
    verdict = judges.AnswerJudge().judge(sample, spec)
    assert not verdict.ok and "不在词表里" in verdict.reason
