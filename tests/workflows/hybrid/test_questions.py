"""题族的测试：十个题族、答案、选项、题面反查。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from jc.workflow import family_of
from workflows.hybrid import locales, questions, render
from workflows.hybrid import spec as sp


def _build(spec: sp.HybridSpec, lang: str = "zh", phrasing: int = 0) -> dict[str, Any]:
    """输入：规格、语言、问法编号。输出：{题族: LabeledQuestion}。

    预期行为：题号即题族名（每族只出一题），方便按名字取。
    """
    labeled = questions.build(spec, locales.load(lang), phrasing, random.Random(7))
    return {item.family: item for item in labeled}


def test_all_ten_families_and_their_types(make_spec: Any) -> None:
    """输入：一份普通日程。输出：题目列表。

    预期行为：十个题族全出、各一题、顺序按 `FAMILIES`；三种题型都产。
    """
    labeled = questions.build(make_spec(5), locales.load("zh"), 0, random.Random(1))
    assert [item.family for item in labeled] == list(questions.FAMILIES)
    kinds = {item.question["type"] for item in labeled}
    assert kinds == {"choice", "noul", "score"}
    assert len(questions.FAMILIES) == 10


def test_choice_shuffles_and_noul_score_do_not(make_spec: Any) -> None:
    """输入：一份日程。输出：题目列表。

    预期行为：choice 允许打乱（防位置捷径）；noul / score 的 criteria 有固定顺序，禁止打乱。
    """
    for item in questions.build(make_spec(5), locales.load("zh"), 0, random.Random(1)):
        assert item.shuffle is (item.question["type"] == "choice")


def test_earliest_start_options_are_the_start_times(make_spec: Any) -> None:
    """输入：一份日程。输出：`earliest_start` 题。

    预期行为：选项名是槽名、描述是各场会的开始时刻，答案是开始最早那一场；
    开始时刻两两不同，所以答案唯一。
    """
    spec = make_spec(5)
    loc = locales.load("zh")
    item = _build(spec)["earliest_start"]
    assert item.question["criteria"] == {
        slot: render.time_text(meeting.start, loc)
        for slot, meeting in zip(spec.slot_names(), spec.meetings, strict=True)
    }
    earliest = min(spec.meetings, key=lambda meeting: meeting.start)
    assert item.question["criteria"][item.answer] == render.time_text(earliest.start, loc)


def test_earliest_start_answer_is_not_always_the_first_slot(make_spec: Any) -> None:
    """输入：一批不同种子的日程。输出：`earliest_start` 的答案集合。

    预期行为：答案不能永远是 `m1`——展示顺序与时间顺序无关（见 spec.sample_spec），
    否则这道题会退化成「答案 = 列表第一行」的位置捷径。
    """
    loc = locales.load("zh")
    answers = {
        item.answer
        for seed in range(12)
        for item in questions.build(make_spec(seed), loc, 0, random.Random(seed))
        if item.family == "earliest_start"
    }
    assert len(answers) > 1


def test_meeting_day_offers_seven_weekdays(make_spec: Any) -> None:
    """输入：一份日程。输出：`meeting_day` 题。

    预期行为：七个选项（星期一到星期日），答案是日程日期的星期几。
    日期写在 state 里、星期几**不写**，所以这是一道要算的题。
    """
    spec = make_spec(5, weekday=6)
    loc = locales.load("zh")
    item = _build(spec)["meeting_day"]
    assert list(item.question["criteria"]) == list(questions.WEEKDAY_KEYS)
    assert item.question["criteria"][item.answer] == render.weekday_text(6, loc)


def test_duration_options_are_fixed_and_match_the_target(make_spec: Any) -> None:
    """输入：一份日程。输出：`duration_of` 题。

    预期行为：选项恒为 1~4 小时（短时长档下后两个是干扰项），
    答案是题面点名的第 n 场会的时长。
    """
    spec = make_spec(5)
    loc = locales.load("zh")
    item = _build(spec)["duration_of"]
    assert list(item.question["criteria"]) == list(questions.DURATION_KEYS)
    target = int(item.question["instructions"].split("第 ")[1].split(" 场")[0]) - 1
    assert item.question["criteria"][item.answer] == render.duration_text(
        spec.meetings[target].dur, loc
    )


def test_room_options_are_deduplicated_on_a_shared_room(make_spec: Any) -> None:
    """输入：`same_room` 档的日程。输出：`room_of` 题。

    预期行为：选项是**去重后**用到的会议室（共用一间房只算一个选项），
    否则那道题会出现两个一模一样的选项描述。
    """
    spec = make_spec(5, **{"meetings": 3, "clash": "same_room"})
    loc = locales.load("zh")
    item = _build(spec)["room_of"]
    assert len(item.question["criteria"]) == len(spec.room_slots) == 2
    assert len(set(item.question["criteria"].values())) == 2


def test_lead_options_are_keyed_by_slot(make_spec: Any) -> None:
    """输入：一份日程。输出：`lead_of` 题。

    预期行为：选项名是槽名、描述是主持人姓名（两两不同），答案就是题面那一场的槽名。
    """
    spec = make_spec(5, meetings=4)
    loc = locales.load("zh")
    item = _build(spec)["lead_of"]
    assert list(item.question["criteria"]) == list(spec.slot_names())
    assert item.question["criteria"][item.answer] == render.lead_text(
        spec, int(item.answer[1:]) - 1, loc
    )


@pytest.mark.parametrize(("family", "field"), [("title_of", "title"), ("agenda_of", "agenda")])
def test_text_families_use_the_written_text(make_spec: Any, family: str, field: str) -> None:
    """输入：一份日程。输出：问文本的两族题。

    预期行为：选项就是**全部**会议的那段文本（真值 = 后端写出来的原文），
    答案是题面点名的那一场；文本两两不同由渲染阶段保证，否则这里就会有两个正确答案。
    """
    spec = make_spec(5, meetings=4)
    item = _build(spec)[family]
    assert list(item.question["criteria"]) == [f"{field[0]}{n}" for n in range(1, 5)]
    target = spec.slot_names()[int(item.answer[1:]) - 1]
    assert item.question["criteria"][item.answer] == getattr(spec.text_of(target), field)
    assert len(set(item.question["criteria"].values())) == 4


def test_noul_answers_follow_the_skeleton(make_spec: Any) -> None:
    """输入：不重叠 / 重叠 / 房间冲突三种日程。输出：两族 noul 的答案。

    预期行为：题族名是**断言**（「互不重叠」「会议室没有冲突」），答案是断言成立与否。
    """
    clean = _build(make_spec(5, **{"meetings": 3, "clash": "none"}))
    assert clean["no_overlap"].answer is True and clean["room_free"].answer is True
    overlap = _build(make_spec(5, **{"meetings": 3, "clash": "overlap"}))
    assert overlap["no_overlap"].answer is False and overlap["room_free"].answer is True
    shared = _build(make_spec(5, **{"meetings": 3, "clash": "same_room"}))
    assert shared["room_free"].answer is False


def test_score_answer_is_the_busy_level(make_spec: Any) -> None:
    """输入：一份日程。输出：`busy_level` 题。

    预期行为：等级表是五档（0~4 个整点），答案是骨架算出来的繁忙度。
    """
    spec = make_spec(5)
    item = _build(spec)["busy_level"]
    assert len(item.question["criteria"]) == sp.BUSY_CAP + 1
    assert item.answer == spec.busy_level
    assert item.question["criteria"][item.answer].startswith(str(item.answer))


@pytest.mark.parametrize("lang", ["zh", "en"])
@pytest.mark.parametrize("phrasing", [0, 1, 2])
def test_locate_roundtrips_every_phrasing(make_spec: Any, lang: str, phrasing: int) -> None:
    """输入：每种问法下的一道题。输出：`locate` 的结果。

    预期行为：题面必须能被反查回自己的题族与目标槽——判别器完全依赖这一步，
    认错就等于拿错的标准去判数据。
    """
    spec = make_spec(5, meetings=4)
    loc = locales.load(lang)
    for item in questions.build(spec, loc, phrasing, random.Random(3)):
        found = questions.locate(item.question["instructions"], loc, len(spec.meetings))
        assert found is not None, item.question["instructions"]
        family, target = found
        assert family == item.family
        if item.family in questions.SLOT_FAMILIES:
            assert target.startswith("m")
        else:
            assert target == ""


def test_locate_returns_none_for_an_unknown_question() -> None:
    """输入：认不出的题面。输出：None。预期行为：不猜。"""
    assert questions.locate("这句话不在任何问法里", locales.load("zh"), 3) is None


def test_choice_rejects_ambiguous_options() -> None:
    """输入：选项描述重复 / 答案不在选项里。输出：ValueError。

    预期行为：这两种都是构造 bug，必须在生成时炸掉——描述重复意味着两个正确答案。
    """
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("x", "q", {"a": "同一个", "b": "同一个"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("x", "q", {"a": "一", "b": "二"}, "c")


def test_key_helpers() -> None:
    """输入：下标 / 时长。输出：选项名。预期行为：一律从 1 数起，与题面里的「第 n 场」对齐。"""
    assert questions.duration_key(3) == "h3"
    assert questions.room_key(0) == "r1"
    assert questions.title_key(1) == "t2"
    assert questions.agenda_key(2) == "a3"
    assert family_of("title_of_2") == "title_of"
