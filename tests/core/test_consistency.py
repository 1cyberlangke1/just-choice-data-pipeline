"""通用层测试：一致性校验（`jc.consistency`）。

要验的就一件事：**同一段素材上的同一道题，只能有一个答案**。素材相同而标签不同的两条样本
是自相矛盾的训练对，模型只会被这一对互相抵消（照抄 Open-Jev 的 `validate_records`）。
"""

from __future__ import annotations

from typing import Any

from jc import consistency


def make_sample(
    sample_id: str,
    *,
    state: Any = "素材X",
    criteria: dict[str, str] | list[str] | None = None,
    answer: dict[str, Any] | None = None,
    instructions: str = "选哪个？",
    qtype: str = "choice",
) -> dict[str, Any]:
    """输入：样本 id 与题目 / 答案的各字段。输出：一条最小契约样本。"""
    if criteria is None:
        criteria = {"p": "甲", "q": "乙"}
    if answer is None:
        answer = {"type": "choice", "choice": 0}
    return {
        "id": sample_id,
        "state": state,
        "questions": {"pick": {"type": qtype, "instructions": instructions, "criteria": criteria}},
        "answers": {"pick": answer},
    }


def test_state_text_normalises_objects() -> None:
    """输入：字符串 state 与对象 state。输出：可比文本。

    预期行为：字符串原样；对象按**排序后的键**序列化——键序不同的同一个对象要认成同一个，
    否则 `{"a":1,"b":2}` 与 `{"b":2,"a":1}` 会被当成两段素材，冲突就漏了。
    """
    assert consistency.state_text("素材") == "素材"
    assert consistency.state_text({"b": 2, "a": 1}) == consistency.state_text({"a": 1, "b": 2})


def test_question_key_shape() -> None:
    """输入：三种题型的题目。输出：指纹。

    预期行为：指纹 = (题型, 问法, **排序后**的选项文字, 金答案文字)；形态不对给 None。
    """
    choice = make_sample("a-train-0000001")
    assert consistency.question_key(
        choice["questions"]["pick"], choice["answers"]["pick"]
    ) == ("choice", "选哪个？", ("乙", "甲"), "甲")
    score = make_sample(
        "a-train-0000002",
        qtype="score",
        criteria=["0：没有", "1：一个"],
        answer={"type": "score", "score": 1},
    )
    assert consistency.question_key(score["questions"]["pick"], score["answers"]["pick"]) == (
        "score",
        "选哪个？",
        ("0：没有", "1：一个"),
        "1：一个",
    )
    noul = make_sample(
        "a-train-0000003",
        qtype="noul",
        criteria={"true": "对", "false": "不对"},
        answer={"type": "noul", "noul": False},
    )
    assert consistency.question_key(noul["questions"]["pick"], noul["answers"]["pick"]) == (
        "noul",
        "选哪个？",
        ("不对", "对"),
        "不对",
    )
    assert consistency.question_key({"type": "choice"}, {}) is None
    assert consistency.question_key({"instructions": "?"}, {}) is None
    assert consistency.question_key({"type": "choice", "instructions": "?"}, {}) == (
        "choice",
        "?",
        (),
        "{}",
    )


def test_no_conflict_when_answers_agree() -> None:
    """输入：同一素材、同一道题、同一个答案（选项顺序被洗过）。输出：无冲突。

    预期行为：选项顺序不同不算冲突——编号是组装器洗过的，本来就该变。
    """
    first = make_sample("a-train-0000001", criteria={"p": "甲", "q": "乙"})
    # 选项顺序换了，但答案还是「甲」——编号从 0 变成 1，这正是「编号不能拿来比」的原因
    second = make_sample(
        "a-train-0000002", criteria={"q": "乙", "p": "甲"}, answer={"type": "choice", "choice": 1}
    )
    assert consistency.conflicts([first, second]) == []


def test_no_conflict_across_different_states() -> None:
    """输入：两段不同素材上的同题不同答。输出：无冲突。预期行为：不同素材本来就该能不同答。"""
    first = make_sample("a-train-0000001", state="素材X")
    second = make_sample(
        "a-train-0000002", state="素材Y", criteria={"p": "乙", "q": "甲"}
    )
    assert consistency.conflicts([first, second]) == []


def test_conflict_is_found_across_shuffled_options() -> None:
    """输入：同一段素材上「选项集合相同、金答案不同」的两条。输出：一条冲突。

    预期行为：冲突记录带素材指纹、问法、选项文字与**每种答案各自的样本 id**——
    没有 id 就没法回产物里查。
    """
    first = make_sample("a-train-0000001", criteria={"p": "甲", "q": "乙"})
    second = make_sample("a-train-0000002", criteria={"q": "乙", "p": "甲"})
    out = consistency.conflicts([first, second])
    assert len(out) == 1
    record = out[0]
    assert record["question"] == "选哪个？"
    assert record["answers"] == {"甲": ["a-train-0000001"], "乙": ["a-train-0000002"]}
    assert record["state_preview"] == "素材X"
    assert len(record["state_sha"]) == 12


def test_one_row_per_conflict_point() -> None:
    """输入：同一冲突点上 5 条样本。输出：**一条**冲突，而不是一堆样本对。

    预期行为：一个冲突点只记一条（带两种答案各自的样本 id 样例）——
    按样本对记的话，一段素材有几百条样本时会刷出上千行，没人看得完。
    """
    samples = [make_sample("a-train-0000001", criteria={"p": "甲", "q": "乙"})]
    samples += [make_sample(f"a-train-000000{i}", criteria={"q": "乙", "p": "甲"}) for i in range(2, 6)]
    out = consistency.conflicts(samples)
    assert len(out) == 1
    assert out[0]["answers"]["乙"] == ["a-train-0000002", "a-train-0000003"]


def test_different_option_sets_are_different_questions() -> None:
    """输入：同一素材、同一问法，但选项集合不同。输出：无冲突。

    预期行为：选项都不一样就是两道不同的题，答案不同是正常的。
    """
    first = make_sample("a-train-0000001", criteria={"p": "甲", "q": "乙"})
    second = make_sample("a-train-0000002", criteria={"p": "丙", "q": "丁"})
    assert consistency.conflicts([first, second]) == []


def test_noul_and_score_conflicts() -> None:
    """输入：noul 与 score 上的同题不同答。输出：各一条冲突。

    预期行为：noul 比的是 true / false 的文案，score 比的是等级文字——
    两种题型都不能只看编号。
    """
    noul_true = make_sample(
        "a-train-0000001",
        qtype="noul",
        criteria={"true": "对", "false": "不对"},
        answer={"type": "noul", "noul": True},
    )
    noul_false = make_sample(
        "a-train-0000002",
        qtype="noul",
        criteria={"true": "对", "false": "不对"},
        answer={"type": "noul", "noul": False},
    )
    assert len(consistency.conflicts([noul_true, noul_false])) == 1
    score_low = make_sample(
        "a-train-0000003",
        qtype="score",
        criteria=["0：没有", "1：一个"],
        answer={"type": "score", "score": 0},
    )
    score_high = make_sample(
        "a-train-0000004",
        qtype="score",
        criteria=["0：没有", "1：一个"],
        answer={"type": "score", "score": 1},
    )
    assert len(consistency.conflicts([score_low, score_high])) == 1


def test_missing_pieces_do_not_crash() -> None:
    """输入：缺 answers / 缺 questions 的样本。输出：无冲突、不抛异常。

    预期行为：这条校验跑在契约校验**之后**，脏样本到不了这里；真到了也不能炸整批。
    """
    assert consistency.conflicts([{"id": "a-train-0000001", "state": "素材"}]) == []
    broken = {"id": "a-train-0000002", "state": "素材", "questions": {"pick": {"type": "choice"}}}
    assert consistency.conflicts([broken]) == []
