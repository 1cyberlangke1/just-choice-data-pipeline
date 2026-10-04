"""特征层测试：判据只认字面、命中组合按构造均衡、两条标签路径互不打架。"""

from __future__ import annotations


from tests.workflows.subqa.conftest import cue_of
from workflows.subqa import features


def test_hit_set_is_purely_literal() -> None:
    """输入：一条字幕对里的一行。输出：无。预期行为：命中组合按 `TAG_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ("Source: [MUSIC] the band picks up again", ("S1",)),
        ("Source: [音乐] 前奏响起来", ("S1",)),
        ("Source: [LAUGHTER] everyone cracks up", ("S1",)),
        ("Source: [inaudible] the tape hisses over the words", ("S1",)),
        ("Source: ♪ we wish you a merry christmas ♪", ("S1",)),
        ("Target: <i>如果还方便的话</i>，我们得把会议挪到周四。", ("S2",)),
        ("Target: 这里写 <b>重点</b> 两个字", ("S2",)),
        ("Target: salt &amp; pepper, nothing else", ("S2",)),
        ("Target: 断行用 &nbsp; 顶开", ("S2",)),
        ("Target: {\\an8}这一行贴着屏幕顶上", ("S2",)),
        ("Timing: 00:04:12,400 --> 00:04:12,400", ("S3",)),
        ("Timing: 01:02:03.500 --> 01:02:03.500", ("S3",)),
        ("Note: 这句的译法待确认，先记一笔", ("S4",)),
        ("Note: TODO 找译者问一下这里的双关", ("S4",)),
        ("Note: 这个术语需要复核一遍", ("S4",)),
        # 这几条是判据「只认字面、各占一行」的证据：形状落在**别的行**上时不该算这一条。
        ("Source: <i>这一行是原文，尖括号不该算 S2</i>", ()),
        ("Target: [MUSIC] 这一行是译文，方括号不该算 S1", ()),
        ("Target: 这一行写着 --> 但没写时间码", ()),
        ("Timing: 00:04:12,400 --> 00:04:15,900", ()),
        # 这几条是「看词边界、只认那几种形状」的证据：正常的一行一条都不该踩。
        ("Source: We need to move the meeting to Thursday, if that still works for you.", ()),
        ("Source: The rain stopped just before we left the house.", ()),
        ("Target: 如果还方便的话，我们得把会议挪到周四。", ()),
        ("Target: 这儿的 & 符号只是原文里本来就有的", ()),
        ("Note: NOTODORO 只是一个人名，不是没写完的痕迹", ()),
        ("Note: 这句直译就行，没有什么术语要查喵", ()),
    ]:
        try:
            assert features.hit_set(text) == expect
            assert features.hits_of(text) == "+".join(expect)
            assert features.parse_hits(features.hits_of(text)) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_key_lines_are_the_four_structure_keys() -> None:
    """输入：无。输出：无。

    预期行为：四个结构键是这一族 state 格式的一部分（键名用英文标识符），`missing_keys` 认的是
    「单独占一行、后面跟冒号」——少一行就定位不到钉在行上的那几条判据。
    """
    assert features.KEY_LINES == ("source", "target", "timing", "note")
    cue = cue_of(())
    assert features.missing_keys(cue) == ()
    assert features.missing_keys(cue.replace("Note:", "备注：")) == ("note",)


def test_line_value_reads_the_first_matching_line() -> None:
    """输入：两行同名的字幕对。输出：无。预期行为：只取第一行，取不到给空串。"""
    assert features.line_value("Note: 第一行\nNote: 第二行", "note") == "第一行"
    assert features.line_value("Note: 第一行", "target") == ""


def test_tracks_are_three_distinct_shapes() -> None:
    """输入：无。输出：无。

    预期行为：三个素材类型键两两不同，且**全空间里每种素材都被枚举到**——留出维只砍取值，
    不该把别的取值一起砍掉（`build_cases` 默认维度表覆盖 `TRACKS`）。
    """
    assert features.TRACKS == ("movie", "series", "lecture")
    assert len(set(features.TRACKS)) == len(features.TRACKS)
    assert features.DIMS["track"] == features.TRACKS
    cases = features.build_cases(512)
    assert {case.track for case in cases} == set(features.TRACKS)


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且与真值只差一两处
    （差得越少越难分辨，题目才有区分度）。
    """
    case = features.SubQaCase("S1+S3", features.TRACK_MOVIE)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.hits == ("S1", "S3")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.hits for combo in plan.distractors)
    assert all(len(set(combo) ^ set(plan.hits)) <= 2 for combo in plan.distractors)
    assert plan.count == 2


def test_two_label_paths_agree_on_every_faithful_cue() -> None:
    """输入：全空间的 48 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的标签逐条相同——
    字幕对兑现了计划，两条路就不该打架。
    """
    cases = features.build_cases(512)
    assert len(cases) == 48  # 16 组命中 × 3 种素材
    for case in cases:
        cue = cue_of(case.tags())
        assert features.hits_of(cue) == case.hits, case.describe()
        assert (
            features.observed_labels(cue).hits
            == features.label_for(features.plan_for(case, 0)).hits
        )


def test_case_describe_and_fingerprint() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：指纹与可读描述都稳定。"""
    case = features.SubQaCase("", features.TRACK_MOVIE)
    assert case.describe() == "无命中-movie"
    assert case.fingerprint() == "hits=|track=movie"
    assert case.tags() == ()
    assert features.parse_hits("") == ()


def test_timecode_hints_are_well_formed_and_distinct() -> None:
    """输入：无。输出：无。

    预期行为：开始时间码池里每个取值都是 `HH:MM:SS,mmm` 形状、两两不同，而且**自己不会**触发
    `S3`（同一串写两次才算零时长，池子里只有开始那一侧，所以它单独出现时不该命中）——形状写错
    或重复，旋钮就会把整批样本拖成同一个值（范式 §9.61）。
    """
    hints = features.TIMECODE_HINTS
    assert len(set(hints)) == len(hints)
    for item in hints:
        assert features.TIMECODE_SHAPE.match(item), item
        assert not features.PATTERNS["S3"].search(f"Timing: {item} --> 00:00:00,000"), item
