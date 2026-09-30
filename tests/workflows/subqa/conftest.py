"""subqa 的假后端与草稿构造器：脚本化出一条双语字幕对，不联网。

假后端按 `task.spec` 现造字幕对——**默认完全兑现计划**（字幕对里的形状由注入特征推出来），另外
几个开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。

**每条判据各占一整行**（`S1` 只看 `Source:` 行、`S2` 只看 `Target:` 行、`S3` 只看 `Timing:` 行、
`S4` 只看 `Note:` 行），所以「踩中某一条」就是把它那一行换成对应的形状，四条之间不会互相遮蔽
——这是这一族从设计上避开「两条判据共用一段文字」的做法（范式 §9.51）。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from workflows.subqa import features, locales, render
from workflows.subqa.features import SubQaCase
from workflows.subqa.spec import SubQaSpec

#: 四行骨架：一条平平无奇、可以直接发的字幕对（四条判据一条都不踩），长度也落在
#: `render.MIN_CONTENT_CHARS` 以上。语义上它就是「直接发」那一档要的东西（0 条命中的字幕对）。
BASE_LINES: Final[dict[str, str]] = {
    "source": "Source: We need to move the meeting to Thursday, if that still works for you.",
    "target": "Target: 如果还方便的话，我们得把会议挪到周四。",
    "timing": "Timing: 00:04:12,400 --> 00:04:15,900",
    "note": "Note: 这句直译就行，没有什么术语要查喵",
}

#: 每条判据的「踩中形状」——它**真的落在自己那一行上**，所以四条各占一行、互不遮蔽。
HIT_LINES: Final[dict[str, str]] = {
    "S1": "Source: [MUSIC] We need to move the meeting to Thursday, if that still works.",
    "S2": "Target: <i>如果还方便的话</i>，我们得把会议挪到周四。",
    "S3": "Timing: 00:04:12,400 --> 00:04:12,400",
    "S4": "Note: 这句的译法待确认，先记一笔",
}

#: 判据编号 → 它钉在哪一个结构键那一行上。
LINE_KEYS: Final[dict[str, str]] = {
    "S1": "source",
    "S2": "target",
    "S3": "timing",
    "S4": "note",
}

#: 每条判据在字幕对里**原样出现**的那一段（反解引文用它，见 `quote_for`）。引文要**连键名一起**
#: 抄全（反解提示词里的口径），所以它自己单拿出来也满足那条判据。
QUOTES: Final[dict[str, str]] = {
    "S1": "Source: [MUSIC]",
    "S2": "Target: <i>如果还方便的话</i>",
    "S3": "Timing: 00:04:12,400 --> 00:04:12,400",
    "S4": "Note: 这句的译法待确认",
}


def lines_of(tags: tuple[str, ...]) -> dict[str, str]:
    """输入：该命中的判据。输出：四行字幕对的取值表（键 → 那一行的全文）。"""
    lines = dict(BASE_LINES)
    for tag in features.TAG_IDS:
        if tag in tags:
            lines[LINE_KEYS[tag]] = HIT_LINES[tag]
    return lines


def render_lines(lines: dict[str, str]) -> str:
    """输入：四行取值表。输出：按 `features.KEY_LINES` 顺序拼起来的字幕对。"""
    return "\n".join(lines[key] for key in features.KEY_LINES if key in lines)


def cue_of(tags: tuple[str, ...]) -> str:
    """输入：该命中的判据。输出：一条完全兑现它们的字幕对。

    预期行为：按 `TAG_IDS` 顺序把该命中的那几行换成「踩中形状」——每条判据各占一整行，换上去
    也不会互相遮蔽，所以按字面算出来的命中组合就是 `tags`。
    """
    return render_lines(lines_of(tags))


def quote_for(tag: str) -> str:
    """输入：判据编号。输出：该判据在字幕对里**原样出现**的那一段（引文）。

    预期行为：直接取 `QUOTES` 里那一段——它确实在 `cue_of` 的产物里，所以反解带上它就站得住
    （`judges.quote_holds` 会原样比对，且它自己也满足那条判据）。
    """
    return QUOTES[tag]


def trace_report(case: SubQaCase, *, lie: bool = False) -> list[dict[str, str]]:
    """输入：注入特征、要不要说反话。输出：反解草稿的 tags（每条带编号与原样引文）。

    预期行为：`lie=True` 时**漏报**最后一条（读者没读出来），这是 `TraceJudge` 该拦的方向；
    多报一条是另一条路径——引文站不住就会被 `judges.quote_holds` 当误报丢掉（见 test_judges）。
    """
    tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
    if lie and tags:
        tags = tags[:-1]
    return [{"tag": tag, "quote": quote_for(tag)} for tag in tags]


def spec_for(
    case: SubQaCase,
    phrasing: int = 0,
    split: str = "train",
    seq: int = 1,
    timecode: str = features.TIMECODE_HINTS[0],
) -> SubQaSpec:
    """输入：case、问法编号、切分、序号、开始时间码。输出：真值规格。

    预期行为：`timecode` 默认取池子里的第一个——夹具只关心「旋钮有没有传下去」，取值本身由
    `test_pipeline` 与 `test_locales` 各自钉住。
    """
    return SubQaSpec(
        case=case,
        phrasing=phrasing,
        timecode=timecode,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(cue: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：一条字幕对、语言。输出：只带 state 的样本。"""
    return {"state": render.to_state(render.ContentDraft(cue=cue), locales.load(lang))}


def faithful(case: SubQaCase, lang: str = "zh") -> tuple[SubQaSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：字幕对完全兑现计划。"""
    spec = spec_for(case)
    cue = cue_of(case.tags())
    render.check_draft(render.ContentDraft(cue=cue), case, locales.load(lang))
    return spec, sample_of(cue, lang)


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时把命中组合说反（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_line` 少写一条该命中的形状、`extra_line` 多写一条不该命中的形状；
    - `missing_key` 把 `Note:` 那一行整行抹掉（验「四个结构键必须齐全」那道闸）；
    - `too_short` 把字幕对缩到一个字符、`fence` 给字幕对套上围栏。
    """

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        trace_raises: Exception | None = None,
        drop_line: bool = False,
        extra_line: bool = False,
        missing_key: bool = False,
        too_short: bool = False,
        fence: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.drop_line = drop_line
        self.extra_line = extra_line
        self.missing_key = missing_key
        self.too_short = too_short
        self.fence = fence
        self.tasks: list[str] = []

    def cue_for(self, case: SubQaCase) -> str:
        """输入：注入特征。输出：按开关写歪之后的一条字幕对。"""
        tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
        if self.drop_line and tags:
            tags = tags[:-1]
        if self.extra_line:
            spare = [tag for tag in features.TAG_IDS if tag not in case.tags()]
            if spare:
                tags = [*tags, spare[0]]
        lines = lines_of(tuple(tags))
        if self.missing_key:
            lines.pop("note")
        if self.too_short:
            return "短"
        cue = render_lines(lines)
        return "```" + cue if self.fence else cue

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        case = SubQaCase(hits=str(spec["case"]["hits"]), track=str(spec["case"]["track"]))
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            report = trace_report(case, lie=self.trace_lies)
            return _draft({"tags": report}, "tags=" + "+".join(item["tag"] for item in report))
        cue = self.cue_for(case)
        return _draft({"cue": cue}, cue)

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：草稿列表。预期行为：逐个走 `generate`。"""
        return [self.generate(task) for task in tasks]


def _draft(data: dict[str, JsonValue], raw: str) -> Draft:
    """输入：解析后的数据、原始文本。输出：`Draft`。预期行为：记账字段填假值，不进产物。"""
    return Draft(
        data=data,
        raw_text=raw,
        backend="fake",
        provider="fake",
        model="fake",
        mode="fake",
    )


@pytest.fixture
def backend() -> FakeBackend:
    """输入：无。输出：一个完全兑现计划的假后端。"""
    return FakeBackend()


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
