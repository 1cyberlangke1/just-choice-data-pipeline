"""chatgate 的假后端与草稿构造器：脚本化出草稿，不联网。

假后端按 `task.spec` 现造消息与三条候选——**默认完全兑现计划**（消息里的片段由计划里的命中判据
推出来、三条候选的毛病按计划分配），另外几个开关各自破坏一件事，用来验「渲染期与判别器真的会
拦下来」。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows._fakes import draft as _draft
from workflows.chatgate import features, locales, render
from workflows.chatgate.features import ChatCase
from workflows.chatgate.spec import ChatGateSpec

#: 消息骨架：一个命中片段都不含，长度落在 `render.MIN_MESSAGE_CHARS` 以上。
#: 语义上它是一条**平平常常、可以直接回**的消息（`degree=0` 那一档要的就是这种东西）。
BASE: Final[str] = "项目文档我更新好了，你有空的时候翻一下。"

#: 每条判据对应的片段：命中哪一条就往消息里拼哪一段。
#: 每一段都**只**踩自己那一条（`test_features` 把这件事钉在数据上）。
FRAGMENTS: Final[dict[str, str]] = {
    "C1": "你看用哪个方案，你定吧。",
    "C2": "今天之内能给我吗？",
    "C3": "先借我两百块应个急。",
    "C4": "把你的验证码发我一下。",
}

#: 每条毛病对应的候选片段：踩哪一条毛病就写哪一段。
FLAW_FRAGMENTS: Final[dict[str, str]] = {
    "F1": "没问题，包在我身上。",
    "F2": "把你的卡号发我，我核对一下。",
    "F3": "再说吧，看情况。",
}

#: 干净候选：一个毛病都不踩。
CLEAN: Final[str] = "收到，我下午看一遍再回你。"


def message_of(hits: tuple[str, ...]) -> str:
    """输入：该命中的判据。输出：一条完全兑现它的消息（片段按 `TAG_IDS` 顺序拼）。"""
    parts = [BASE]
    parts.extend(FRAGMENTS[tag] for tag in features.TAG_IDS if tag in hits)
    return "".join(parts)


def candidate_of(flaws: tuple[str, ...]) -> str:
    """输入：该踩的毛病。输出：一条完全兑现它的候选（空元组给那条干净的）。"""
    if not flaws:
        return CLEAN
    return "".join(FLAW_FRAGMENTS[flaw] for flaw in features.FLAW_IDS if flaw in flaws)


def candidates_of(flaws: tuple[tuple[str, ...], ...]) -> list[str]:
    """输入：三条候选的毛病计划。输出：三条候选正文。"""
    return [candidate_of(item) for item in flaws]


def quote_for(tag: str) -> str:
    """输入：判据编号。输出：该判据在消息里**原样出现**的那一段（引文）。

    预期行为：直接取 `FRAGMENTS` 里那一段（去掉两端空白）——它确实在 `message_of` 的产物里，
    所以反解带上它就站得住（`judges.quote_holds` 会原样比对消息）。
    """
    return FRAGMENTS[tag].strip()


def trace_report(hits: tuple[str, ...], *, lie: bool = False) -> list[dict[str, str]]:
    """输入：该命中的判据、要不要说反话。输出：反解草稿的 tags（每条带编号与原样引文）。

    预期行为：`lie=True` 时**漏报**最后一条（读者没读出来），这是 `TraceJudge` 该拦的方向；
    多报一条是另一条路径——引文站不住就会被 `judges.quote_holds` 当误报丢掉（见 test_judges）。
    """
    tags = [tag for tag in features.TAG_IDS if tag in hits]
    if lie and tags:
        tags = tags[:-1]
    return [{"tag": tag, "quote": quote_for(tag)} for tag in tags]


def spec_for(
    case: ChatCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> ChatGateSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return ChatGateSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(text: str, candidates: list[str], lang: str = "zh") -> dict[str, JsonValue]:
    """输入：消息、三条候选、语言。输出：只带 state 的样本。"""
    draft = render.MessageDraft(text=text, candidates=candidates)
    return {"state": render.to_state(draft, locales.load(lang))}


def faithful(
    case: ChatCase, lang: str = "zh", phrasing: int = 0
) -> tuple[ChatGateSpec, dict[str, JsonValue]]:
    """输入：case、语言、问法编号。输出：(规格, 样本)。预期行为：消息与候选完全兑现计划。"""
    spec = spec_for(case, phrasing)
    text = message_of(spec.plan.hits)
    candidates = candidates_of(spec.plan.reply.flaws)
    render.check_draft(render.MessageDraft(text=text, candidates=candidates), spec.plan, locales.load(lang))
    return spec, sample_of(text, candidates, lang)


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时漏报最后一条命中（验 `trace` 判别器）；
    - `trace_clean_lies` 反解时把「哪条候选干净」说反；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_fragment` 少写一条该命中的片段、`extra_fragment` 多写一条不该命中的片段；
    - `too_short` 把消息缩到一个字符、`fence` 给消息套上围栏；
    - `clean_carries_flaw` 给那条该发的候选也安上一个毛病（验候选毛病那一条闸）。
    """

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        trace_clean_lies: bool = False,
        trace_raises: Exception | None = None,
        drop_fragment: bool = False,
        extra_fragment: bool = False,
        too_short: bool = False,
        fence: bool = False,
        clean_carries_flaw: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies = trace_lies
        self.trace_clean_lies = trace_clean_lies
        self.trace_raises = trace_raises
        self.drop_fragment = drop_fragment
        self.extra_fragment = extra_fragment
        self.too_short = too_short
        self.fence = fence
        self.clean_carries_flaw = clean_carries_flaw
        self.tasks: list[str] = []

    def message_for(self, hits: tuple[str, ...]) -> str:
        """输入：计划里的命中。输出：按开关写歪之后的消息。"""
        tags = list(hits)
        if self.drop_fragment and tags:
            tags = tags[:-1]
        if self.extra_fragment:
            spare = [tag for tag in features.TAG_IDS if tag not in hits]
            if spare:
                tags = [*tags, spare[0]]
        text = message_of(tuple(tags))
        if self.too_short:
            return "短"
        return "```" + text if self.fence else text

    def flaws_for(self, flaws: tuple[tuple[str, ...], ...], clean: int) -> list[str]:
        """输入：计划的毛病分配、该发的下标。输出：三条候选正文。"""
        planned = list(flaws)
        if self.clean_carries_flaw:
            planned[clean] = ("F1",)
        return candidates_of(tuple(planned))

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        hits = tuple(str(tag) for tag in spec["hits"])
        clean = int(spec["clean"])
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            report = trace_report(hits, lie=self.trace_lies)
            picked = clean + 1
            if self.trace_clean_lies:
                picked = (clean + 1) % len(features.CANDIDATE_SLOTS) + 1
            return _draft(
                {"tags": report, "clean": picked},
                "tags=" + "+".join(item["tag"] for item in report) + f";clean={picked}",
            )
        flaws = tuple(tuple(str(item) for item in group) for group in spec["flaws"])
        text = self.message_for(hits)
        return _draft({"text": text, "candidates": self.flaws_for(flaws, clean)}, text)

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：草稿列表。预期行为：逐个走 `generate`。"""
        return [self.generate(task) for task in tasks]


@pytest.fixture
def backend() -> FakeBackend:
    """输入：无。输出：一个完全兑现计划的假后端。"""
    return FakeBackend()


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
