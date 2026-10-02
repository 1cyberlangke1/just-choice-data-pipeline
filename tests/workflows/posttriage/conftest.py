"""posttriage 的假后端与草稿构造器：脚本化出草稿，不联网。

假后端按 `task.spec` 现造正文——**默认完全兑现计划**（正文里的片段由注入特征推出来），另外几个
开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft as _draft
from workflows.posttriage import features, locales, render
from workflows.posttriage.features import PostTriageCase
from workflows.posttriage.spec import PostTriageSpec

#: 正文的骨架：一个命中片段都不含，长度落在 `render.MIN_CONTENT_CHARS` 以上。
BASE: Final[str] = "今天这行情有点意思，先把自己的看法记下来，回头再看看对不对。"

#: 每条判据对应的片段：命中哪一条就往正文里拼哪一段。
FRAGMENTS: Final[dict[str, str]] = {
    "S1": " 关注 $ABCDE",
    "S2": " 现价 512.30",
    "S3": " 涨了 7.5%",
    "S4": " 详情见 https://example.com/chart",
}

#: 诱饵片段：小写 cashtag，看着像代码，按判据（只认大写）**不算**命中 S1。
DECOY: Final[str] = f" 顺带一提 {features.DECOY_EXAMPLE} 也有人提"


def text_of(case: PostTriageCase) -> str:
    """输入：注入特征。输出：一段完全兑现它的正文（片段按 `SIGNAL_IDS` 顺序拼）。"""
    parts = [BASE]
    parts.extend(
        FRAGMENTS[signal] for signal in features.SIGNAL_IDS if signal in case.hits()
    )
    if case.decoy == features.DECOY_LOWER:
        parts.append(DECOY)
    return "".join(parts)


def quote_for(signal: str) -> str:
    """输入：判据编号。输出：该判据在正文里**原样出现**的那一段（引文）。

    预期行为：直接取 `FRAGMENTS` 里那一段（去掉行首空格）——它确实在 `text_of` 的产物里，
    所以反解带上它就站得住（`judges.quote_holds` 会原样比对正文）。
    """
    return FRAGMENTS[signal].strip()


def trace_report(case: PostTriageCase, *, lie: bool = False) -> list[dict[str, str]]:
    """输入：注入特征、要不要说反话。输出：反解草稿的 signals（每条带编号与原样引文）。

    预期行为：`lie=True` 时**漏报**最后一条（读者没读出来），这是 `TraceJudge` 该拦的方向；
    多报一条是另一条路径——引文站不住就会被 `judges.quote_holds` 当误报丢掉（见 test_judges）。
    """
    signals = [signal for signal in features.SIGNAL_IDS if signal in case.hits()]
    if lie and signals:
        signals = signals[:-1]
    return [{"signal": signal, "quote": quote_for(signal)} for signal in signals]


def spec_for(
    case: PostTriageCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> PostTriageSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return PostTriageSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(text: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：正文、语言。输出：只带 state 的样本。"""
    return {"state": render.to_state(render.ContentDraft(text=text), locales.load(lang))}


def faithful(case: PostTriageCase, lang: str = "zh") -> tuple[PostTriageSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：正文完全兑现计划。"""
    spec = spec_for(case)
    text = text_of(case)
    render.check_draft(render.ContentDraft(text=text), case)
    return spec, sample_of(text, lang)


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时把命中组合说反（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_fragment` 少写一条该命中的片段、`extra_fragment` 多写一条不该命中的片段；
    - `too_short` 把正文缩到一个字符、`fence` 给正文套上围栏；
    - `decoy_missing` 该埋诱饵时不埋、`decoy_extra` 不该埋诱饵时埋上。
    """

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        trace_raises: Exception | None = None,
        drop_fragment: bool = False,
        extra_fragment: bool = False,
        too_short: bool = False,
        fence: bool = False,
        decoy_missing: bool = False,
        decoy_extra: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.drop_fragment = drop_fragment
        self.extra_fragment = extra_fragment
        self.too_short = too_short
        self.fence = fence
        self.decoy_missing = decoy_missing
        self.decoy_extra = decoy_extra
        self.tasks: list[str] = []

    def text_for(self, case: PostTriageCase) -> str:
        """输入：注入特征。输出：按开关写歪之后的正文。"""
        signals = [signal for signal in features.SIGNAL_IDS if signal in case.hits()]
        if self.drop_fragment and signals:
            signals = signals[:-1]
        if self.extra_fragment:
            spare = [signal for signal in features.SIGNAL_IDS if signal not in case.hits()]
            if spare:
                signals = [*signals, spare[0]]
        parts = [BASE]
        parts.extend(FRAGMENTS[signal] for signal in features.SIGNAL_IDS if signal in signals)
        wants_decoy = case.decoy == features.DECOY_LOWER
        if wants_decoy and not self.decoy_missing:
            parts.append(DECOY)
        elif not wants_decoy and self.decoy_extra:
            parts.append(DECOY)
        text = "".join(parts)
        if self.too_short:
            return "短"
        return "```" + text if self.fence else text

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            case = PostTriageCase(
                signals=str(spec["case"]["signals"]), decoy=str(spec["case"]["decoy"])
            )
            report = trace_report(case, lie=self.trace_lies)
            return _draft(
                {"signals": report}, "signals=" + "+".join(item["signal"] for item in report)
            )
        case = PostTriageCase(
            signals=str(spec["case"]["signals"]), decoy=str(spec["case"]["decoy"])
        )
        text = self.text_for(case)
        return _draft({"text": text}, text)


@pytest.fixture
def backend() -> FakeBackend:
    """输入：无。输出：一个完全兑现计划的假后端。"""
    return FakeBackend()


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
