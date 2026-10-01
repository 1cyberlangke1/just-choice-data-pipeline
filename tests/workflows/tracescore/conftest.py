"""tracescore 的假后端与草稿构造器：脚本化出草稿，不联网。

假后端按 `task.spec` 现造执行轨迹——**默认完全兑现计划**（轨迹里的片段由注入特征推出来），
另外几个开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows._fakes import draft as _draft
from workflows.tracescore import features, locales, render
from workflows.tracescore.features import TraceScoreCase
from workflows.tracescore.spec import TraceScoreSpec

#: 轨迹的骨架：一个命中片段都不含，长度落在 `render.MIN_CONTENT_CHARS` 以上。
#: 语义上它是一段**每一步都在推进**的轨迹（0 条命中那一档要的就是这种东西）。
BASE: Final[str] = (
    "用户要求给配置解析加上超时。我先打开 config.py 找到 load_settings，"
    "把 timeout 参数一路接进请求调用，接着改掉两个调用点，"
    "顺手把默认值写进文档字符串，再把参数从入口一路透传下来，"
    "最后整理了一遍这次改动的说明，收工。"
)

#: 每条信号对应的片段：命中哪一条就往轨迹里拼哪一段。
FRAGMENTS: Final[dict[str, str]] = {
    "S1": " 跑了测试，全部通过",
    "S2": " 又试了一次，还是同样的报错",
    "S3": " 输出显示有两个用例失败",
    "S4": " 又看了一遍同样的日志",
}


def text_of(case: TraceScoreCase) -> str:
    """输入：注入特征。输出：一段完全兑现它的轨迹（片段按 `TAG_IDS` 顺序拼）。"""
    parts = [BASE]
    parts.extend(FRAGMENTS[tag] for tag in features.TAG_IDS if tag in case.tags())
    return "".join(parts)


def quote_for(tag: str) -> str:
    """输入：信号编号。输出：该信号在轨迹里**原样出现**的那一段（引文）。

    预期行为：直接取 `FRAGMENTS` 里那一段（去掉两端空白）——它确实在 `text_of` 的产物里，
    所以反解带上它就站得住（`judges.quote_holds` 会原样比对轨迹）。
    """
    return FRAGMENTS[tag].strip()


def trace_report(case: TraceScoreCase, *, lie: bool = False) -> list[dict[str, str]]:
    """输入：注入特征、要不要说反话。输出：反解草稿的 tags（每条带编号与原样引文）。

    预期行为：`lie=True` 时**漏报**最后一条（读者没读出来），这是 `TraceJudge` 该拦的方向；
    多报一条是另一条路径——引文站不住就会被 `judges.quote_holds` 当误报丢掉（见 test_judges）。
    """
    tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
    if lie and tags:
        tags = tags[:-1]
    return [{"tag": tag, "quote": quote_for(tag)} for tag in tags]


def spec_for(
    case: TraceScoreCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> TraceScoreSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return TraceScoreSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(text: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：轨迹、语言。输出：只带 state 的样本。"""
    return {"state": render.to_state(render.ContentDraft(text=text), locales.load(lang))}


def faithful(case: TraceScoreCase, lang: str = "zh") -> tuple[TraceScoreSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：轨迹完全兑现计划。"""
    spec = spec_for(case)
    text = text_of(case)
    render.check_draft(render.ContentDraft(text=text), case, locales.load(lang))
    return spec, sample_of(text, lang)


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时把命中组合说反（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_fragment` 少写一条该命中的片段、`extra_fragment` 多写一条不该命中的片段；
    - `too_short` 把轨迹缩到一个字符、`fence` 给轨迹套上围栏。
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
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.drop_fragment = drop_fragment
        self.extra_fragment = extra_fragment
        self.too_short = too_short
        self.fence = fence
        self.tasks: list[str] = []

    def text_for(self, case: TraceScoreCase) -> str:
        """输入：注入特征。输出：按开关写歪之后的轨迹。"""
        tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
        if self.drop_fragment and tags:
            tags = tags[:-1]
        if self.extra_fragment:
            spare = [tag for tag in features.TAG_IDS if tag not in case.tags()]
            if spare:
                tags = [*tags, spare[0]]
        parts = [BASE]
        parts.extend(FRAGMENTS[tag] for tag in features.TAG_IDS if tag in tags)
        text = "".join(parts)
        if self.too_short:
            return "短"
        return "```" + text if self.fence else text

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        case = TraceScoreCase(hits=str(spec["case"]["hits"]), source=str(spec["case"]["source"]))
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            report = trace_report(case, lie=self.trace_lies)
            return _draft({"tags": report}, "tags=" + "+".join(item["tag"] for item in report))
        text = self.text_for(case)
        return _draft({"text": text}, text)

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
