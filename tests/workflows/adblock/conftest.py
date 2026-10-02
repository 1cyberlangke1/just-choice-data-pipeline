"""adblock 的假后端与草稿构造器：脚本化出 HTML 片段，不联网。

假后端按 `task.spec` 现造片段——**默认完全兑现计划**（片段里的形状由注入特征推出来），另外几个
开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft as _draft
from workflows.adblock import features, locales, render
from workflows.adblock.features import AdBlockCase
from workflows.adblock.spec import AdBlockSpec

#: 片段骨架：一个正常的内容元素，四条判据一条都不踩，长度也落在 `render.MIN_CONTENT_CHARS` 以上。
#: 语义上它就是「该保留」的那一档要的东西（0 条命中的元素）。
BASE: Final[str] = (
    '<div class="post-body">'
    "<p>今天的评审会改到下午三点，会议室还是老地方，你先跟产品那边说一声。</p>"
    "</div>"
)

#: 每条判据各占一处（class / id 取值、标签之间的文字、资源与投放属性、style 与尺寸属性），
#: 所以它们可以各拼各的、互不遮蔽。命中哪一条就往骨架后面拼哪一段。
FRAGMENTS: Final[dict[str, str]] = {
    "A1": '<div class="ad-slot">推荐阅读</div>',
    "A2": '<span class="note">本栏目为赞助内容</span>',
    "A3": '<img src="https://googlesyndication.com/pixel.gif" alt="">',
    "A4": '<div style="display:none" aria-hidden="true"></div>',
}

#: 每条判据在片段里**原样出现**的那一段（反解引文用它，见 `quote_for`）。
QUOTES: Final[dict[str, str]] = {
    "A1": 'class="ad-slot"',
    "A2": "赞助内容",
    "A3": "googlesyndication",
    "A4": "display:none",
}


def text_of(tags: tuple[str, ...]) -> str:
    """输入：该命中的判据。输出：一段完全兑现它们的 HTML 片段。

    预期行为：按 `TAG_IDS` 顺序把该命中的那几段拼到骨架上——每条判据各占一处，拼在一起也不会
    互相遮蔽，所以按字面算出来的命中组合就是 `tags`。
    """
    body = BASE
    for tag in features.TAG_IDS:
        if tag in tags:
            body += FRAGMENTS[tag]
    return body


def quote_for(tag: str) -> str:
    """输入：判据编号。输出：该判据在片段里**原样出现**的那一段（引文）。

    预期行为：直接取 `QUOTES` 里那一段——它确实在 `text_of` 的产物里，所以反解带上它就站得住
    （`judges.quote_holds` 会原样比对片段，且它自己也满足那条判据）。
    """
    return QUOTES[tag]


def trace_report(case: AdBlockCase, *, lie: bool = False) -> list[dict[str, str]]:
    """输入：注入特征、要不要说反话。输出：反解草稿的 tags（每条带编号与原样引文）。

    预期行为：`lie=True` 时**漏报**最后一条（读者没读出来），这是 `TraceJudge` 该拦的方向；
    多报一条是另一条路径——引文站不住就会被 `judges.quote_holds` 当误报丢掉（见 test_judges）。
    """
    tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
    if lie and tags:
        tags = tags[:-1]
    return [{"tag": tag, "quote": quote_for(tag)} for tag in tags]


def spec_for(
    case: AdBlockCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> AdBlockSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return AdBlockSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(snippet: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：HTML 片段、语言。输出：只带 state 的样本。"""
    return {
        "state": render.to_state(render.ContentDraft(snippet=snippet), locales.load(lang))
    }


def faithful(case: AdBlockCase, lang: str = "zh") -> tuple[AdBlockSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：片段完全兑现计划。"""
    spec = spec_for(case)
    snippet = text_of(case.tags())
    render.check_draft(render.ContentDraft(snippet=snippet), case, locales.load(lang))
    return spec, sample_of(snippet, lang)


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时把命中组合说反（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_fragment` 少写一条该命中的形状、`extra_fragment` 多写一条不该命中的形状；
    - `too_short` 把片段缩到一个字符、`fence` 给片段套上围栏、`no_html` 写一段没有标签的文本。
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
        no_html: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.drop_fragment = drop_fragment
        self.extra_fragment = extra_fragment
        self.too_short = too_short
        self.fence = fence
        self.no_html = no_html
        self.tasks: list[str] = []

    def snippet_for(self, case: AdBlockCase) -> str:
        """输入：注入特征。输出：按开关写歪之后的 HTML 片段。"""
        tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
        if self.drop_fragment and tags:
            tags = tags[:-1]
        if self.extra_fragment:
            spare = [tag for tag in features.TAG_IDS if tag not in case.tags()]
            if spare:
                tags = [*tags, spare[0]]
        snippet = text_of(tuple(tags))
        if self.too_short:
            return "短"
        if self.no_html:
            return "x" * (render.MIN_CONTENT_CHARS + 5)
        return "```" + snippet if self.fence else snippet

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        case = AdBlockCase(hits=str(spec["case"]["hits"]), surface=str(spec["case"]["surface"]))
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            report = trace_report(case, lie=self.trace_lies)
            return _draft({"tags": report}, "tags=" + "+".join(item["tag"] for item in report))
        snippet = self.snippet_for(case)
        return _draft({"snippet": snippet}, snippet)


@pytest.fixture
def backend() -> FakeBackend:
    """输入：无。输出：一个完全兑现计划的假后端。"""
    return FakeBackend()


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend

