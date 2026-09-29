"""citation 的假后端与草稿构造器：脚本化出草稿，不联网。

假后端按 `task.spec` 现造短文——**默认完全兑现计划**（正文与计划无关，引文由代码剪），
另外几个开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from workflows.citation import features, locales, render
from workflows.citation.spec import CitationSpec

#: 短文的标题。
TITLE: Final[str] = "Pump maintenance log"

#: 六句话：每句都有具体数字、都够长（`render.MIN_SENTENCE_CHARS`）、两两不重复，而且**前半句
#: 也两两不同**——`partial` 截的就是前半句，撞上就会被「引文该只出在第 n 节」判掉。
SENTENCES: Final[tuple[str, ...]] = (
    "The intake pump logged a pressure of 41 kPa on the first run.",
    "A second sensor reported the coolant flow as 12 litres per minute.",
    "Operators replaced the gasket after 7 failed leak checks.",
    "Spare rotor stock arrived with a measured runout of 88 microns.",
    "Night shift recorded a vibration peak of 23 hertz at the bearing.",
    "Final inspection found 64 loose fasteners on the frame.",
)


def draft_data(
    *,
    duplicate: bool = False,
    drop_sentence: bool = False,
    no_digit: bool = False,
    short_sentence: bool = False,
    fence_line: bool = False,
    title_fence: bool = False,
) -> dict[str, JsonValue]:
    """输入：六个「故意写歪」的开关。输出：草稿字典。

    预期行为：默认**完全兑现计划**；开关各自破坏一件事——`duplicate` 让两句一模一样、
    `drop_sentence` 少给一句、`no_digit` 把一句里的数字全删掉、`short_sentence` 让一句短于
    下限、`fence_line` 让一句带围栏、`title_fence` 让标题带围栏。
    """
    sentences = list(SENTENCES)
    if duplicate:
        sentences[1] = sentences[0]
    if drop_sentence:
        sentences = sentences[:-1]
    if no_digit:
        sentences[2] = "Operators replaced the gasket after several leak checks."
    if short_sentence:
        sentences[4] = "Peak hit 23 Hz."
    if fence_line:
        sentences[0] = "```" + sentences[0]
    title = "```json " + TITLE if title_fence else TITLE
    return {"title": title, "sentences": sentences}


def state_of(
    spec: CitationSpec, data: dict[str, JsonValue], lang: str = "zh"
) -> dict[str, JsonValue]:
    """输入：规格、草稿数据、语言。输出：只带 state 的样本。

    预期行为：用 `model_construct` 绕开 pydantic 的字段校验——这里要造的正是**不合法**的草稿
    （少一句之类），走正常构造会被 pydantic 先拦下来，测试就验不到判别器那一层了。
    """
    draft = render.CitationDraft.model_construct(
        title=str(data["title"]), sentences=[str(item) for item in data["sentences"]]
    )
    return {"state": render.to_state(draft, spec.plan, locales.load(lang))}


def spec_for(
    case: features.CitationCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> CitationSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return CitationSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def faithful(case: features.CitationCase, lang: str = "zh") -> tuple[CitationSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：草稿完全兑现计划。"""
    spec = spec_for(case)
    data = draft_data()
    render.check_draft(
        render.CitationDraft(
            title=str(data["title"]), sentences=[str(item) for item in data["sentences"]]
        ),
        spec.plan,
        locales.load(lang),
    )
    return spec, state_of(spec, data, lang)


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时把「在不在文档里」与「出自第几节」说反（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - 其余见 `draft_data`。
    """

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        trace_raises: Exception | None = None,
        duplicate: bool = False,
        drop_sentence: bool = False,
        no_digit: bool = False,
        short_sentence: bool = False,
        fence_line: bool = False,
        title_fence: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.duplicate = duplicate
        self.drop_sentence = drop_sentence
        self.no_digit = no_digit
        self.short_sentence = short_sentence
        self.fence_line = fence_line
        self.title_fence = title_fence
        self.tasks: list[str] = []

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            form = str(spec["form"])
            in_doc = form != features.FORM_ALTERED
            section = int(spec["true_section"]) + 1 if in_doc else None
            if self.trace_lies:
                in_doc = not in_doc
                section = None if in_doc else 1
            return _draft({"in_doc": in_doc, "section": section}, f"in_doc={in_doc}")
        data = draft_data(
            duplicate=self.duplicate,
            drop_sentence=self.drop_sentence,
            no_digit=self.no_digit,
            short_sentence=self.short_sentence,
            fence_line=self.fence_line,
            title_fence=self.title_fence,
        )
        return _draft(data, "\n".join(str(item) for item in data["sentences"]))

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
