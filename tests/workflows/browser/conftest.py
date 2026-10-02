"""浏览器 / GUI 操作测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：
- 渲染任务 → 按意图挑一句相称的话，并把 `features.wanted_label` 原样写进去；
- 反解任务 → 直接把注入的意图、目标行行号与目标元素名当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies_intent=True)` / `FakeBackend(missing_label=True)`
之类的开关故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。

夹具里那几句话**刻意不提界面状态、也不提行号**：那些归元素表管，写进那句话就会跟它打架
——这也是 `render.check_draft` 会拦的写法。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft
from workflows.browser import features
from workflows.browser.features import TagSet

#: 每一档意图的底稿（中文），`{label}` 会换成注入的目标元素名。
REQUESTS_ZH: Final[dict[str, str]] = {
    "fill": "帮我把{label}填一下，别的先别动。",
    "open": "我想看看{label}，麻烦打开它。",
    "switch": "切到{label}那一页去。",
    "submit": "确认没问题了，点一下{label}。",
}

#: 同上，英文那一份。
REQUESTS_EN: Final[dict[str, str]] = {
    "fill": "Please fill in the {label} for me.",
    "open": "I'd like to open the {label}.",
    "switch": "Switch me over to the {label}.",
    "submit": "Looks good, click the {label} for me.",
}


def request_for(tags: dict[str, str], lang: str = "zh") -> str:
    """输入：注入特征、语言。输出：与特征相称的那句话。

    预期行为：夹具里没有这一档时直接抛 KeyError——静默兜底会让「那句话承载意图」这条
    前提在测试里悄悄失效。
    """
    table = REQUESTS_ZH if lang == "zh" else REQUESTS_EN
    label = features.wanted_label(TagSet(**tags))
    return table[tags["intent"]].format(label=label)


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：
    - `lang` 写哪一国的话（与管线要跑的语言一致）；
    - `trace_lies_intent` 反解时把意图说成另一个取值（验 `trace` 判别器）；
    - `trace_lies_row` 反解时把行号说错（验同一判别器的第二维）；
    - `trace_lies_label` 反解时把元素名说成另一个（验第三维）；
    - `multi_line_request` 把那句话写成两行（验 `check_draft` 与 `hygiene` 判别器）；
    - `fence_request` 把那句话包进 Markdown 围栏（验 `check_draft` 与 `hygiene` 判别器）；
    - `empty_request` 把那句话写成空白（验 `check_draft` 与 `hygiene` 判别器）；
    - `label_request` 在那句话开头再写一遍「用户请求：」（验渲染期的段落名残留检查）；
    - `state_request` 把界面状态写进那句话（验渲染期的「那句话不许提状态」检查）；
    - `missing_label` 把那句话里的元素名换掉（验渲染期的「必须原样出现元素名」检查）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        lang: str = "zh",
        trace_lies_intent: bool = False,
        trace_lies_row: bool = False,
        trace_lies_label: bool = False,
        multi_line_request: bool = False,
        fence_request: bool = False,
        empty_request: bool = False,
        label_request: bool = False,
        state_request: bool = False,
        missing_label: bool = False,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：十三个作弊开关。输出：假后端实例。"""
        self.lang = lang
        self.trace_lies_intent = trace_lies_intent
        self.trace_lies_row = trace_lies_row
        self.trace_lies_label = trace_lies_label
        self.multi_line_request = multi_line_request
        self.fence_request = fence_request
        self.empty_request = empty_request
        self.label_request = label_request
        self.state_request = state_request
        self.missing_label = missing_label
        self.trace_raises = trace_raises
        self.trace_data = trace_data
        self.calls: list[str] = []

    def generate(self, task: object) -> Draft:
        """输入：一个任务。输出：草稿。

        预期行为：按任务 id 后缀区分「渲染」与「反解」；任务里没带 tags 就报错，
        免得测试因为夹具写错而静默通过。
        """
        self.calls.append(task.task_id)
        tags = task.spec.get("tags")
        assert tags, f"假后端要求任务带上 tags：{task.task_id}"
        if task.task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            if self.trace_data is not None:
                return self._draft(dict(self.trace_data))
            return self._draft(self._trace(tags))
        return self._draft(self._render(tags))

    def _render(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`BrowserDraft` 形态的字典。

        预期行为：那句话必须与意图对得上、并原样写出目标元素名（`trace` 判别器就靠这个
        得出答案）；默认写法单行、无围栏、不提状态。
        """
        body = request_for(tags, self.lang)
        if self.missing_label:
            body = body.replace(features.wanted_label(TagSet(**tags)), "那个东西")
        if self.state_request:
            # 只收高精度禁用语，所以这里必须写**清单里那条**说法（`置灰`），
            # 写「那个按钮点不动」这种不在清单里的说法，渲染期本来就不会拦。
            body = f"{body}（那个按钮是置灰的）"
        if self.label_request:
            body = f"用户请求：{body}"
        if self.fence_request:
            body = f"```{body}```"
        if self.multi_line_request:
            body = f"{body}\n（后面还有一句）"
        if self.empty_request:
            body = "   "
        return {"request": body}

    def _trace(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        case = TagSet(**tags)
        out: dict[str, JsonValue] = {
            "intent": tags["intent"],
            "row": features.wanted_index(case),
            "label": features.wanted_label(case),
        }
        if self.trace_lies_intent:
            out["intent"] = next(value for value in features.INTENTS if value != tags["intent"])
        if self.trace_lies_row:
            out["row"] = int(out["row"]) + 1
        if self.trace_lies_label:
            label = str(out["label"])
            out["label"] = next(
                other
                for pool in features.LABELS.values()
                for other in pool
                if other != label
            )
        return out

    @staticmethod
    def _draft(data: dict[str, JsonValue]) -> Draft:
        """输入：草稿数据。输出：`Draft`。预期行为：记账字段填假值（共用工厂 `_fakes.draft`）。"""
        return draft(data)


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies_intent=True)` 造一个假后端。"""
    return FakeBackend