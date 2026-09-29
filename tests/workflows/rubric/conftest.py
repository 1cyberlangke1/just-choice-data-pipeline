"""打分与评分测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：

- 渲染任务 → 按（有没有留空占位 × 有没有出处标记 × 篇幅 × 形式 × 有没有数据）拼出几行材料；
- 反解任务 → 直接把注入的两维当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies_placeholders=True)` / `FakeBackend(fence_body=True)`
之类的开关故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。

夹具里那几行材料**刻意不写表名**（`待评材料` / `材料清单` / …）：那些是段落结构，
材料里再写一遍就会跟段落头打架——这也是 `render.check_draft` 会拦的写法。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from workflows.rubric import features

#: 各篇幅的**目标行数**（取区间下界；承载特征的几行会把它顶上去）。
BASE_LINES: Final[dict[str, int]] = {"short": 3, "medium": 6, "long": 10}

#: 无害的填充句：**一个阿拉伯数字都没有**、不含留空占位、不含出处标记、不含表名。
FILLERS: Final[tuple[str, ...]] = (
    "The upload path now abandons a stalled transfer instead of waiting forever.",
    "Retries back off on their own, so a flaky link no longer stalls the whole batch.",
    "The operator sees the same summary as before, only the timing changed.",
    "We kept the old rollback switch, so nobody has to relearn anything.",
    "The change touches one module and leaves the public interface alone.",
    "Reviewers asked for the same wording in the changelog entry.",
)

#: 留空占位那一维的承载行（`features.PLACEHOLDER_MARKS` 里挑一个）。
PLACEHOLDER_LINE: Final[str] = "TODO: fill in the load-test numbers"

#: 出处标记那一维的承载行（`[1]` 里的数字由 `features.has_number` 先抹掉，不算量化数据）。
CITATION_LINE: Final[str] = "The new build is twice as fast as the previous one[1]."

#: 量化数据那一维的承载行。
NUMBERS_LINE: Final[str] = "Latency dropped from 120ms to 80ms on the same hardware."


def payload_lines(tags: dict[str, str]) -> list[str]:
    """输入：注入特征。输出：承载三维特征的行（留空占位 / 出处标记 / 量化数据）。

    预期行为：有就写、没有就不写——`check_draft` 会逐条比对，夹具写反了当场报错。
    """
    out: list[str] = []
    if tags["placeholders"] == "yes":
        out.append(PLACEHOLDER_LINE)
    if tags["citations"] == "yes":
        out.append(CITATION_LINE)
    if tags["data"] == "numbers":
        out.append(NUMBERS_LINE)
    return out


def body_for(tags: dict[str, str]) -> str:
    """输入：注入特征。输出：与之相称的待评材料（多行）。

    预期行为：行数落在 `features.length_line_bounds(篇幅)` 里；「分条」每行都带 `- ` 前缀、
    「行文」一行都不带；两维正文特征有就是有、没有就是没有。夹具里没有这一格时直接抛
    KeyError——静默兜底会让「正文承载两维」这条前提在测试里悄悄失效。
    """
    length = tags["length"]
    form = tags["form"]
    data = tags["data"]
    if length not in BASE_LINES or form not in features.FORMS or data not in features.DATAS:
        raise KeyError(f"夹具没有这一格：{length!r} / {form!r} / {data!r}")
    lines = payload_lines(tags)
    while len(lines) < BASE_LINES[length]:
        lines.append(FILLERS[len(lines) % len(FILLERS)])
    if form == "bulleted":
        lines = [f"- {line}" for line in lines]
    return "\n".join(lines)


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies_placeholders` 反解时把「有没有留空占位」说成另一个取值（验 `trace` 判别器）；
    - `trace_lies_citations` 反解时把「有没有出处标记」说反（验同一判别器的第二维）；
    - `fence_body` 把材料包进 Markdown 围栏（验 `check_draft` 与 `hygiene` 判别器）；
    - `empty_body` 把材料写成空白（验同一对）；
    - `label_body` 在材料开头再写一遍「待评材料：」（验渲染期的段落名残留检查）；
    - `table_body` 把表名写进材料（验渲染期的「材料不许写表名」检查）；
    - `short_body` 把材料砍到一行（验渲染期的行数检查）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        trace_lies_placeholders: bool = False,
        trace_lies_citations: bool = False,
        fence_body: bool = False,
        empty_body: bool = False,
        label_body: bool = False,
        table_body: bool = False,
        short_body: bool = False,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：九个作弊开关。输出：假后端实例。"""
        self.trace_lies_placeholders = trace_lies_placeholders
        self.trace_lies_citations = trace_lies_citations
        self.fence_body = fence_body
        self.empty_body = empty_body
        self.label_body = label_body
        self.table_body = table_body
        self.short_body = short_body
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

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：与入参顺序一致的草稿列表。"""
        return [self.generate(task) for task in tasks]

    def _render(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`ScoreDraft` 形态的字典。

        预期行为：材料必须与「有没有留空占位」和「有没有出处标记」都对得上
        （`trace` 判别器就靠这个得出答案）；默认写法不写表名、行数落在区间里。
        """
        body = body_for(tags)
        if self.table_body:
            # 只收表名，所以这里必须写**清单里那条**说法（`材料清单`），
            # 写「这份材料」这种不在清单里的说法，渲染期本来就不会拦。
            body = f"{body}\n见材料清单"
        if self.label_body:
            body = f"待评材料：{body}"
        if self.short_body:
            body = body.splitlines()[0]
        if self.fence_body:
            body = f"```\n{body}\n```"
        if self.empty_body:
            body = "   "
        return {"body": body}

    def _trace(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        out: dict[str, JsonValue] = {
            "placeholders": tags["placeholders"],
            "citations": tags["citations"],
        }
        if self.trace_lies_placeholders:
            out["placeholders"] = next(
                value for value in features.PLACEHOLDERS if value != tags["placeholders"]
            )
        if self.trace_lies_citations:
            out["citations"] = next(
                value for value in features.CITATIONS if value != tags["citations"]
            )
        return out

    @staticmethod
    def _draft(data: dict[str, JsonValue]) -> Draft:
        """输入：草稿数据。输出：`Draft`。预期行为：填满必填的记账字段。"""
        return Draft(
            data=data,
            raw_text="<fake>",
            backend="fake",
            provider="fake",
            model="fake-model",
            mode="json_schema",
            usage={},
        )


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies_placeholders=True)` 造一个假后端。"""
    return FakeBackend
