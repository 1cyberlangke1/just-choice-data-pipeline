"""改动审查测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：

- 渲染任务 → 按（有没有凭据字面量 × 有没有调试输出 × 变更类型 × 规模）拼出几行改动；
- 反解任务 → 直接把注入的两维当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies_secret=True)` / `FakeBackend(fence_body=True)`
之类的开关故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。

夹具里那几行改动**刻意不写表名**（`改动清单` / `审查规则` / …）：那些是段落结构，
正文里再写一遍就会跟段落头打架——这也是 `render.check_draft` 会拦的写法。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft
from workflows.review import features

#: 各规模的**目标行数**（取区间下界；`modified` 与两维特征都可能把它顶上去）。
BASE_LINES: Final[dict[str, int]] = {"small": 1, "medium": 4, "large": 7}


def payload_lines(tags: dict[str, str]) -> list[str]:
    """输入：注入特征。输出：承载两维特征的**新增行**（没有特征就给空列表）。

    预期行为：凭据写成带引号的字面量、调试输出写成 logger.debug(，好让反解判别器一眼看出来。
    """
    out: list[str] = []
    if tags["secret"] == "yes":
        out.append('+ db_password = "hunter2"')
    if tags["debug"] == "yes":
        out.append('+ logger.debug("state=%s", state)')
    return out


def _filler(index: int, prefix: str) -> str:
    """输入：行号、前缀。输出：一行无害的改动（用来把行数补到区间里）。"""
    return f"{prefix} value_{index} = compute({index})"


def body_for(tags: dict[str, str]) -> str:
    """输入：注入特征。输出：与之相称的改动正文（多行）。

    预期行为：行数落在 `features.SIZE_LINES[规模]` 里、每行都带前缀、前缀与变更类型不打架；
    `修改` 这一档同时有新增行与删除行。夹具里没有这一格时直接抛 KeyError——静默兜底会让
    「正文承载两维」这条前提在测试里悄悄失效。
    """
    size = tags["size"]
    kind = tags["change_kind"]
    if size not in BASE_LINES or kind not in features.CHANGE_KINDS:
        raise KeyError(f"夹具没有这一格：{size!r} / {kind!r}")
    payload = payload_lines(tags)
    if kind == "deleted":
        lines = [_filler(index, "-") for index in range(BASE_LINES[size])]
    elif kind == "added":
        lines = list(payload)
        while len(lines) < BASE_LINES[size]:
            lines.append(_filler(len(lines), "+"))
    else:
        # 修改：至少一行删除、至少一行新增；两维特征写在新增行上
        need = max(BASE_LINES[size], len(payload) + 1, 2)
        lines = list(payload) + [_filler(len(payload), "-")]
        while len(lines) < need:
            lines.append(_filler(len(lines), "+"))
    return "\n".join(lines)


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies_secret` 反解时把「有没有凭据字面量」说成另一个取值（验 `trace` 判别器）；
    - `trace_lies_debug` 反解时把「有没有调试输出」说反（验同一判别器的第二维）；
    - `fence_body` 把改动包进 Markdown 围栏（验 `check_draft` 与 `hygiene` 判别器）；
    - `empty_body` 把改动写成空白（验同一对）；
    - `no_prefix_body` 去掉每行的 `+` 前缀（验同一对）；
    - `label_body` 在改动开头再写一遍「改动内容：」（验渲染期的段落名残留检查）；
    - `table_body` 把表名写进改动（验渲染期的「正文不许写表名」检查）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        trace_lies_secret: bool = False,
        trace_lies_debug: bool = False,
        fence_body: bool = False,
        empty_body: bool = False,
        no_prefix_body: bool = False,
        label_body: bool = False,
        table_body: bool = False,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：九个作弊开关。输出：假后端实例。"""
        self.trace_lies_secret = trace_lies_secret
        self.trace_lies_debug = trace_lies_debug
        self.fence_body = fence_body
        self.empty_body = empty_body
        self.no_prefix_body = no_prefix_body
        self.label_body = label_body
        self.table_body = table_body
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
        """输入：注入特征。输出：`ReviewDraft` 形态的字典。

        预期行为：改动正文必须与「有没有凭据字面量」和「有没有调试输出」都对得上
        （`trace` 判别器就靠这个得出答案）；默认写法每行都带前缀、不写表名。
        """
        body = body_for(tags)
        if self.no_prefix_body:
            body = "\n".join(line[1:].lstrip() for line in body.splitlines())
        if self.table_body:
            # 只收表名，所以这里必须写**清单里那条**说法（`改动清单`），
            # 写「这次改了什么」这种不在清单里的说法，渲染期本来就不会拦。
            body = f"{body}\n+ # 见改动清单"
        if self.label_body:
            body = f"改动内容：{body}"
        if self.fence_body:
            body = f"```\n{body}\n```"
        if self.empty_body:
            body = "   "
        return {"body": body}

    def _trace(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        out: dict[str, JsonValue] = {"secret": tags["secret"], "debug": tags["debug"]}
        if self.trace_lies_secret:
            out["secret"] = next(value for value in features.SECRETS if value != tags["secret"])
        if self.trace_lies_debug:
            out["debug"] = next(value for value in features.DEBUGS if value != tags["debug"])
        return out

    @staticmethod
    def _draft(data: dict[str, JsonValue]) -> Draft:
        """输入：草稿数据。输出：`Draft`。预期行为：记账字段填假值（共用工厂 `_fakes.draft`）。"""
        return draft(data)


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies_secret=True)` 造一个假后端。"""
    return FakeBackend