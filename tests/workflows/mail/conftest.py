"""邮件 workflow 测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：
- 渲染任务 → 造一封正文里有 / 没有链接的邮件（链接有无与注入一致）；
- 反解任务 → 直接把注入特征当成反解结果（也就是「模型完全答对」）。

测试要验「判别器能抓错」时，用 `FakeBackend(trace_lies=True)` 或
`FakeBackend(force_url=True)` 故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。
"""

from __future__ import annotations

from typing import Any

import pytest

from jc.backend.types import Draft
from workflows.mail import features


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段：`force_url` 渲染时无视注入特征硬塞一个链接（用来验 `link` 判别器）；
    `trace_lies` 反解时把 `requested` 说成另一个取值（用来验 `trace` 判别器）；
    `smuggles` 反解时报出一个「清单之外的索取」（用来验夹带检测）；
    `trace_raises` 反解时直接抛异常（用来验端点故障被记成跳过而不是拒绝）；
    `trace_data` 反解时原样吐这份草稿（用来验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        force_url: bool = False,
        trace_lies: bool = False,
        smuggles: str | None = None,
        trace_raises: Exception | None = None,
        trace_data: dict[str, Any] | None = None,
    ) -> None:
        """输入：四个作弊开关 + 一个原样吐草稿的口子。输出：假后端实例。"""
        self.force_url = force_url
        self.trace_lies = trace_lies
        self.smuggles = smuggles
        self.trace_raises = trace_raises
        self.trace_data = trace_data
        self.calls: list[str] = []

    def generate(self, task: Any) -> Draft:
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
            return self._draft(task, self._trace(tags))
        return self._draft(task, self._render(tags))

    def generate_many(self, tasks: list[Any]) -> list[Draft]:
        """输入：任务列表。输出：与入参顺序一致的草稿列表。"""
        return [self.generate(task) for task in tasks]

    def _render(self, tags: dict[str, str]) -> dict[str, Any]:
        """输入：注入特征。输出：`MailDraft` 形态的字典。"""
        has_link = tags["link"] != features.LINK_NONE or self.force_url
        # 链接只写在落款里（正文不许出现网址），落款是独立字段——与渲染规则一致
        return {
            "from_name": "测试发件人",
            "from_address": "noreply@example.test",
            "subject": "本月账单",
            "body": "请核对本月账单，如有疑问回复本邮件。",
            "signature": "测试发件人",
            # 陌生个人不声称机构身份，「官方域名」字段必须留空（结构上强制，见 render 的检查）
            "official_domain": "" if tags["sender"] == "unknown_personal" else "example.test",
            "link_line": "详情见 https://example.test/a" if has_link else "",
        }

    def _trace(self, tags: dict[str, str]) -> dict[str, Any]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        if self.trace_data is not None:
            return dict(self.trace_data)
        case = features.TagSet(**tags)
        labels = case.label()
        traced = {
            "requested": labels.requested,
            "extra_requests": [],
            "pressure": labels.pressure,
            "link_text": labels.link_text,
            "sender_identity": labels.sender_identity,
        }
        if self.trace_lies:
            traced["requested"] = "none" if labels.requested != "none" else "credentials"
        if self.smuggles:
            traced["extra_requests"] = [self.smuggles]
        return traced

    @staticmethod
    def _draft(task: Any, data: dict[str, Any]) -> Draft:
        """输入：任务、草稿数据。输出：`Draft`。预期行为：填满必填的记账字段。"""
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
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
