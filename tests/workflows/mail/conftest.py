"""邮件 workflow 测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：
- 渲染任务 → 造一封正文里有 / 没有链接的邮件（链接有无与注入一致）；
- 反解任务 → 直接把注入特征当成反解结果（也就是「模型完全答对」）。

测试要验「判别器能抓错」时，用 `FakeBackend(trace_lies=True)` 或
`FakeBackend(force_url=True)` 故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue



from jc.backend.types import Draft
from tests.workflows import _fakes
from tests.workflows._fakes import draft
from workflows.mail import features


class FakeBackend(_fakes.ScriptedBackend):
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
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：四个作弊开关 + 一个原样吐草稿的口子。输出：假后端实例。"""
        self.force_url = force_url
        self.trace_lies = trace_lies
        self.smuggles = smuggles
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
            return self._draft(task, self._trace(tags))
        return self._draft(task, self._render(tags))

    def _render(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`MailDraft` 形态的字典。

        预期行为：链接行按渲染规则写——文字与目标一致时只写一个地址、不一致时**两个都写**
        （零 token 的 `link_text` 判别器就是靠数地址得出这一族的答案）。
        """
        has_link = tags["link"] != features.LINK_NONE or self.force_url
        _, anchor = features.split_link(tags["link"])
        if not has_link:
            link_line = ""
        elif anchor == "mismatch":
            link_line = "https://other.test/verify（实际指向 https://example.test/verify）"
        else:
            link_line = "详情见 https://example.test/a"
        # 链接只写在落款里（正文不许出现网址），落款是独立字段——与渲染规则一致
        # 冒充件：显示名声称的是别家，发件地址与「官方域名」**对不上**——零 token 的
        # `SenderIdentityJudge` 就是读这两个域名得出这一族的答案（夹具得如实模拟）。
        address = (
            "noreply@example-secure.test"
            if tags["sender"] == "forged_brand"
            else "noreply@example.test"
        )
        return {
            "from_name": "测试发件人",
            "from_address": address,
            "subject": "本月账单",
            "body": "请核对本月账单，如有疑问回复本邮件。",
            "signature": "测试发件人",
            # 陌生个人不声称机构身份，「官方域名」字段必须留空（结构上强制，见 render 的检查）
            "official_domain": "" if tags["sender"] == "unknown_personal" else "example.test",
            "link_line": link_line,
        }

    def _trace(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        if self.trace_data is not None:
            return dict(self.trace_data)
        case = features.TagSet(**tags)
        labels = case.label()
        traced = {
            "requested": labels.requested,
            "extra_requests": [],
            "pressure": labels.pressure,
        }
        if self.trace_lies:
            traced["requested"] = "none" if labels.requested != "none" else "credentials"
        if self.smuggles:
            traced["extra_requests"] = [self.smuggles]
        return traced

    @staticmethod
    def _draft(task: object, data: dict[str, JsonValue]) -> Draft:
        """输入：草稿数据。输出：`Draft`。预期行为：记账字段填假值（共用工厂 `_fakes.draft`）。"""
        return draft(data)


fake_backend = _fakes.factory_fixture(FakeBackend)
