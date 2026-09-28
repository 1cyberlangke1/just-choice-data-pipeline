"""检索结果判读测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：
- 渲染任务 → 按（查询意图 × 是否切题 × 是不是垃圾）挑一段与之相称的查询与结果条目；
- 反解任务 → 直接把注入的三维当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies_kind=True)` / `FakeBackend(fence_title=True)`
之类的开关故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。

夹具里那些标题与摘要**刻意不提来源与时效**：那两格归「来源与时效」那一行管，
文本里再写一遍就会跟它打架——这也是 `render.check_draft` 会拦的写法。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from workflows.search import features

#: 每个查询意图的底稿（切题 + 干净那一格）。另外两维由 `_derive` 叠上去——
#: 三维一共 12 格，逐格手写既啰嗦又容易漏。
BASE: Final[dict[str, dict[str, str]]] = {
    "navigational": {
        "query": "小蓝鲸云控制台 登录",
        "title": "小蓝鲸云控制台 - 登录入口",
        "snippet": "输入账号与密码即可进入控制台。",
    },
    "informational": {
        "query": "为什么构建脚本总是超时",
        "title": "构建脚本超时的三个常见原因",
        "snippet": "缓存没命中时依赖会重新下载，构建时间会明显变长。",
    },
    "transactional": {
        "query": "下载 蓝鲸笔记 桌面版",
        "title": "蓝鲸笔记桌面版下载页",
        "snippet": "选择系统版本后点击下载即可。",
    },
}

#: 跑题时换上的摘要：与查询明显不是同一个话题。
OFF_TOPIC_SNIPPET: Final[str] = "顺带聊聊我们团队去年搬办公室的经过。"

#: 垃圾内容时换上的标题与摘要：关键词堆砌 + 导流。
SPAM_TITLE: Final[str] = "构建超时怎么办_构建超时原因_构建超时解决_构建超时大全"
SPAM_SNIPPET: Final[str] = "构建超时 构建超时 构建超时 立即点击咨询 限时优惠 名额有限"


def _derive(base: dict[str, str], topicality: str, spam: str) -> dict[str, str]:
    """输入：底稿、是否切题、是不是垃圾。输出：这一格的查询与结果条目。

    预期行为：后两维必须真的改变文本——`judges.TraceJudge` 要反解的就是它们；
    三维同时变时垃圾优先（垃圾文本本来就不切题）。
    """
    out = dict(base)
    if topicality == "off_topic":
        out["snippet"] = OFF_TOPIC_SNIPPET
    if spam == "spam":
        out["title"] = SPAM_TITLE
        out["snippet"] = SPAM_SNIPPET
    return out


#: 三维的 12 格全表（查询意图 × 是否切题 × 是不是垃圾）。
TEXTS: Final[dict[tuple[str, str, str], dict[str, str]]] = {
    (query_kind, topicality, spam): _derive(base, topicality, spam)
    for query_kind, base in BASE.items()
    for topicality in features.TOPICALITIES
    for spam in features.SPAMS
}

def texts_for(tags: dict[str, str]) -> dict[str, str]:
    """输入：注入特征。输出：与之相称的查询与结果条目。

    预期行为：夹具里没有这一格时直接抛 KeyError——静默兜底会让「文本承载三维」这条
    前提在测试里悄悄失效。
    """
    key = (tags["query_kind"], tags["topicality"], tags["spam"])
    return TEXTS[key]


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：
    - `trace_lies_kind` 反解时把「查询意图」说成另一个取值（验 `trace` 判别器）；
    - `trace_lies_spam` 反解时把「是不是垃圾」说成另一个取值（验同一判别器的第二维）；
    - `multi_line_title` 把标题写成两行（验 `check_draft` 与 `hygiene` 判别器）；
    - `fence_title` 把标题包进 Markdown 围栏（验 `check_draft` 与 `hygiene` 判别器）；
    - `empty_title` 把标题写成空白（验 `check_draft` 与 `hygiene` 判别器）；
    - `label_title` 在标题开头再写一遍「标题：」（验渲染期的段落名残留检查）；
    - `source_title` 把来源词写进标题（验渲染期的「标题不许提来源」检查）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        trace_lies_kind: bool = False,
        trace_lies_spam: bool = False,
        multi_line_title: bool = False,
        fence_title: bool = False,
        empty_title: bool = False,
        label_title: bool = False,
        source_title: bool = False,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：九个作弊开关。输出：假后端实例。"""
        self.trace_lies_kind = trace_lies_kind
        self.trace_lies_spam = trace_lies_spam
        self.multi_line_title = multi_line_title
        self.fence_title = fence_title
        self.empty_title = empty_title
        self.label_title = label_title
        self.source_title = source_title
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
        """输入：注入特征。输出：`SearchDraft` 形态的字典。"""
        texts = dict(texts_for(tags))
        if self.multi_line_title:
            texts["title"] = f"{texts['title']}\n第二行"
        if self.fence_title:
            texts["title"] = f"```{texts['title']}```"
        if self.empty_title:
            texts["title"] = "   "
        if self.label_title:
            texts["title"] = f"标题：{texts['title']}"
        if self.source_title:
            texts["title"] = f"官方网站 - {texts['title']}"
        return {"query": texts["query"], "title": texts["title"], "snippet": texts["snippet"]}

    def _trace(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        out: dict[str, JsonValue] = {
            "query_kind": tags["query_kind"],
            "topicality": tags["topicality"],
            "spam": tags["spam"],
        }
        if self.trace_lies_kind:
            out["query_kind"] = next(
                value for value in features.QUERY_KINDS if value != tags["query_kind"]
            )
        if self.trace_lies_spam:
            out["spam"] = next(value for value in features.SPAMS if value != tags["spam"])
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
    """工厂夹具：`fake_backend(trace_lies_kind=True)` 造一个假后端。"""
    return FakeBackend