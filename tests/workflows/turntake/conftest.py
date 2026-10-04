"""turntake 的假后端与草稿构造器：脚本化出草稿，不联网。

假后端按 `task.spec` 现造用户输入——**默认完全兑现计划**（输入里的片段由注入特征推出来），另外
几个开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。
"""

from __future__ import annotations

from typing import Final

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft as _draft
from workflows.turntake import features, locales, render
from workflows.turntake.features import TurnTakeCase
from workflows.turntake.spec import TurnTakeSpec

#: 输入的骨架：一条命中片段都不含，长度落在 `render.MIN_CONTENT_CHARS` 以上。
#: 语义上它是一段**真的说完了**的用户输入（0 条命中那一档要的就是这种东西）。
BASE: Final[str] = "今天的评审会改到下午三点，会议室还是老地方，你先跟产品那边说一声。"

#: 句中片段（与「整段怎么收尾」无关的两条判据）：命中哪一条就往输入里拼哪一段。
MIDDLE: Final[dict[str, str]] = {
    "T3": "（还有一件事",
    "T4": "，等一下",
}

#: 收尾片段：`T1`（以连接词收尾）与 `T2`（以没落地的标点收尾）都只看**整段的结尾**，
#: 所以它们不能各拼各的，得由这一张表统一决定结尾长什么样。
#: 两条都命中时结尾写成「，然后，」——连接词后面再跟一个逗号，两个形状同时成立。
ENDINGS: Final[dict[tuple[bool, bool], str]] = {
    (False, False): "",
    (True, False): "，然后",
    (False, True): "，",
    (True, True): "，然后，",
}

#: 每条判据在输入里**原样出现**的那一段（反解引文用它，见 `quote_for`）。
QUOTES: Final[dict[str, str]] = {
    "T1": "然后",
    "T2": "，",
    "T3": "（还有一件事",
    "T4": "，等一下",
}


def text_of(tags: tuple[str, ...]) -> str:
    """输入：该命中的判据。输出：一段完全兑现它们的用户输入。

    预期行为：先拼句中片段（按 `TAG_IDS` 顺序），再按「T1 / T2 命中没有」查 `ENDINGS` 定结尾。
    """
    body = BASE
    for tag in ("T3", "T4"):
        if tag in tags:
            body += MIDDLE[tag]
    return body + ENDINGS[("T1" in tags, "T2" in tags)]


def quote_for(tag: str) -> str:
    """输入：判据编号。输出：该判据在输入里**原样出现**的那一段（引文）。

    预期行为：直接取 `QUOTES` 里那一段——它确实在 `text_of` 的产物里，所以反解带上它就站得住
    （`judges.quote_holds` 会原样比对输入，且它自己也满足那条判据）。
    """
    return QUOTES[tag]


def trace_report(case: TurnTakeCase, *, lie: bool = False) -> list[dict[str, str]]:
    """输入：注入特征、要不要说反话。输出：反解草稿的 tags（每条带编号与原样引文）。

    预期行为：`lie=True` 时**漏报**最后一条（读者没读出来），这是 `TraceJudge` 该拦的方向；
    多报一条是另一条路径——引文站不住就会被 `judges.quote_holds` 当误报丢掉（见 test_judges）。
    """
    tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
    if lie and tags:
        tags = tags[:-1]
    return [{"tag": tag, "quote": quote_for(tag)} for tag in tags]


def spec_for(
    case: TurnTakeCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> TurnTakeSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return TurnTakeSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(text: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：用户输入、语言。输出：只带 state 的样本。"""
    return {"state": render.to_state(render.ContentDraft(text=text), locales.load(lang))}


def faithful(case: TurnTakeCase, lang: str = "zh") -> tuple[TurnTakeSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：输入完全兑现计划。"""
    spec = spec_for(case)
    text = text_of(case.tags())
    render.check_draft(render.ContentDraft(text=text), case, locales.load(lang))
    return spec, sample_of(text, lang)


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时把命中组合说反（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_fragment` 少写一条该命中的片段、`extra_fragment` 多写一条不该命中的片段；
    - `too_short` 把输入缩到一个字符、`fence` 给输入套上围栏。
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

    def text_for(self, case: TurnTakeCase) -> str:
        """输入：注入特征。输出：按开关写歪之后的用户输入。"""
        tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
        if self.drop_fragment and tags:
            tags = tags[:-1]
        if self.extra_fragment:
            spare = [tag for tag in features.TAG_IDS if tag not in case.tags()]
            if spare:
                tags = [*tags, spare[0]]
        text = text_of(tuple(tags))
        if self.too_short:
            return "短"
        return "```" + text if self.fence else text

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        case = TurnTakeCase(hits=str(spec["case"]["hits"]), channel=str(spec["case"]["channel"]))
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            report = trace_report(case, lie=self.trace_lies)
            return _draft({"tags": report}, "tags=" + "+".join(item["tag"] for item in report))
        text = self.text_for(case)
        return _draft({"text": text}, text)


backend = _fakes.instance_fixture(FakeBackend)


fake_backend = _fakes.factory_fixture(FakeBackend)
