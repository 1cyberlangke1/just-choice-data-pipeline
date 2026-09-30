"""改动引入性判定测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec` 现造草稿，所以它天然「知道」这一条的场景事实：
- 渲染任务 → 按指向的行号与要引用的符号拼出那条发现，再按修法维拼出修法；
- 反解任务 → 直接把注入的三件事实当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies_cite=True)` 之类的开关故意说反话；
要验「端点故障不算数据问题」时，用 `FakeBackend(trace_raises=TransportError(...))`。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue

#: 发现那一行的写法：**按语言分开**——行号写法是语言相关的（`第 N 行` / `line N`），
#: 夹具写错了就会被 `render.check_draft` 当成「没写清指向哪一行」当场拒掉。
FINDING: Final[dict[str, str]] = {
    "zh": "第 {line} 行：`{symbol}` {tail}",
    "en": "line {line}: `{symbol}` {tail}",
}

#: 每个 `cite` 取值一句发现底稿的尾巴。
TAILS: Final[dict[str, dict[str, str]]] = {
    "zh": {
        "touched": "这里刚改过，但取值没夹紧，配置一放大就会出岔子。",
        "untouched": "这里一直如此，取值没夹紧，配置一放大就会出岔子。",
        "missing": "这处取值没夹紧，配置一放大就会出岔子。",
    },
    "en": {
        "touched": "this was just changed, but the value is not clamped, so a bigger config breaks it.",
        "untouched": "this has always been like that, the value is not clamped, so a bigger config breaks it.",
        "missing": "the value here is not clamped, so a bigger config breaks it.",
    },
}

#: `fix=concrete` 时的修法底稿（点名那个符号）。
FIX_CONCRETE: Final[dict[str, str]] = {
    "zh": "在 `{symbol}` 那一处补一个上限校验。",
    "en": "Add an upper-bound check at `{symbol}`.",
}

#: `fix=vague` 时的修法底稿（一个符号都不点、一个反引号都没有）。
FIX_VAGUE: Final[dict[str, str]] = {
    "zh": "需要进一步排查确认。",
    "en": "Needs further investigation.",
}


def finding_for(tags: dict[str, JsonValue], plan: dict[str, JsonValue], lang: str) -> str:
    """输入：注入特征、场景事实、语言。输出：与三者相称的那条发现。

    预期行为：夹具里没有这一格时直接抛 KeyError——静默兜底会让「文本承载 cite」这条
    前提在测试里悄悄失效。
    """
    return FINDING[lang].format(
        line=plan["cite_line"],
        symbol=plan["evidence_symbol"],
        tail=TAILS[lang][str(tags["cite"])],
    )


def fix_for(tags: dict[str, JsonValue], plan: dict[str, JsonValue], lang: str) -> str:
    """输入：注入特征、场景事实、语言。输出：与修法维相称的那句修法。"""
    if str(tags["fix"]) == "concrete":
        return FIX_CONCRETE[lang].format(symbol=plan["fix_symbol"])
    return FIX_VAGUE[lang]


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意做坏事」的开关，默认全关）：
    - `trace_lies_cite` 反解时把指向的行数说成另一个选项（验 `trace` 判别器）；
    - `trace_lies_symbol` 反解时把符号说成另一个（验同一个）；
    - `trace_lies_actionable` 反解时把「落不落地」说反（验同一个）；
    - `wrong_line` 发现里写一个别的行号（验渲染期的行号检查）；
    - `no_backtick` 发现里不把符号括起来（验渲染期的反引号检查）；
    - `vague_with_backtick` 泛泛的修法里却点了名（验渲染期的修法检查）；
    - `concrete_without_backtick` 点名了却没括起来（验同一个）；
    - `multi_line_finding` 把发现写成两行（验 `check_draft` 与 `hygiene` 判别器）；
    - `fence_finding` 把发现包进 Markdown 围栏（验同一对）；
    - `empty_fix` 把修法写成空白（验同一对）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        trace_lies_cite: bool = False,
        trace_lies_symbol: bool = False,
        trace_lies_actionable: bool = False,
        wrong_line: bool = False,
        no_backtick: bool = False,
        vague_with_backtick: bool = False,
        concrete_without_backtick: bool = False,
        multi_line_finding: bool = False,
        fence_finding: bool = False,
        empty_fix: bool = False,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：十二个作弊开关。输出：假后端实例。"""
        self.trace_lies_cite = trace_lies_cite
        self.trace_lies_symbol = trace_lies_symbol
        self.trace_lies_actionable = trace_lies_actionable
        self.wrong_line = wrong_line
        self.no_backtick = no_backtick
        self.vague_with_backtick = vague_with_backtick
        self.concrete_without_backtick = concrete_without_backtick
        self.multi_line_finding = multi_line_finding
        self.fence_finding = fence_finding
        self.empty_fix = empty_fix
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
        lang = str(task.spec.get("locale", "zh"))
        if task.task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            if self.trace_data is not None:
                return self._draft(dict(self.trace_data))
            return self._draft(self._trace(task.spec["truth"]))
        return self._draft(self._render(tags, task.spec["plan"], lang))

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：与入参顺序一致的草稿列表。"""
        return [self.generate(task) for task in tasks]

    def _render(
        self, tags: dict[str, JsonValue], plan: dict[str, JsonValue], lang: str
    ) -> dict[str, JsonValue]:
        """输入：注入特征、场景事实、语言。输出：`FindingDraft` 形态的字典。"""
        finding = finding_for(tags, plan, lang)
        fix = fix_for(tags, plan, lang)
        if self.wrong_line:
            finding = finding_for(
                {**tags, "cite": str(tags["cite"])},
                {**plan, "cite_line": int(plan["cite_line"]) + 1000},
                lang,
            )
        if self.no_backtick:
            finding = FINDING[lang].format(
                line=plan["cite_line"],
                symbol=plan["evidence_symbol"],
                tail=TAILS[lang][str(tags["cite"])],
            ).replace("`", "")
        if self.vague_with_backtick:
            fix = f"{FIX_VAGUE[lang]} `{plan['fix_symbol']}`"
        if self.concrete_without_backtick:
            fix = FIX_CONCRETE[lang].format(symbol=plan["fix_symbol"]).replace("`", "")
        if self.multi_line_finding:
            finding = f"{finding}\n第二行"
        if self.fence_finding:
            finding = f"```{finding}```"
        if self.empty_fix:
            fix = "   "
        return {"finding": finding, "fix": fix}

    def _trace(self, truth: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """输入：注入的三件事实。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        cite = str(truth["cite"])
        symbol = str(truth["symbol"])
        actionable = bool(truth["actionable"])
        if self.trace_lies_cite:
            cite = "s1" if cite != "s1" else "none"
        if self.trace_lies_symbol:
            symbol = f"{symbol}_wrong"
        if self.trace_lies_actionable:
            actionable = not actionable
        return {"cite": cite, "symbol": symbol, "actionable": actionable}

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
    """工厂夹具：`fake_backend(trace_lies_cite=True)` 造一个假后端。"""
    return FakeBackend
