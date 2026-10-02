"""结构恢复的假后端与夹具：脚本化出草稿，不联网。

假后端按 `task.spec` 里的 case **重算块计划**，再为每一块现造文字——**默认完全兑现计划**；
另外几个开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。

正文骨架刻意写成**语言中立**的短语，只有**标记类**文案（步骤词、提示框关键词、署名前缀）
从语言表里取：判据认的恰好就是那些标记，所以同一套夹具能同时喂 zh 与 en 两种语言。

反解任务（`-trace`）的假后端**从提示词里的 state 自己读**块结构（真后端也只能这么干），
所以 `TraceJudge` 不必把真值塞进任务里——它想抄也抄不到。
"""

from __future__ import annotations

import re
from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft as _draft
from workflows.autoformat import features, locales, render
from workflows.autoformat.features import DocCase, DocPlan
from workflows.autoformat.spec import DocSpec

#: 标题骨架：不以标点结尾、不含提示框关键词；长度落在两种语言的预算里（最紧的是中文的 6~24）。
HEADING_BODY: Final[str] = "Migration memo"
#: 段落骨架：以句末标点结尾、不含提示框关键词；长度落在两种语言的预算里（74~240）。
PARAGRAPH_BODY: Final[str] = (
    "we agreed to keep the rollout slow and reversible, and to write down every move before "
    "touching production. everyone signed off on the plan last week, so there is no rush at all."
)
#: 列表项骨架：不以标点结尾；有序那一档前面拼一个步骤词，无序那一档直接用（长度落在两种语言的预算里）。
ORDERED_BODY: Final[str] = "back up old config"
BULLET_BODY: Final[str] = "platform and release leads"
#: 引用骨架：正文 + 署名前缀 + 署名（合起来的长度落在两种语言的预算里，最紧的是中文的 24~120）。
QUOTE_BODY: Final[str] = (
    "measure first and automate later, we said that in the team many times but few projects "
    "ever did it"
)
QUOTE_NAME: Final[str] = "Alex"
#: 提示框骨架：前面拼一个本种类的关键词，以句末标点结尾（合起来的长度落在两种语言的预算里）。
CALLOUT_BODY: Final[str] = (
    "run the same commands in staging ahead of time, they behave the same way there and it is "
    "cheap to find out early."
)
#: 命令行块的两个后缀（拼在工具名后面，凑够 2~4 行）。
CODE_SUFFIXES: Final[tuple[str, ...]] = ("--version", "--help")
#: state 的行号形态：`L007| 正文`。
STATE_LINE: Final[re.Pattern[str]] = re.compile(r"^L\d{3}\|")


def texts_for(plan: DocPlan, loc: dict[str, JsonValue]) -> list[str]:
    """输入：块计划、语言字典。输出：完全兑现计划的各块文字（按块序）。

    预期行为：按块类型取骨架，标记类文案从语言表里取；同类块的下标轮转，所以同一份文档里相邻的
    同类块不会写成一模一样（那会让「并段」那一族出现没法判的重复行）。
    """
    out: list[str] = []
    counters: dict[str, int] = {}
    for slot in plan.slots:
        index = counters.get(slot.kind, 0)
        counters[slot.kind] = index + 1
        out.append(_text_of(slot, index, loc))
    return out


def _text_of(slot: features.Slot, index: int, loc: dict[str, JsonValue]) -> str:
    """输入：槽位、同类块序号、语言字典。输出：这一块的忠实文字。"""
    if slot.kind == features.KIND_HEADING:
        return HEADING_BODY if index == 0 else f"{HEADING_BODY} {index + 1}"
    if slot.kind == features.KIND_PARAGRAPH:
        return PARAGRAPH_BODY if index == 0 else f"Part {index + 1}: {PARAGRAPH_BODY}"
    if slot.kind == features.KIND_LIST_ITEM:
        if slot.list_ordered:
            return f"{_steps(loc)[index % len(_steps(loc))]} {ORDERED_BODY}"
        return BULLET_BODY
    if slot.kind == features.KIND_CODE:
        tools = [str(item) for item in loc["code_tool"][slot.code_family]]
        return "\n".join(
            f"{tool} {suffix}" for tool, suffix in zip(tools[: len(CODE_SUFFIXES)], CODE_SUFFIXES)
        )
    if slot.kind == features.KIND_QUOTE:
        return f"{QUOTE_BODY}{_marks(loc)[0]}{QUOTE_NAME}"
    if slot.kind == features.KIND_CALLOUT:
        word = str(loc["callout_word"][slot.callout_kind][0])
        return word + CALLOUT_BODY
    raise AssertionError(f"认不出的块类型：{slot.kind!r}")


def _steps(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：步骤词表。"""
    return tuple(str(item) for item in loc["seq_word"]["step"])


def _marks(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：署名前缀表。"""
    return tuple(str(item) for item in loc["mark"]["attribution"])


def plan_of(task: object) -> DocPlan:
    """输入：渲染任务。输出：它那一份块计划。预期行为：由 `task.spec` 里的 case 重算（与 workflow 同源）。"""
    raw = getattr(task, "spec")["case"]
    case = DocCase(
        topic=str(raw["topic"]),
        wrap=int(raw["wrap"]),
        shape=str(raw["shape"]),
        variant=str(raw["variant"]),
        code_family=str(raw["code_family"]),
    )
    return features.build_plan(case)


def state_of(task: object) -> str:
    """输入：反解任务。输出：提示词里那段 state。预期行为：找不到行号就报错（说明提示词拼错了）。"""
    lines = str(getattr(task, "prompt")).split("\n")
    for index, line in enumerate(lines):
        if STATE_LINE.match(line):
            return "\n".join(lines[index:])
    raise AssertionError("反解提示词里没有 state")


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由任务现造，不联网。

    字段 `locale`：假后端**替模型写字**，而标记类文案（步骤词、提示框关键词、署名前缀）
    是语言相关的，所以它必须知道目标语言。默认 `zh`（与老夹具的行为一致）；真跑 en 时要
    显式传 `locale="en"`，否则它会拿 zh 的标记去喂 en 的判据，被渲染期当场拒掉
    ——`tests/workflows/test_llm_locale_sweep.py` 第一次跑就是这么抓到这里的。

    开关（默认全关，都是「故意说反话」）：

    - `drop_text`：少写一块（验「段数与计划对不上」）；
    - `heading_punct`：标题末尾加句号（验标题的结尾标点）；
    - `short_paragraph`：把段落压到 4 个字（验长度区间）；
    - `callout_wrong_word`：提示框改用别的种类的关键词开头（验提示框种类）；
    - `code_wrong_tool`：命令行块换成不以工具名开头（验代码块判据）；
    - `marker_leak`：标题前面加 `# `（验 `hygiene` 抓排版标记残留）；
    - `trace_lies`：反解时少报最后一块（验 `trace`）；
    - `trace_raises`：反解时抛异常（验「端点故障只算没判成」）。
    """

    def __init__(
        self,
        *,
        locale: str = "zh",
        drop_text: bool = False,
        heading_punct: bool = False,
        short_paragraph: bool = False,
        callout_wrong_word: bool = False,
        code_wrong_tool: bool = False,
        marker_leak: bool = False,
        trace_lies: bool = False,
        trace_raises: Exception | None = None,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.locale = locale
        self.drop_text = drop_text
        self.heading_punct = heading_punct
        self.short_paragraph = short_paragraph
        self.callout_wrong_word = callout_wrong_word
        self.code_wrong_tool = code_wrong_tool
        self.marker_leak = marker_leak
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.tasks: list[str] = []

    def texts(self, plan: DocPlan, loc: dict[str, JsonValue]) -> list[str]:
        """输入：块计划、语言字典。输出：按开关写歪之后的各块文字。"""
        out = [self._bend(slot, text, loc) for slot, text in zip(plan.slots, texts_for(plan, loc), strict=True)]
        if self.drop_text and out:
            out.pop()
        return out

    def _bend(self, slot: features.Slot, text: str, loc: dict[str, JsonValue]) -> str:
        """输入：槽位、忠实文案、语言字典。输出：按开关改坏之后的文案。"""
        if self.heading_punct and slot.kind == features.KIND_HEADING:
            return text + "。"
        if self.short_paragraph and slot.kind == features.KIND_PARAGRAPH:
            return "太短了。"
        if self.marker_leak and slot.kind == features.KIND_HEADING:
            return "# " + text
        if self.code_wrong_tool and slot.kind == features.KIND_CODE:
            return "\n".join("echo " + line for line in text.split("\n"))
        if self.callout_wrong_word and slot.kind == features.KIND_CALLOUT:
            other = next(kind for kind in features.CALLOUT_KINDS if kind != slot.callout_kind)
            own = [str(item) for item in loc["callout_word"][slot.callout_kind]]
            body = text
            for word in own:
                if body.startswith(word):
                    body = body[len(word) :]
                    break
            return str(loc["callout_word"][other][0]) + body
        return text

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按任务现造，不发网络。"""
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        if task_id.endswith("-trace"):
            return self._trace(task)
        plan = plan_of(task)
        texts = self.texts(plan, locales.load(self.locale))
        return _draft({"texts": texts}, f"texts={len(texts)}")

    def _trace(self, task: object) -> Draft:
        """输入：反解任务。输出：草稿。预期行为：**自己从 state 读**块结构，不抄真值。"""
        if self.trace_raises is not None:
            raise self.trace_raises
        doc = render.parse_state(state_of(task))
        kinds = list(doc.kinds())
        if self.trace_lies and kinds:
            kinds = kinds[:-1]
        callouts = doc.callouts()
        items = doc.list_items()
        return _draft(
            {
                "kinds": kinds,
                "callout_kind": callouts[0].callout_kind if callouts else "",
                "list_ordered": items[0].list_ordered if items else False,
            },
            "kinds=" + ",".join(kinds),
        )


def spec_for(case: DocCase, phrasing: int = 0, split: str = "train", seq: int = 1) -> DocSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格（`texts` 还没回填）。"""
    return DocSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        plan=features.build_plan(case),
    )


def filled(case: DocCase, lang: str = "zh") -> DocSpec:
    """输入：case、语言。输出：回填了忠实文字的真值规格。预期行为：先过一遍 `check_draft`。"""
    spec = spec_for(case)
    texts = texts_for(spec.plan, locales.load(lang))
    render.check_draft(spec.plan, texts, locales.load(lang))
    spec.texts = texts
    return spec


@pytest.fixture
def backend() -> FakeBackend:
    """输入：无。输出：一个完全兑现计划的假后端。"""
    return FakeBackend()


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
