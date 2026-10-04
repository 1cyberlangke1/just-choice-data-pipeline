"""debugstep 的假后端与草稿构造器：脚本化出一屏调试会话，不联网。

假后端按 `task.spec` 现造会话——**默认完全兑现计划**（会话里的形状由注入特征推出来），另外几个
开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。

**每条判据各占一整行**（`D1` 只看 `Stopped:` 行、`D2` 只看 `Locals:` 行、`D3` 只看 `Stack:` 行、
`D4` 只看 `Breakpoints:` 行），所以「踩中某一条」就是把它那一行换成对应的形状，四条之间不会
互相遮蔽——这是这一族从设计上避开「两条判据共用一段文字」的做法。
"""

from __future__ import annotations

from typing import Final

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft as _draft
from workflows.debugstep import features, locales, render
from workflows.debugstep.features import DebugStepCase
from workflows.debugstep.spec import DebugStepSpec

#: 五行骨架：一屏平平无奇的暂停（四条判据一条都不踩），长度也落在 `render.MIN_CONTENT_CHARS` 以上。
#: 语义上它就是「该继续跑」的那一档要的东西（0 条命中的会话）。
BASE_LINES: Final[dict[str, str]] = {
    "stopped": "Stopped: pause requested at a user breakpoint (app/main.py:42, in run)",
    "stack": "Stack: run (app/main.py:42)",
    "locals": "Locals: count=3, name=alpha, items=[1, 2]",
    "breakpoints": "Breakpoints: app/main.py:42, app/config.py:17",
    "note": "Note: 手动暂停，先看看当前这一帧里的取值喵",
}

#: 每条判据的「踩中形状」——它**真的落在自己那一行上**，所以四条各占一行、互不遮蔽。
HIT_LINES: Final[dict[str, str]] = {
    "D1": "Stopped: uncaught TypeError raised while loading config (app/config.py:17, in load_config)",
    "D2": "Locals: count=3, name=alpha, cache=None",
    "D3": "Stack: load_config (app/config.py:17) <- run (app/main.py:42) <- main (app/main.py:9)",
    "D4": "Breakpoints: app/main.py:42, app/main.py:42",
}

#: 判据编号 → 它钉在哪一个结构键那一行上。
LINE_KEYS: Final[dict[str, str]] = {
    "D1": "stopped",
    "D2": "locals",
    "D3": "stack",
    "D4": "breakpoints",
}

#: 每条判据在会话里**原样出现**的那一段（反解引文用它，见 `quote_for`）。引文要**连键名一起**
#: 抄全（反解提示词里的口径），所以它自己单拿出来也满足那条判据。
QUOTES: Final[dict[str, str]] = {
    "D1": "Stopped: uncaught TypeError",
    "D2": "Locals: count=3, name=alpha, cache=None",
    "D3": "Stack: load_config (app/config.py:17) <- run (app/main.py:42)",
    "D4": "Breakpoints: app/main.py:42, app/main.py:42",
}


def lines_of(tags: tuple[str, ...]) -> dict[str, str]:
    """输入：该命中的判据。输出：五行会话的取值表（键 → 那一行的全文）。"""
    lines = dict(BASE_LINES)
    for tag in features.TAG_IDS:
        if tag in tags:
            lines[LINE_KEYS[tag]] = HIT_LINES[tag]
    return lines


def render_lines(lines: dict[str, str]) -> str:
    """输入：五行取值表。输出：按 `features.KEY_LINES` 顺序拼起来的会话。"""
    return "\n".join(lines[key] for key in features.KEY_LINES if key in lines)


def session_of(tags: tuple[str, ...]) -> str:
    """输入：该命中的判据。输出：一屏完全兑现它们的调试会话。

    预期行为：按 `TAG_IDS` 顺序把该命中的那几行换成「踩中形状」——每条判据各占一整行，换上去
    也不会互相遮蔽，所以按字面算出来的命中组合就是 `tags`。
    """
    return render_lines(lines_of(tags))


def quote_for(tag: str) -> str:
    """输入：判据编号。输出：该判据在会话里**原样出现**的那一段（引文）。

    预期行为：直接取 `QUOTES` 里那一段——它确实在 `session_of` 的产物里，所以反解带上它就站得住
    （`judges.quote_holds` 会原样比对，且它自己也满足那条判据）。
    """
    return QUOTES[tag]


def trace_report(case: DebugStepCase, *, lie: bool = False) -> list[dict[str, str]]:
    """输入：注入特征、要不要说反话。输出：反解草稿的 tags（每条带编号与原样引文）。

    预期行为：`lie=True` 时**漏报**最后一条（读者没读出来），这是 `TraceJudge` 该拦的方向；
    多报一条是另一条路径——引文站不住就会被 `judges.quote_holds` 当误报丢掉（见 test_judges）。
    """
    tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
    if lie and tags:
        tags = tags[:-1]
    return [{"tag": tag, "quote": quote_for(tag)} for tag in tags]


def spec_for(
    case: DebugStepCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> DebugStepSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return DebugStepSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(session: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：一屏会话、语言。输出：只带 state 的样本。"""
    return {"state": render.to_state(render.ContentDraft(session=session), locales.load(lang))}


def faithful(case: DebugStepCase, lang: str = "zh") -> tuple[DebugStepSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：会话完全兑现计划。"""
    spec = spec_for(case)
    session = session_of(case.tags())
    render.check_draft(render.ContentDraft(session=session), case, locales.load(lang))
    return spec, sample_of(session, lang)


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时把命中组合说反（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_line` 少写一条该命中的形状、`extra_line` 多写一条不该命中的形状；
    - `missing_key` 把 `Note:` 那一行整行抹掉（验「五个结构键必须齐全」那道闸）；
    - `too_short` 把会话缩到一个字符、`fence` 给会话套上围栏。
    """

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        trace_raises: Exception | None = None,
        drop_line: bool = False,
        extra_line: bool = False,
        missing_key: bool = False,
        too_short: bool = False,
        fence: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.drop_line = drop_line
        self.extra_line = extra_line
        self.missing_key = missing_key
        self.too_short = too_short
        self.fence = fence
        self.tasks: list[str] = []

    def session_for(self, case: DebugStepCase) -> str:
        """输入：注入特征。输出：按开关写歪之后的一屏会话。"""
        tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
        if self.drop_line and tags:
            tags = tags[:-1]
        if self.extra_line:
            spare = [tag for tag in features.TAG_IDS if tag not in case.tags()]
            if spare:
                tags = [*tags, spare[0]]
        lines = lines_of(tuple(tags))
        if self.missing_key:
            lines.pop("note")
        if self.too_short:
            return "短"
        session = render_lines(lines)
        return "```" + session if self.fence else session

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        case = DebugStepCase(
            hits=str(spec["case"]["hits"]), runtime=str(spec["case"]["runtime"])
        )
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            report = trace_report(case, lie=self.trace_lies)
            return _draft({"tags": report}, "tags=" + "+".join(item["tag"] for item in report))
        session = self.session_for(case)
        return _draft({"session": session}, session)


backend = _fakes.instance_fixture(FakeBackend)


fake_backend = _fakes.factory_fixture(FakeBackend)
