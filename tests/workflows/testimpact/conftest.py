"""testimpact 的假后端与草稿构造器：脚本化出草稿，不联网。

假后端按 `task.spec` 现造改动——**默认完全兑现计划**（diff 碰的文件由注入特征推出来），另外几个
开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。
"""

from __future__ import annotations

from typing import Final

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft as _draft
from workflows.testimpact import features, locales, render
from workflows.testimpact.features import ImpactCase
from workflows.testimpact.spec import ImpactSpec

#: 中文提交标题的骨架：一句话说清这次改了什么，长度落在 `render.MIN_TITLE_CHARS` 以上。
TITLE_ZH: Final[str] = "修复请求解析的边界判断"

#: 英文提交标题（en 侧的语言判据要能认出它）。
TITLE_EN: Final[str] = "Fix the request parsing boundary check"

#: 每个文件一个 diff 块：块头 + 一行上下文 + 一行删除 + 两行新增。单文件也有 300 字符以上，
#: 稳稳落在 `render.MIN_DIFF_CHARS` 之上。
BLOCK: Final[tuple[str, ...]] = (
    "index 1a2b3c4..5d6e7f8 100644",
    "@@ -12,4 +12,5 @@ def handle(request):",
    "     payload = request.json()",
    "-    if payload is None:",
    '+    if payload is None or "id" not in payload:',
    "         return None",
    '+    request_id = payload["id"]',
    "     return lookup(request_id)",
)


def file_block(path: str) -> list[str]:
    """输入：文件路径。输出：这一个文件的 diff 块（块头 + `---` / `+++` + 块体）。

    预期行为：块头里的两条路径都用同一个文件（`a/路径` 与 `b/路径`），这正是
    `features.HEADER_RE` 与审计读的那一行。
    """
    return [f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}", *BLOCK]


def diff_for(case: ImpactCase) -> str:
    """输入：注入特征。输出：一份完全兑现它的改动（碰到的文件正好是注入的那些）。"""
    lines: list[str] = []
    for pid in case.touched():
        lines.extend(file_block(features.PATH_FILES[pid]))
    return "\n".join(lines)


def title_for(lang: str = "zh") -> str:
    """输入：语言。输出：该语言的提交标题。

    预期行为：语言是模型写的正文，两种语言给两句话——`language_texts` 与审计的语言判据才有东西
    可查。
    """
    return TITLE_ZH if lang == "zh" else TITLE_EN


def quote_for(pid: str) -> str:
    """输入：路径编号。输出：该文件在改动里**原样出现**的那一段（引文）。

    预期行为：直接取 `diff --git` 那一行——它确实在 `diff_for` 的产物里，而且带着 `b/<路径>`，
    所以反解带上它就站得住（`judges.quote_holds` 会原样比对改动）。
    """
    path = features.PATH_FILES[pid]
    return f"diff --git a/{path} b/{path}"


def trace_report(case: ImpactCase, *, lie: bool = False) -> list[dict[str, str]]:
    """输入：注入特征、要不要说反话。输出：反解草稿的 files（每条带路径与原样引文）。

    预期行为：`lie=True` 时**漏报**最后一个文件（读者没读出来），这是 `TraceJudge` 该拦的方向；
    多报一条是另一条路径——引文站不住就会被 `judges.quote_holds` 当误报丢掉（见 test_judges）。
    """
    pids = list(case.touched())
    if lie and pids:
        pids = pids[:-1]
    return [{"file": features.PATH_FILES[pid], "quote": quote_for(pid)} for pid in pids]


def spec_for(
    case: ImpactCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> ImpactSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return ImpactSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(diff: str, title: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：改动、提交标题、语言。输出：只带 state 的样本。"""
    state = render.to_state(render.ContentDraft(title=title, diff=diff), locales.load(lang))
    return {"state": state}


def faithful(
    case: ImpactCase, lang: str = "zh"
) -> tuple[ImpactSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：改动完全兑现计划。"""
    spec = spec_for(case)
    diff = diff_for(case)
    render.check_draft(
        render.ContentDraft(title=title_for(lang), diff=diff), case, locales.load(lang)
    )
    return spec, sample_of(diff, title_for(lang), lang)


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时漏报最后一个文件（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_file` 少写一个文件的 diff 块、`extra_file` 多写一个不在计划里的文件的块；
    - `prose` 把改动写成散文、`fence` 给改动套上围栏、`short_title` 把标题缩到一个字。
    """

    def __init__(
        self,
        *,
        locale: str = "zh",
        trace_lies: bool = False,
        trace_raises: Exception | None = None,
        drop_file: bool = False,
        extra_file: bool = False,
        prose: bool = False,
        fence: bool = False,
        short_title: bool = False,
    ) -> None:
        """输入：语言与各开关。输出：实例。"""
        self.locale = locale
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.drop_file = drop_file
        self.extra_file = extra_file
        self.prose = prose
        self.fence = fence
        self.short_title = short_title
        self.tasks: list[str] = []

    def diff_for(self, case: ImpactCase) -> str:
        """输入：注入特征。输出：按开关写歪之后的改动。"""
        pids = list(case.touched())
        if self.drop_file and len(pids) > 1:
            pids = pids[:-1]
        if self.extra_file:
            spare = [pid for pid in features.PATH_IDS if pid not in case.touched()]
            if spare:
                pids = [*pids, spare[0]]
        if self.prose:
            return "这次改动调整了请求解析里的边界判断，顺手补了一条用例。"
        body = "\n".join(line for pid in pids for line in file_block(features.PATH_FILES[pid]))
        return f"```\n{body}\n```" if self.fence else body

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        case = ImpactCase(
            touch=str(spec["case"]["touch"]), flavor=str(spec["case"]["flavor"])
        )
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            report = trace_report(case, lie=self.trace_lies)
            return _draft({"files": report}, "files=" + "+".join(item["file"] for item in report))
        title = "短" if self.short_title else title_for(self.locale)
        diff = self.diff_for(case)
        return _draft({"title": title, "diff": diff}, title + "\n" + diff)


backend = _fakes.instance_fixture(FakeBackend)


fake_backend = _fakes.factory_fixture(FakeBackend)
